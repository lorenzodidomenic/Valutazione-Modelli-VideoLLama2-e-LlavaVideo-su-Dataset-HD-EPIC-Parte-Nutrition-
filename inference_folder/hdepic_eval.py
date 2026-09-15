"""
hdepic_eval - motore di valutazione per HD-EPIC VQA, categoria Nutrition.

Usato da entrambi i notebook di inferenza (VideoLLaMA2 e LLaVA-Video): tenere
prompt, parsing e metriche in un unico posto e' cio' che rende confrontabili i
risultati dei due modelli.

La parte specifica del modello e' tutta in `infer_fn(visual, prompt) -> str`.

Protocollo (fedele al paper HD-EPIC):
  - domande a 5 scelte, risposta come LETTERA (A-E)
  - accuracy di categoria = MEDIA DELLE ACCURACY PER PROTOTIPO
  - le domande con file mancanti o corrotti sono ESCLUSE, non contate sbagliate
"""
import json, os, re, glob, unicodedata
import pandas as pd

VERSION = "1.0"
LETTERS = "ABCDEFGH"

# Valori di riferimento del paper HD-EPIC (Tab. 2, colonna Nutrition, % accuracy)
PAPER_NUTRITION = {
    "VideoLLaMA 2 7B":       32.7,
    "LLaVA-Video 7B":        38.7,
    "Llama 3.2 90B (blind)": 36.7,
    "Gemini Pro":            34.7,
    "Human":                 85.0,
    "Random":                20.0,
}

# task -> (file di annotation, modalita' di input nativa)
TASK_FILES = {
    "video_nutrition_estimation": ("nutrition_video_nutrition_estimation.json", "video"),
    "nutrition_change":           ("nutrition_nutrition_change.json",           "video"),
    "image_nutrition_estimation": ("nutrition_image_nutrition_estimation.json", "images"),
}
VIDEO_TASKS = [t for t, (_, m) in TASK_FILES.items() if m == "video"]


# ============================================================ indici dei file
def build_video_index(roots):
    """Indicizza TUTTI gli mp4 sotto `roots` -> {'sampled': {id: path}, 'full': {id: path}}.

    Indicizzare globalmente invece che cartella-per-task e' importante: lo stesso
    video puo' servire a piu' task ed essere stato scaricato in una sola cartella.
    """
    idx = {"sampled": {}, "full": {}}
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True):
            b = os.path.basename(p)
            if b.endswith("_sampled.mp4"):
                idx["sampled"].setdefault(b[:-len("_sampled.mp4")], p)
            else:
                idx["full"].setdefault(b[:-len(".mp4")], p)
    return idx


def build_panels_index(roots):
    """Indicizza le cartelle di pannelli -> {video_id: [path dei panel ordinati]}.

    Struttura attesa (output di mosaic.py): <root>/<video_id>/<video_id>_panel_0000.jpg
    """
    idx = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if not os.path.isdir(full):
                continue
            panels = sorted(glob.glob(os.path.join(full, "*.jpg")) +
                            glob.glob(os.path.join(full, "*.jpeg")) +
                            glob.glob(os.path.join(full, "*.png")))
            if panels:
                idx.setdefault(d, panels)
    return idx


def resolve_video(video_id, index, prefer="sampled"):
    """-> (path, sorgente) con sorgente in {'sampled','full',None}."""
    order = ("sampled", "full") if prefer == "sampled" else ("full", "sampled")
    for src in order:
        if video_id in index[src]:
            return index[src][video_id], src
    return None, None


def hhmmss_to_sec(t):
    """'00:13:43.393' -> 823.393"""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_annotations(ann_dir, task):
    with open(os.path.join(ann_dir, TASK_FILES[task][0])) as f:
        return json.load(f)


# =================================================================== prompt
ANSWER_INSTRUCTION = "Answer with the option's letter from the given choices directly."

# Prompt di dominio opzionale: NON usarlo per il confronto col paper, che non ne ha uno.
SYSTEM_PROMPT = (
    "You are a precise video analysis assistant. Track ingredients, actions and "
    "weight measurements on digital scales to answer nutrition-related questions. "
    "Base your answer ONLY on the visual evidence provided."
)


def build_prompt(question, choices, system=None):
    """Prompt a lettere A-E. `system=None` = versione minimale, fedele al paper."""
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    head = f"{system}\n\n" if system else ""
    return f"{head}{question}\n{opts}\n{ANSWER_INSTRUCTION}"


NUTRIENT_RE = re.compile(r"\b(?:highest|higher)\s+(calories|carbs|fat|protein)\b", re.I)


def q_type_of(task, question):
    """Tipo di domanda: il nutriente chiesto, oppure il nome del prototipo."""
    m = NUTRIENT_RE.search(question)
    if m:
        return m.group(1).lower()
    return "nutrition_change" if task == "nutrition_change" else "unknown"


# ================================================================== parsing
def _norm(s):
    s = unicodedata.normalize("NFKD", s).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def parse_answer(response, choices):
    """Estrae l'indice della scelta -> (idx, metodo).

    A livelli, dal piu' stretto al piu' permissivo. Il metodo viene salvato nei
    risultati, cosi' e' sempre ispezionabile COME si e' arrivati a quell'indice.
    idx = -1 significa 'risposta non interpretabile'.
    """
    n = len(choices)
    valid = LETTERS[:n]
    s = (response or "").strip()
    if not s:
        return -1, "empty"

    # 1) inizia con la lettera: "C", "C.", "(C)", "C) spring greens"
    #    il lookahead evita di leggere la C di "Calories changed by..."
    m = re.match(r"^\W*\(?([A-Za-z])\)?(?=[\s.,:;)\-]|$)", s)
    if m and m.group(1).upper() in valid:
        return valid.index(m.group(1).upper()), "leading_letter"

    # 2) formula esplicita: "the answer is C", "Option B"
    m = re.search(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Za-z])\)?\b", s, re.I)
    if m and m.group(1).upper() in valid:
        return valid.index(m.group(1).upper()), "answer_is_letter"

    # 3) ha ricopiato il testo di una scelta invece della lettera
    ns = _norm(s)
    hits = [i for i, c in enumerate(choices) if _norm(c) and _norm(c) in ns]
    if len(hits) == 1:
        return hits[0], "choice_text"
    hits = [i for i, c in enumerate(choices) if ns and ns in _norm(c)]
    if len(hits) == 1:
        return hits[0], "choice_text_partial"

    # 4) una sola lettera valida isolata in tutto il testo
    cand = {x.upper() for x in re.findall(r"\b([A-Za-z])\b", s)} & set(valid)
    if len(cand) == 1:
        return valid.index(cand.pop()), "single_letter_in_text"

    # 5) ha risposto con una cifra nonostante il formato a lettere
    m = re.match(r"^\W*(\d)\b", s)
    if m and 1 <= int(m.group(1)) <= n:
        return int(m.group(1)) - 1, "digit_1indexed"

    return -1, "unparsed"


# ================================================= integrita' dati / pre-volo
def is_readable(path, cache=None):
    """Il file esiste ED e' decodificabile?

    Nel dataset scaricato ci sono mp4 da 0 byte e mp4 troncati (download
    interrotti): esistono ma non si aprono. Contarli come risposte sbagliate
    falserebbe l'accuracy, quindi vanno individuati PRIMA di girare.
    """
    if cache is not None and path in cache:
        return cache[path]
    ok = False
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            ok = bool(cap.read()[0])
            cap.release()
        except Exception:
            ok = False
    if cache is not None:
        cache[path] = ok
    return ok


def _required_inputs(item, task, index, panels_index, input_mode, prefer):
    """-> [(video_id, risorsa)] necessaria a rispondere alla domanda.

    `risorsa` e' un path mp4, una lista di path di pannelli, o None se assente.
    """
    modality = TASK_FILES[task][1]
    out = []
    if modality == "images":
        # 5 fotogrammi a timestamp precisi: serve per forza il video integrale
        for key in sorted(item["inputs"], key=lambda k: int(re.search(r"\d+", k).group())):
            inp = item["inputs"][key]
            path, _ = resolve_video(inp["id"], index, prefer="full")
            out.append((inp["id"], path))
    elif input_mode == "panels":
        for inp in item["inputs"].values():
            out.append((inp["id"], (panels_index or {}).get(inp["id"])))
    else:
        for inp in item["inputs"].values():
            path, _ = resolve_video(inp["id"], index, prefer=prefer)
            out.append((inp["id"], path))
    return out


def usable_items(ann_dir, task, index, panels_index=None, input_mode="video",
                 prefer="sampled", cache=None):
    """Divide le domande in (utilizzabili, scartate-con-motivo)."""
    ann = load_annotations(ann_dir, task)
    ok, skipped = {}, {}
    for key, item in ann.items():
        bad = []
        for vid, res in _required_inputs(item, task, index, panels_index,
                                         input_mode, prefer):
            if isinstance(res, list):          # pannelli: ne basta almeno uno
                if not res:
                    bad.append(vid)
            elif not is_readable(res, cache):  # video: deve aprirsi davvero
                bad.append(vid)
        if bad:
            skipped[key] = sorted(set(bad))
        else:
            ok[key] = item
    return ok, skipped


def preflight(ann_dir, index, tasks=None, panels_index=None, input_mode="video",
              prefer="sampled", cache=None):
    """Tabella di copertura reale. Da stampare SEMPRE prima di un run."""
    cache = {} if cache is None else cache
    rows, details = [], {}
    for task in (tasks or list(TASK_FILES)):
        ann = load_annotations(ann_dir, task)
        ok, skipped = usable_items(ann_dir, task, index, panels_index,
                                   input_mode, prefer, cache)
        details[task] = skipped
        rows.append({"task": task,
                     "input": "frames" if TASK_FILES[task][1] == "images" else input_mode,
                     "domande": len(ann),
                     "utilizzabili": len(ok),
                     "scartate": len(skipped),
                     "file_inutilizzabili": len({v for ids in skipped.values() for v in ids})})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"task": "TOTALE", "input": "", "domande": df["domande"].sum(),
                       "utilizzabili": df["utilizzabili"].sum(),
                       "scartate": df["scartate"].sum(), "file_inutilizzabili": ""}
    return df, details


# ======================================================== estrazione frame
def extract_frame(video_path, sec, out_path):
    """Salva il frame all'istante `sec`. Prova decord, poi ripiega su OpenCV.

    Serve per image_nutrition_estimation, dove ogni domanda mostra 5 fotogrammi
    presi da 5 video diversi a timestamp indicati nelle annotation.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        from decord import VideoReader
        from PIL import Image
        vr = VideoReader(video_path, num_threads=1)
        i = min(int(round(sec * vr.get_avg_fps())), len(vr) - 1)
        Image.fromarray(vr[i].asnumpy()).save(out_path, quality=95)
        return out_path
    except Exception:
        pass
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        i = int(round(sec * fps))
        if total:
            i = min(i, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:                                  # seek per indice fallito
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"frame non leggibile a {sec:.3f}s in {video_path}")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    finally:
        cap.release()
    return out_path


def _visual_input(item, task, index, panels_index, input_mode, prefer,
                  frame_extractor, frame_cache):
    """Prepara cio' che va passato al modello -> (input, etichetta sorgente)."""
    if TASK_FILES[task][1] == "images":
        paths = []
        for key in sorted(item["inputs"], key=lambda k: int(re.search(r"\d+", k).group())):
            inp = item["inputs"][key]
            video, _ = resolve_video(inp["id"], index, prefer="full")
            sec = hhmmss_to_sec(inp["time"])
            out = os.path.join(frame_cache, f"{inp['id']}_{int(round(sec*1000)):09d}.jpg")
            if not os.path.exists(out):
                (frame_extractor or extract_frame)(video, sec, out)
            paths.append(out)
        return paths, "frames"
    vid = item["inputs"]["video 1"]["id"]
    if input_mode == "panels":
        return panels_index[vid], "panels"
    path, src = resolve_video(vid, index, prefer=prefer)
    return path, src


# ================================================================ inferenza
def run_task(task, ann_dir, index, infer_fn, out_jsonl, panels_index=None,
             input_mode="video", prefer="sampled", system=None,
             frame_extractor=None, frame_cache=None, limit=None,
             verbose=True, readable_cache=None):
    """Inferenza su un task, con ripresa da dove si era interrotto.

    Ogni risposta viene scritta su `out_jsonl` subito: se Colab si disconnette,
    rilanciare la cella riprende senza rifare le domande gia' fatte.
    """
    ann, skipped = usable_items(ann_dir, task, index, panels_index,
                                input_mode, prefer, readable_cache)
    if frame_cache:
        os.makedirs(frame_cache, exist_ok=True)
    if verbose:
        print(f"  {len(ann)} domande utilizzabili"
              + (f", {len(skipped)} escluse (file mancanti o corrotti)" if skipped else ""))

    done = {}
    if os.path.exists(out_jsonl):
        with open(out_jsonl) as f:
            for line in f:
                r = json.loads(line)
                done[r["key"]] = r
        if verbose and done:
            print(f"  ripresa: {len(done)} risposte gia' presenti")

    results = []
    for n, (key, item) in enumerate(ann.items()):
        if limit and n >= limit:
            break
        if key in done:
            results.append(done[key])
            continue

        question, choices, correct = item["question"], item["choices"], item["correct_idx"]
        rec = {"key": key, "task": task, "input_mode": input_mode,
               "q_type": q_type_of(task, question),
               "video_id": list(item["inputs"].values())[0]["id"],
               "n_choices": len(choices), "correct_idx": correct}
        try:
            visual, src = _visual_input(item, task, index, panels_index, input_mode,
                                        prefer, frame_extractor, frame_cache)
            response = infer_fn(visual, build_prompt(question, choices, system=system))
            idx, method = parse_answer(response, choices)
            rec.update(source=src, response=response, predicted_idx=idx,
                       parse_method=method, error=None)
        except Exception as e:
            rec.update(source=None, response=None, predicted_idx=-1,
                       parse_method="error", error=f"{type(e).__name__}: {e}")

        rec["is_correct"] = bool(rec["predicted_idx"] == correct)
        with open(out_jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
        results.append(rec)

        if verbose:
            mark = "ERR" if rec["error"] else ("OK " if rec["is_correct"] else "KO ")
            shown = (rec["response"] or rec["error"] or "")[:55]
            print(f"  [{n+1}/{len(ann)}] {mark} {key.split('_')[-1]:>3} | "
                  f"pred={rec['predicted_idx']} gt={correct} | "
                  f"{rec['parse_method']:<18} | {shown!r}")

    df = pd.DataFrame(results)
    df.attrs["skipped"] = skipped
    return df


def run_all(tasks, ann_dir, index, infer_fn, out_dir, tag, **kw):
    """Gira piu' task e concatena i risultati."""
    os.makedirs(out_dir, exist_ok=True)
    out, skipped = [], {}
    for task in tasks:
        print(f"\n{'='*70}\n  TASK: {task}   (run: {tag})\n{'='*70}")
        df = run_task(task, ann_dir, index, infer_fn,
                      os.path.join(out_dir, f"{tag}_{task}.jsonl"), **kw)
        skipped[task] = df.attrs.get("skipped", {})
        out.append(df)
    allr = pd.concat(out, ignore_index=True)
    allr.attrs["skipped"] = skipped
    return allr


# ================================================================= metriche
def accuracy_report(df, label="", paper_ref=None, ann_dir=None, show_bias=True):
    """Report allineato al protocollo del paper.

    HD-EPIC Tab. 2 e' la MEDIA DELLE ACCURACY PER PROTOTIPO, non l'accuracy
    aggregata sulle domande: con prototipi da 50/50/100 i due numeri non
    coincidono. Qui vengono stampati entrambi, cosi' il confronto e' esplicito.
    """
    out = {}
    per_task = df.groupby("task")["is_correct"].agg(corrette="sum", valutate="count")
    per_task["accuracy_%"] = (per_task["corrette"] / per_task["valutate"] * 100).round(1)
    if ann_dir:
        per_task["su_totale"] = [len(load_annotations(ann_dir, t)) for t in per_task.index]
    out["per_task"] = per_task
    out["nutrition_paper"] = round(per_task["accuracy_%"].mean(), 1)
    out["nutrition_pooled"] = round(df["is_correct"].mean() * 100, 1)

    print("=" * 70)
    print(f"  {label}   (n = {len(df)} domande valutate)")
    print("=" * 70)
    print(per_task.to_string())
    print(f"\n  Nutrition, metrica del paper (media per prototipo) : {out['nutrition_paper']}%")
    print(f"  Nutrition, accuracy aggregata sulle domande        : {out['nutrition_pooled']}%")
    print(f"  Baseline random                                    : 20.0%")
    if paper_ref in PAPER_NUTRITION:
        ref = PAPER_NUTRITION[paper_ref]
        print(f"  Paper, {paper_ref:<28}: {ref}%   "
              f"(delta {out['nutrition_paper'] - ref:+.1f})")

    inval = df[df["predicted_idx"] < 0]
    print(f"\n  Risposte non interpretate o in errore: {len(inval)}/{len(df)} "
          f"({len(inval)/max(len(df),1)*100:.1f}%)")
    print("  Come e' stata letta la risposta:")
    print("   ", df["parse_method"].value_counts().to_dict())
    out["parse_methods"] = df["parse_method"].value_counts()

    by_type = df.groupby("q_type")["is_correct"].agg(corrette="sum", valutate="count")
    by_type["accuracy_%"] = (by_type["corrette"] / by_type["valutate"] * 100).round(1)
    print("\n  Accuracy per tipo di domanda:")
    print(by_type.to_string())
    out["per_q_type"] = by_type

    if show_bias:
        print("\n  Bias posizionale (% risposte su ciascuna lettera, vs ground truth):")
        bias = {}
        for task, g in df.groupby("task"):
            n = int(g["n_choices"].max())
            pred = g[g["predicted_idx"] >= 0]["predicted_idx"].value_counts(normalize=True)
            gt = g["correct_idx"].value_counts(normalize=True)
            row = (pd.DataFrame({"pred": pred, "gt": gt})
                     .reindex(range(n)).fillna(0) * 100).round(0).astype(int)
            bias[task] = row["pred"]
            fmt = lambda c: "[" + " ".join(f"{v:3d}" for v in row[c]) + "]"
            print(f"    {task:<28} pred(A-E)={fmt('pred')}  gt(A-E)={fmt('gt')}")
        out["bias"] = pd.DataFrame(bias)
    return out


def compare_runs(runs, label="Confronto"):
    """Confronto fra piu' run: {nome: DataFrame} -> tabella accuracy per task."""
    rows = {}
    for name, df in runs.items():
        acc = (df.groupby("task")["is_correct"].mean() * 100).round(1)
        acc["NUTRITION (media prototipi)"] = round(acc.mean(), 1)
        rows[name] = acc
    tab = pd.DataFrame(rows)
    if len(runs) == 2:
        a, b = list(runs)
        tab["delta"] = (tab[b] - tab[a]).round(1)
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(tab.to_string())
    return tab
