"""
ablazione.py - passo 3: l'esperimento dell'ORACOLO, in versione semplice.

L'idea in tre righe:
  - ogni domanda ha 5 opzioni originali, e la risposta giusta e' una di queste
    (guardando il video);
  - aggiungiamo una SESTA opzione, l'oracolo: un ingrediente non presente nel
    video, ma molto piu' alto degli altri per il nutriente chiesto;
  - se il modello sceglie l'oracolo spesso, risponde dal testo (dalle sue
    conoscenze) e non guarda il video.

Questo file fa tre cose:
  1. costruisci_annotazioni()  crea il file delle domande a 6 opzioni;
  2. analizza()                conta quante volte il modello sceglie l'oracolo;
  3. confronta()               stampa una tabella con i risultati dei run.

Il motore di inferenza e' hdepic_eval.py (nella stessa cartella). Non lo tocchiamo.
"""
import copy
import json
import math
import os
import random

# ----------------------------------------------------------- percorsi
# La cartella del test ha sempre la stessa struttura, in locale e su Drive:
#   <cartella>/dati/        oracoli.json, domande originali, annotazioni_oracolo/
#   <cartella>/risultati/   un .jsonl per modello e condizione
# In locale e' hd-epic-annotations/test_oracolo. Su Colab il notebook imposta
# la variabile TEST_ORACOLO_DIR prima di importare questo file.
CARTELLA_ORACOLO = os.environ.get(
    "TEST_ORACOLO_DIR",
    "/Users/lorenzodidomenico/Desktop/PROGETTO DEEP LEARNING 1/hd-epic-annotations/test_oracolo")
DATI = os.path.join(CARTELLA_ORACOLO, "dati")
RISULTATI = os.path.join(CARTELLA_ORACOLO, "risultati")

DOMANDE_ORIGINALI = os.path.join(DATI, "nutrition_video_nutrition_estimation.json")
ORACOLI = os.path.join(DATI, "oracoli.json")
ANNOTAZIONI_ORACOLO = os.path.join(DATI, "annotazioni_oracolo",
                                   "nutrition_video_nutrition_estimation.json")
ANN_ORACOLO_DIR = os.path.join(DATI, "annotazioni_oracolo")   # cartella letta dal motore
TASK = "video_nutrition_estimation"

# ----------------------------------------------------------- regole dell'esperimento
SEED = 0                 # per la posizione casuale dell'oracolo: sempre lo stesso risultato
SOGLIA_TESTO = 0.50      # se il modello sceglie l'oracolo in piu' del 50% dei casi, risponde dal testo
                         # (valore proposto, da confermare col professore)
CASO = 1 / 6             # probabilita' di indovinare a caso con 6 opzioni


# =========================================================== 1. annotazioni a 6 opzioni
def costruisci_annotazioni(seed=SEED):
    """Crea il file delle domande con l'oracolo inserito in una posizione casuale.

    Per ogni domanda:
      - scegliamo a caso (con il seed) in che posizione mettere l'oracolo, da 0 a 5;
      - inseriamo l'oracolo in quella posizione nelle opzioni;
      - spostiamo di 1 la risposta giusta se l'oracolo finisce prima di lei.
    Le domande scartate (senza oracolo) non entrano nel file.
    """
    with open(DOMANDE_ORIGINALI, encoding="utf-8") as f:
        originali = json.load(f)
    with open(ORACOLI, encoding="utf-8") as f:
        oracoli = json.load(f)["oracoli"]

    nuove = {}
    for chiave, o in oracoli.items():
        dom = copy.deepcopy(originali[chiave])
        rng = random.Random(f"{seed}-{chiave}")       # un seed per domanda: non dipende dall'ordine
        posizione = rng.randint(0, len(dom["choices"]))

        dom["choices"].insert(posizione, o["oracolo"])
        dom["metadata"].insert(posizione, {"source": "oracolo",
                                           "val_gap": o["rapporto"],
                                           "ing_id": o["oracolo_ing_id"]})
        if posizione <= dom["correct_idx"]:
            dom["correct_idx"] += 1                   # la risposta giusta e' scivolata di una posizione

        nuove[chiave] = dom

    os.makedirs(os.path.dirname(ANNOTAZIONI_ORACOLO), exist_ok=True)
    with open(ANNOTAZIONI_ORACOLO, "w", encoding="utf-8") as f:
        json.dump(nuove, f, ensure_ascii=False, indent=2)

    print(f"Annotazioni a 6 opzioni: {len(nuove)} domande -> {ANNOTAZIONI_ORACOLO}")
    return nuove


def lettera_oracolo(annotazione):
    """In quale posizione (0-5) sta l'oracolo in questa domanda."""
    for i, m in enumerate(annotazione["metadata"]):
        if m["source"] == "oracolo":
            return i
    raise ValueError("domanda senza oracolo")


# =========================================================== 2. analisi di un run
def wilson(successi, n, z=1.96):
    """Intervallo di confidenza al 95% per una percentuale (Wilson).

    Con poche domande l'intervallo e' largo: e' normale, va riportato sempre.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successi / n
    denom = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / denom
    margine = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centro - margine), min(1.0, centro + margine))


def leggi_risultati(percorso):
    """Legge un file .jsonl prodotto dal motore: una riga per domanda."""
    with open(percorso, encoding="utf-8") as f:
        return [json.loads(riga) for riga in f if riga.strip()]


def analizza(percorso, annotazioni=None):
    """Conta quante volte il modello sceglie l'oracolo, in un run.

    Ritorna un dizionario con:
      n                  domande valutate (senza errori di lettura del video)
      scelte_oracolo     quante volte ha scelto l'oracolo
      tasso_oracolo      scelte_oracolo / n
      ic_95              intervallo di confidenza (Wilson)
      accuratezza        quante volte ha indovinato la risposta giusta (su 6 opzioni)
    """
    if annotazioni is None:
        with open(ANNOTAZIONI_ORACOLO, encoding="utf-8") as f:
            annotazioni = json.load(f)

    righe = [r for r in leggi_risultati(percorso)
             if r["key"] in annotazioni and r["predicted_idx"] >= 0]

    scelte = sum(1 for r in righe if r["predicted_idx"] == lettera_oracolo(annotazioni[r["key"]]))
    corrette = sum(1 for r in righe if r["predicted_idx"] == annotazioni[r["key"]]["correct_idx"])
    n = len(righe)
    return {
        "file": os.path.basename(percorso),
        "n": n,
        "scelte_oracolo": scelte,
        "tasso_oracolo": scelte / n if n else 0.0,
        "ic_95": wilson(scelte, n),
        "accuratezza": corrette / n if n else 0.0,
    }


def verdetto(tasso):
    """Traduce il tasso di scelta dell'oracolo in una frase."""
    if tasso >= SOGLIA_TESTO:
        return "risponde dal TESTO: sceglie l'oracolo piu' di quanto il caso dia"
    return "non sceglie l'oracolo piu' di quanto il caso dia: il video sembra contare"


# =========================================================== 3. esecuzione dei run
def solo_testo(infer_testo):
    """Condizione O1: il modello vede solo la domanda e le opzioni, nessun video.

    Il motore chiama sempre infer_fn(visual, prompt). Qui il video viene scartato:
    al modello arriva solo il prompt. Cosi' O1 usa esattamente lo stesso prompt di O0.
    """
    def infer_fn(visual, prompt):
        return infer_testo(prompt)
    return infer_fn


def nome_file(modello, condizione, prova=False):
    """Nome del .jsonl, come nel motore: <modello>_<condizione>_<task>.jsonl"""
    prefisso = "prova_" if prova else ""
    return f"{modello}_{prefisso}{condizione}_{TASK}.jsonl"


def esegui(modello, condizione, infer_fn, cartella_annotazioni, index, limit=None,
           prova=False, **kw):
    """Esegue un run e salva il risultato in risultati/.

    cartella_annotazioni:
      ANN_ORACOLO_DIR  per O0 e O1 (49 domande a 6 opzioni)
      DATI             per il controllo C0 (le 50 domande originali)
    Se il run si interrompe, rilanciarlo riprende da dove era rimasto.
    """
    import hdepic_eval as H          # importato qui: l'analisi locale non ne ha bisogno
    os.makedirs(RISULTATI, exist_ok=True)
    out = os.path.join(RISULTATI, nome_file(modello, condizione, prova))
    return H.run_task(TASK, cartella_annotazioni, index, infer_fn, out,
                      limit=limit, **kw)


def controllo_c0(percorso_controllo, percorso_baseline):
    """Verifica che l'ambiente Colab dia le stesse risposte del baseline.

    Confronta, domanda per domanda, la lettera scelta nel controllo con quella del
    baseline salvato. Se le risposte coincidono quasi tutte, l'ambiente e' quello giusto.
    """
    base = {r["key"]: r["predicted_idx"] for r in leggi_risultati(percorso_baseline)}
    controllo = leggi_risultati(percorso_controllo)
    uguali = sum(1 for r in controllo if base.get(r["key"]) == r["predicted_idx"])
    print(f"Controllo C0: {uguali}/{len(controllo)} risposte identiche al baseline")
    if uguali != len(controllo):
        print("ATTENZIONE: l'ambiente non coincide col baseline. Prima di continuare, "
              "rifare C0 in questa sessione.")
    return uguali, len(controllo)


# =========================================================== 4. tabella di confronto
def confronta(percorsi):
    """Stampa una tabella con un riga per run.

    percorsi: lista di file .jsonl (es. O0 e O1 per un modello).
    """
    print(f"{'run':<55} {'n':>3} {'oracolo':>8} {'%':>6} {'IC 95%':>16} {'accur.':>7}")
    print("-" * 100)
    for p in percorsi:
        r = analizza(p)
        lo, hi = r["ic_95"]
        print(f"{r['file']:<55} {r['n']:>3} {r['scelte_oracolo']:>8} "
              f"{100 * r['tasso_oracolo']:>5.1f}% {f'{100*lo:.0f}-{100*hi:.0f}%':>16} "
              f"{100 * r['accuratezza']:>6.1f}%")
    print(f"\nCaso: {100 * CASO:.1f}%   soglia per 'risponde dal testo': {100 * SOGLIA_TESTO:.0f}%")


if __name__ == "__main__":
    costruisci_annotazioni()
