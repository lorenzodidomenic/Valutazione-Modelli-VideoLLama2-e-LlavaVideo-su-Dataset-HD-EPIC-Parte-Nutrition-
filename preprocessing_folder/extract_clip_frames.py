"""
extract_clip_frames.py
----------------------
Estrae i fotogrammi delle clip del task nutrition_change leggendo i video
direttamente dal server di Bristol, senza scaricarli.

Ogni domanda di nutrition_change indica una finestra [start_time, end_time]
(da 1 a 27 secondi) dentro un video lungo decine di minuti: l'input del
benchmark e' la CLIP, non il video intero. Per ogni finestra lo script decodifica
con ffmpeg solo quel tratto (seek preciso al frame, richieste HTTP parziali) e
salva N_FRAMES fotogrammi uniformi al suo interno.

Output, uno per finestra (stesso nome che cerca hdepic_eval.clip_frames):
    <out>/<video_id>_<start ms a 9 cifre>_<end ms a 9 cifre>/frame_000.jpg ... frame_127.jpg

Le clip piu' corte di N_FRAMES fotogrammi (sotto ~4 s a 30 fps) hanno fotogrammi
ripetuti: e' lo stesso comportamento di np.linspace(..., dtype=int) usato dai
loader ufficiali. I modelli poi ne prendono 8 (VideoLLaMA 2) o 32 (LLaVA-Video)
in modo uniforme.

La cartella va caricata su Drive come `frames_nutrition_change`.
Lo script e' riprendibile: le clip gia' complete vengono saltate.

Requisiti: ffmpeg nel PATH (macOS: brew install ffmpeg).

Uso:
    python3 extract_clip_frames.py                      # output ~/Desktop/frames_nutrition_change
    python3 extract_clip_frames.py --out <cartella> --workers 4 --max-side 768
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/Videos"
ANN = Path(__file__).resolve().parent / "nutrition_nutrition_change.json"
N_FRAMES = 128


def hhmmss_to_sec(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def clip_name(video_id, start, end):
    # identico a hdepic_eval.clip_name
    return f"{video_id}_{int(round(start * 1000)):09d}_{int(round(end * 1000)):09d}"


def collect_jobs():
    """-> [(video_id, start_sec, end_sec)] unici richiesti dalle domande."""
    with open(ANN) as f:
        ann = json.load(f)
    jobs = set()
    for item in ann.values():
        for inp in item["inputs"].values():
            jobs.add((inp["id"], hhmmss_to_sec(inp["start_time"]), hhmmss_to_sec(inp["end_time"])))
    return sorted(jobs)


def is_complete(clip_dir, n=N_FRAMES):
    files = list(clip_dir.glob("frame_*.jpg"))
    return len(files) == n and all(f.stat().st_size > 0 for f in files)


def extract_clip(video_id, start, end, out_dir, max_side, n=N_FRAMES, retries=3):
    """Decodifica [start, end) del video remoto e salva n fotogrammi uniformi."""
    name = clip_name(video_id, start, end)
    final = out_dir / name
    if is_complete(final, n):
        return name, "gia' presente", None

    url = f"{BASE_URL}/{video_id.split('-')[0]}/{video_id}.mp4"
    scale = []
    if max_side:
        # riduce solo il lato lungo, mai ingrandisce (i video HD-EPIC sono 1408x1408)
        scale = ["-vf", f"scale='if(gte(iw,ih),min(iw,{max_side}),-2)':'if(gte(iw,ih),-2,min(ih,{max_side}))'"]

    err = None
    for attempt in range(1, retries + 1):
        with tempfile.TemporaryDirectory() as tmp:
            # -ss prima di -i + ricodifica = seek preciso al frame (ffmpeg >= 2.1)
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error",
                   "-ss", f"{start:.3f}", "-i", url, "-t", f"{end - start:.3f}",
                   *scale, "-q:v", "2", f"{tmp}/%05d.jpg"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            decoded = sorted(Path(tmp).glob("*.jpg"))
            if res.returncode != 0 or not decoded:
                err = (res.stderr.strip().splitlines() or ["nessun frame decodificato"])[-1]
                time.sleep(5 * attempt)
                continue

            sel = np.linspace(0, len(decoded) - 1, n, dtype=int)
            partial = out_dir / (name + ".tmp")
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir(parents=True)
            for i, j in enumerate(sel):
                shutil.copyfile(decoded[j], partial / f"frame_{i:03d}.jpg")
            shutil.rmtree(final, ignore_errors=True)
            partial.rename(final)            # la cartella compare solo se completa
            return name, f"{len(decoded)} frame nella clip, {len(set(sel.tolist()))} unici salvati", None
    return name, None, err


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(Path.home() / "Desktop" / "frames_nutrition_change"))
    parser.add_argument("--workers", type=int, default=4, help="clip elaborate in parallelo")
    parser.add_argument("--max-side", type=int, default=768,
                        help="lato lungo massimo in pixel (0 = risoluzione originale). "
                             "I modelli lavorano a 336-384 px, 768 lascia margine anche per i pannelli.")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg non trovato nel PATH (macOS: brew install ffmpeg)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = collect_jobs()
    print(f"{len(jobs)} clip da {len({j[0] for j in jobs})} video, {N_FRAMES} fotogrammi per clip -> {out_dir}\n")

    failed = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(extract_clip, v, s, e, out_dir, args.max_side) for v, s, e in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            name, info, err = fut.result()
            mark = "OK" if err is None else "KO"
            print(f"[{k:2d}/{len(jobs)}] {mark}  {name}  {info or err}  ({time.time() - t0:.0f}s)")
            if err is not None:
                failed[name] = err

    complete = sum(is_complete(out_dir / clip_name(*j)) for j in jobs)
    print(f"\nClip complete: {complete}/{len(jobs)}")
    if failed:
        print("Clip con errori (rilancia lo script per riprovare):")
        for name, err in failed.items():
            print(f"  {name}: {err}")
        sys.exit(1)
    print("Completato. Carica la cartella su Drive come frames_nutrition_change.")


if __name__ == "__main__":
    main()
