"""
extract_image_frames.py
-----------------------
Estrae i fotogrammi del task image_nutrition_estimation leggendo i video
direttamente dal server di Bristol, senza scaricarli (OpenCV fa richieste
HTTP parziali e scarica solo il pezzo di file attorno al timestamp).

I file escono con lo stesso nome che usa la cache del notebook
(hdepic_eval._visual_input): <video_id>_<millisecondi a 9 cifre>.jpg
e vanno caricati nella cartella frames_image_nutrition_estimation su Drive.

Lo script e' riprendibile: i fotogrammi gia' presenti vengono saltati.

Uso:
    python3 extract_image_frames.py                      # output ~/Desktop/frames_image_nutrition_estimation
    python3 extract_image_frames.py --out <cartella> --workers 6
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/Videos"
ANN = Path(__file__).resolve().parent / "nutrition_image_nutrition_estimation.json"


def hhmmss_to_sec(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def frame_name(video_id, sec):
    # identico a hdepic_eval: f"{id}_{int(round(sec*1000)):09d}.jpg"
    return f"{video_id}_{int(round(sec * 1000)):09d}.jpg"


def collect_jobs():
    """-> {video_id: [sec, ...]} con i timestamp unici richiesti dalle domande."""
    with open(ANN) as f:
        ann = json.load(f)
    jobs = defaultdict(set)
    for item in ann.values():
        for inp in item["inputs"].values():
            jobs[inp["id"]].add(hhmmss_to_sec(inp["time"]))
    return {v: sorted(s) for v, s in jobs.items()}


def extract_video(video_id, secs, out_dir, retries=3):
    """Apre il video remoto una volta sola ed estrae tutti i suoi fotogrammi."""
    todo = [s for s in secs if not (out_dir / frame_name(video_id, s)).exists()]
    if not todo:
        return video_id, len(secs), 0, []

    url = f"{BASE_URL}/{video_id.split('-')[0]}/{video_id}.mp4"
    errors = []
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            errors = [f"apertura fallita (tentativo {attempt})"]
            time.sleep(5 * attempt)
            continue
        # stessa conversione secondi -> indice usata da extract_frame nel notebook
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        errors = []
        for sec in list(todo):
            i = int(round(sec * fps))
            if total:
                i = min(i, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
                ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(out_dir / frame_name(video_id, sec)), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                todo.remove(sec)
            else:
                errors.append(f"frame non leggibile a {sec:.3f}s")
        cap.release()
        if not todo:
            break
        time.sleep(5 * attempt)
    return video_id, len(secs), len(todo), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(Path.home() / "Desktop" / "frames_image_nutrition_estimation"))
    parser.add_argument("--workers", type=int, default=4, help="video elaborati in parallelo")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = collect_jobs()
    n_frames = sum(len(s) for s in jobs.values())
    print(f"{len(jobs)} video, {n_frames} fotogrammi unici -> {out_dir}\n")

    failed = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(extract_video, v, s, out_dir) for v, s in jobs.items()]
        for n, fut in enumerate(as_completed(futures), 1):
            vid, tot, missing, errors = fut.result()
            mark = "OK" if not missing else "KO"
            print(f"[{n:2d}/{len(jobs)}] {mark}  {vid}  {tot - missing}/{tot} frame"
                  f"  ({time.time() - t0:.0f}s)")
            if missing:
                failed[vid] = errors

    present = len(list(out_dir.glob("*.jpg")))
    print(f"\nFotogrammi presenti: {present}/{n_frames}")
    if failed:
        print("Video con errori (rilancia lo script per riprovare):")
        for vid, errs in failed.items():
            print(f"  {vid}: {'; '.join(sorted(set(errs)))}")
        sys.exit(1)
    print("Completato. Carica il contenuto della cartella in frames_image_nutrition_estimation su Drive.")


if __name__ == "__main__":
    main()
