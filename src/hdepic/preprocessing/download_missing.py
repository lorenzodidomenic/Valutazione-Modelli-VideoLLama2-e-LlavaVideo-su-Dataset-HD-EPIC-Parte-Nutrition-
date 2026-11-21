"""
download_missing.py
-------------------
Scarica i video mancanti per ciascun task direttamente nella cartella
appropriata, usando ThreadPoolExecutor per download paralleli.

URL base:  https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/Videos/<PXX>/<video_id>.mp4

Uso:
    python3 download_missing.py              # default: 4 worker
    python3 download_missing.py --workers 8
    python3 download_missing.py --dry-run    # stampa cosa scaricherebbe senza farlo
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm.auto import tqdm
    TQDM = True
except ImportError:
    TQDM = False
    print("[INFO] tqdm non disponibile, progresso testuale.")

# ── Costanti ──────────────────────────────────────────────────────────────────
BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/"
BASE_DIR = Path(__file__).resolve().parent.parent  # root del progetto

TASKS = {
    "nutrition_estimation": {
        "ann": BASE_DIR / "preprocessing_folder" / "nutrition_video_nutrition_estimation.json",
        "video_dir": BASE_DIR / "videos_nutrition_estimation",
    },
    "nutrition_change": {
        "ann": BASE_DIR / "preprocessing_folder" / "nutrition_nutrition_change.json",
        "video_dir": BASE_DIR / "videos_nutrition_change",
    },
    "image_nutrition_estimation": {
        "ann": BASE_DIR / "preprocessing_folder" / "nutrition_image_nutrition_estimation.json",
        "video_dir": BASE_DIR / "videos_image_nutrition_estimation",
    },
}


# ── Utilità ───────────────────────────────────────────────────────────────────

def log(msg: str):
    if TQDM:
        tqdm.write(msg)
    else:
        print(msg)


def build_url(video_id: str) -> str:
    """Costruisce la URL del video dato il suo ID (e.g. P06-20240510-090328)."""
    participant = video_id.split("-")[0]          # P06
    rel_path = f"Videos/{participant}/{video_id}.mp4"
    return BASE_URL + urllib.parse.quote(rel_path)


def find_missing(task_name: str, config: dict) -> list[tuple[str, Path]]:
    """Restituisce lista di (video_id, dest_path) da scaricare."""
    ann_path = config["ann"]
    video_dir = config["video_dir"]

    if not ann_path.exists():
        log(f"[WARN] Annotazione non trovata: {ann_path}")
        return []

    with open(ann_path) as f:
        data = json.load(f)

    needed = set()
    for entry in data.values():
        for v in entry["inputs"].values():
            needed.add(v["id"])

    video_dir.mkdir(parents=True, exist_ok=True)
    present = {p.stem for p in video_dir.glob("*.mp4")}
    missing = sorted(needed - present)

    log(f"[{task_name}] {len(needed)} richiesti, {len(present)} presenti, {len(missing)} mancanti")
    return [(vid, video_dir / f"{vid}.mp4") for vid in missing]


def download_one(video_id: str, dest: Path, dry_run: bool, block_size: int = 8192000) -> tuple[str, bool, str]:
    """
    Scarica un singolo video.
    Restituisce (video_id, success, error_msg).
    """
    if dry_run:
        log(f"  [DRY-RUN] {video_id} → {dest}")
        dest.touch()
        return video_id, True, ""

    url = build_url(video_id)
    tmp = dest.with_suffix(".part")

    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.getheader("content-length", 0))
            with open(tmp, "wb") as fh:
                downloaded = 0
                while True:
                    buf = resp.read(block_size)
                    if not buf:
                        break
                    fh.write(buf)
                    downloaded += len(buf)
        tmp.rename(dest)
        return video_id, True, ""
    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        return video_id, False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return video_id, False, str(e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scarica video mancanti per i task di inferenza.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Numero di download paralleli (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra cosa verrebbe scaricato senza effettuare il download")
    args = parser.parse_args()

    # Raccogli tutti i download necessari
    jobs: list[tuple[str, Path]] = []
    for task_name, config in TASKS.items():
        jobs.extend(find_missing(task_name, config))

    if not jobs:
        print("\nNessun video mancante. Tutto è già presente.")
        return

    print(f"\nDa scaricare: {len(jobs)} video  |  Worker paralleli: {args.workers}")
    if args.dry_run:
        print("[DRY-RUN] Nessun file verrà effettivamente scaricato.\n")

    errors: list[str] = []
    progress = tqdm(total=len(jobs), unit="video", desc="Download") if TQDM else None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, vid, dest, args.dry_run): vid
            for vid, dest in jobs
        }
        for future in as_completed(futures):
            vid, ok, err = future.result()
            if ok:
                log(f"  ✓  {vid}")
            else:
                log(f"  ✗  {vid}  —  {err}")
                errors.append(f"{vid}: {err}")
            if progress:
                progress.update()

    if progress:
        progress.close()

    print("\n" + "─" * 60)
    if errors:
        print(f"Completato con {len(errors)} errori:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"Download completato — {len(jobs)} video scaricati con successo.")


if __name__ == "__main__":
    main()
