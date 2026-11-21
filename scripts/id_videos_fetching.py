"""Stampa il comando del downloader ufficiale per scaricare i video di un prototipo.

Uso:  python3 scripts/id_videos_fetching.py [prototipo] [cartella-output]
      prototipo: image_nutrition_estimation (default), video_nutrition_estimation, nutrition_change
"""
import json
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
prototipo = sys.argv[1] if len(sys.argv) > 1 else "image_nutrition_estimation"
output_path = sys.argv[2] if len(sys.argv) > 2 else "./HD-EPIC"

with open(RADICE / "data" / "hd-epic" / f"nutrition_{prototipo}.json") as f:
    data = json.load(f)

# Estrai tutti gli ID univoci dai campi inputs (es. "video 1", "video 2", ...)
video_ids = set()
for entry in data.values():
    for video in entry["inputs"].values():
        video_ids.add(video["id"])

print(len(video_ids), "video")
video_ids_str = ",".join(sorted(video_ids))

# il downloader legge data/md5.txt dalla cartella corrente: va lanciato da li'
print("cd third_party/hd-epic-downloader")
print(f"python3 hd-epic-downloader.py {output_path} --videos --video-id {video_ids_str}")
