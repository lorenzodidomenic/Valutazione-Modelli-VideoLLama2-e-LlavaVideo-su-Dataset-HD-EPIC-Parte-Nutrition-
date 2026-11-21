#!/bin/bash
# sync_drive.sh - copia codice e dati del test dell'oracolo dal Mac a Google Drive.
#
# Uso:   ./sync_drive.sh          copia
#        ./sync_drive.sh -n       prova a vuoto: mostra cosa copierebbe, senza copiare
#
# Solo Mac -> Drive. Non cancella nulla su Drive (niente --delete).
# Esclusi: ambienti virtuali, cache Python, git, checkpoint Jupyter, video.
set -euo pipefail

PROG="$(cd "$(dirname "$0")" && pwd)"   # la cartella del progetto e' quella di questo script
DRIVE=$(echo "$HOME"/Library/CloudStorage/GoogleDrive-*/"Il mio Drive/Progetto Deep Learning 1")
[ -d "$DRIVE" ] || { echo "Cartella Drive non trovata: $DRIVE (Google Drive e' aperto?)"; exit 1; }

# Sempre un elemento nell'array: con la bash 3.2 di macOS e set -u un array vuoto da' "unbound variable"
PROVA=(--itemize-changes)
[ "${1:-}" = "-n" ] && PROVA=(--dry-run) && echo "== PROVA A VUOTO: nessun file viene copiato =="

ESCLUDI=(--exclude 'venv/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.git/'
         --exclude '.ipynb_checkpoints/' --exclude '.DS_Store' --exclude '*.mp4')

echo "-- codice (.py): inference_folder/test_oracolo -> test_oracolo"
rsync -rtv "${PROVA[@]}" "${ESCLUDI[@]}" --include '*.py' --exclude '*' \
      "$PROG/inference_folder/test_oracolo/" "$DRIVE/test_oracolo/"

echo "-- dati: hd-epic-annotations/test_oracolo/dati -> test_oracolo/dati"
rsync -rtv "${PROVA[@]}" "${ESCLUDI[@]}" \
      "$PROG/hd-epic-annotations/test_oracolo/dati/" "$DRIVE/test_oracolo/dati/"
