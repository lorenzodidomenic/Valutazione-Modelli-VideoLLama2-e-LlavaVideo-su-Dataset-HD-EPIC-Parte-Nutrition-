#!/bin/bash
# sync_drive.sh - copia codice, dati piccoli e notebook dal Mac a Google Drive.
#
# Uso:   scripts/sync_drive.sh          copia
#        scripts/sync_drive.sh -n       prova a vuoto: mostra cosa copierebbe, senza copiare
#
# Su Drive la cartella "Progetto Deep Learning 1" ha la stessa struttura del repository
# per queste parti, cosi' src/hdepic/*.py calcola gli stessi percorsi sul Mac e su Colab:
#   src/        codice (solo .py)
#   data/       annotazioni HD-EPIC e dati derivati (json piccoli)
#   notebooks/  i notebook da aprire con Colab
# I video, i fotogrammi e results/ restano solo su Drive e non vengono toccati.
#
# Solo Mac -> Drive. Non cancella nulla su Drive (niente --delete).
set -euo pipefail

PROG="$(cd "$(dirname "$0")/.." && pwd)"   # radice del progetto: la cartella sopra scripts/
DRIVE=$(echo "$HOME"/Library/CloudStorage/GoogleDrive-*/"Il mio Drive/Progetto Deep Learning 1")
[ -d "$DRIVE" ] || { echo "Cartella Drive non trovata: $DRIVE (Google Drive e' aperto?)"; exit 1; }

# Sempre un elemento nell'array: con la bash 3.2 di macOS e set -u un array vuoto da' "unbound variable"
PROVA=(--itemize-changes)
[ "${1:-}" = "-n" ] && PROVA=(--dry-run) && echo "== PROVA A VUOTO: nessun file viene copiato =="

ESCLUDI=(--exclude 'venv/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.git/'
         --exclude '.ipynb_checkpoints/' --exclude '.DS_Store' --exclude '*.mp4')

echo "-- codice: src/ -> src/   (solo .py)"
rsync -rtv "${PROVA[@]}" "${ESCLUDI[@]}" --include '*/' --include '*.py' --exclude '*' \
      "$PROG/src/" "$DRIVE/src/"

echo "-- dati: data/ -> data/"
rsync -rtv "${PROVA[@]}" "${ESCLUDI[@]}" "$PROG/data/" "$DRIVE/data/"

echo "-- notebook: notebooks/ -> notebooks/   (solo .ipynb)"
rsync -rtv "${PROVA[@]}" "${ESCLUDI[@]}" --include '*/' --include '*.ipynb' --exclude '*' \
      "$PROG/notebooks/" "$DRIVE/notebooks/"
