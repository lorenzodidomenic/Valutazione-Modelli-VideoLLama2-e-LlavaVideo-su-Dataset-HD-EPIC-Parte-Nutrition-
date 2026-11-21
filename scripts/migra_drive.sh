#!/bin/bash
# migra_drive.sh - UNA SOLA VOLTA: sposta su Drive i risultati gia' prodotti nella nuova struttura.
#
#   risultati/                 ->  results/baseline/     (baseline e Video Panels)
#   test_oracolo/risultati/    ->  results/oracolo/      (baseline C0 e run dell'oracolo)
#
# Uso:   scripts/migra_drive.sh           prova a vuoto: mostra cosa farebbe
#        scripts/migra_drive.sh --esegui  sposta davvero
#
# Non cancella e non sovrascrive: se la destinazione esiste ed e' non vuota, si ferma.
set -euo pipefail

DRIVE=$(echo "$HOME"/Library/CloudStorage/GoogleDrive-*/"Il mio Drive/Progetto Deep Learning 1")
[ -d "$DRIVE" ] || { echo "Cartella Drive non trovata: $DRIVE (Google Drive e' aperto?)"; exit 1; }

ESEGUI=0
[ "${1:-}" = "--esegui" ] && ESEGUI=1
[ "$ESEGUI" = 1 ] || echo "== PROVA A VUOTO: usare --esegui per spostare davvero =="

sposta() {
  local da="$DRIVE/$1" a="$DRIVE/$2"
  if [ ! -d "$da" ]; then echo "salto: $1 non esiste"; return; fi
  if [ -d "$a" ] && [ -n "$(ls -A "$a" 2>/dev/null)" ]; then
    echo "STOP: $2 esiste gia' e non e' vuota: controllare a mano"; exit 1
  fi
  echo "$1  ->  $2   ($(ls "$da" | wc -l | tr -d ' ') file)"
  if [ "$ESEGUI" = 1 ]; then
    mkdir -p "$(dirname "$a")"
    rmdir "$a" 2>/dev/null || true
    mv "$da" "$a"
  fi
}

sposta "risultati" "results/baseline"
sposta "test_oracolo/risultati" "results/oracolo"
