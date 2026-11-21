#!/usr/bin/env bash
# Avvia il server di VideoLLaMA 2 in background. Da lanciare a ogni accensione del pod.
# Il server resta acceso anche se chiudi la connessione SSH.
set -euo pipefail

W=${W:-/workspace}
QUI=$(cd "$(dirname "$0")" && pwd)
PORTA=${PORTA:-8001}
PY=$W/envs/videollama2/bin/python

[ -x "$PY" ] || { echo "Ambiente assente: lancia prima setup_videollama2.sh"; exit 1; }
if pgrep -f "server_videollama2:app" >/dev/null; then
    echo "Il server e' gia' acceso. Per riavviarlo: pkill -f server_videollama2:app"
    exit 0
fi

# chiave API: generata la prima volta, poi riletta dal volume
CHIAVE_FILE=$W/chiave_api.txt
[ -s "$CHIAVE_FILE" ] || "$PY" -c "import secrets; print(secrets.token_hex(24))" > "$CHIAVE_FILE"

export CHIAVE_API=$(cat "$CHIAVE_FILE")
export HF_HOME=$W/hf_cache
export HF_HUB_OFFLINE=1                         # pesi gia' scaricati dal setup
export DATI=${DATI:-$W/dati}                    # cartelle video* copiate da Drive
export SRC=${SRC:-$QUI/../src}                   # per importare hdepic.eval
export VL2_REPO=$W/repos/VideoLLaMA2

mkdir -p "$W/logs"
cd "$VL2_REPO"                                  # come il %cd del notebook
nohup "$PY" -m uvicorn server_videollama2:app --app-dir "$QUI" \
      --host 0.0.0.0 --port "$PORTA" > "$W/logs/videollama2.log" 2>&1 &

echo "Server avviato (pid $!). Il caricamento del modello richiede 1-2 minuti."
echo "  log:     tail -f $W/logs/videollama2.log   (pronto quando compare 'Application startup complete')"
echo "  URL:     https://${RUNPOD_POD_ID:-<POD_ID>}-${PORTA}.proxy.runpod.net"
echo "  chiave:  $CHIAVE_API"
