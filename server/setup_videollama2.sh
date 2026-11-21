#!/usr/bin/env bash
# Installa VideoLLaMA 2 7B sul volume persistente del pod (/workspace). Si lancia UNA volta.
# Rilanciarlo e' sicuro: salta quello che e' gia' fatto.
#
# Riproduce l'ambiente del notebook 01_videollama2.ipynb (Colab):
#   torch 2.7.0 cu128, transformers 4.40.0, tokenizers 0.19.1, accelerate 0.26.1, decord 0.6.0,
#   repo ufficiale con la patch flash_attention_2 -> eager in encoder.py.
# Tutto (Python, pacchetti, repo, pesi) sta in /workspace, l'unica cartella che sopravvive
# allo spegnimento del pod.
set -euo pipefail

W=${W:-/workspace}
ENV=$W/envs/videollama2
REPO=$W/repos/VideoLLaMA2
COMMIT=c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414     # HEAD del repo al 21/11/2026

export HF_HOME=$W/hf_cache                        # pesi del modello sul volume
export UV_PYTHON_INSTALL_DIR=$W/uv/python         # anche Python sta sul volume
export UV_CACHE_DIR=$W/uv/cache

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true
echo "   (serve CUDA Version >= 12.8 per torch cu128)"

echo "== 1/5 uv"
command -v uv >/dev/null || pip install -q uv

echo "== 2/5 repository VideoLLaMA2 @ ${COMMIT:0:7}"
mkdir -p "$W/repos"
[ -d "$REPO/.git" ] || git clone -q https://github.com/DAMO-NLP-SG/VideoLLaMA2.git "$REPO"
git -C "$REPO" checkout -q "$COMMIT"
ENC=$REPO/videollama2/model/encoder.py
if grep -q '"flash_attention_2"' "$ENC"; then
    sed -i 's/"flash_attention_2"/"eager"/g' "$ENC"     # stessa patch del notebook
    echo "   patch applicata: flash_attention_2 -> eager in encoder.py"
else
    echo "   patch gia' presente"
fi

echo "== 3/5 ambiente Python 3.12 (come Colab)"
[ -x "$ENV/bin/python" ] || uv venv -q --python 3.12 --python-preference only-managed "$ENV"
PY=$ENV/bin/python
uv pip install -q --python "$PY" torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
# versioni del notebook; opencv headless perche' il pod non ha uno schermo (stesso cv2)
# peft fissato a una versione coeva di transformers 4.40 (le recenti non la supportano)
uv pip install -q --python "$PY" \
    "transformers==4.40.0" "tokenizers==0.19.1" "accelerate==0.26.1" "peft==0.10.0" \
    timm einops "decord==0.6.0" av opencv-python-headless \
    fastapi "uvicorn[standard]" huggingface_hub pandas
# il repo non si installa con pip (tirerebbe dentro altre versioni): lo si aggiunge al path
SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])")
echo "$REPO" > "$SITE/videollama2_repo.pth"

echo "== 4/5 controllo import"
"$PY" -c "import torch, transformers, decord, videollama2; \
print('   torch', torch.__version__, '| cuda', torch.cuda.is_available(), \
'| transformers', transformers.__version__)"

echo "== 5/5 pesi (circa 17 GB, solo la prima volta)"
"$PY" - <<'EOF'
import json, os
from huggingface_hub import snapshot_download
p = snapshot_download("DAMO-NLP-SG/VideoLLaMA2-7B")
vt = json.load(open(os.path.join(p, "config.json"))).get("mm_vision_tower")
if vt and not os.path.isdir(vt):
    snapshot_download(vt)          # encoder visivo, caricato da model_init
print("   modello:", p, "| encoder visivo:", vt)
EOF

uv pip freeze --python "$PY" > "$W/envs/videollama2_freeze.txt"
echo
echo "Fatto. Versioni esatte salvate in $W/envs/videollama2_freeze.txt (da citare nella tesi)."
echo "Ora: bash $(dirname "$0")/avvia_videollama2.sh"
