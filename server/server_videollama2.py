"""
Server HTTP di VideoLLaMA 2 7B per la tesi (ipotesi 1).

Il calcolo e' identico al notebook 01_videollama2.ipynb: stesso checkpoint, stessa patch
(eager al posto di flash_attention_2), elementi visivi campionati da processor["video"]
(8 frame), decoding greedy, 32 token. Cambia solo la macchina su cui gira.

I video stanno sul pod, sotto DATI. Il client invia il percorso sul pod (ottenuto da
/indice), non i frame: cosi' il campionamento resta quello del repo ufficiale.

Endpoint (tutti richiedono l'header X-Chiave):
  GET  /salute   versioni, GPU, commit e patch: da annotare nella tesi
  GET  /indice   video presenti sul pod e se sono decodificabili (sostituisce
                 build_video_index + preflight in locale)
  POST /infer    {"modal": "video" | "text", "prompt": str, "visual": path o lista di path}

Avvio: avvia_videollama2.sh (imposta le variabili d'ambiente qui sotto).
"""
import glob
import os
import secrets
import subprocess
import sys
import threading
from typing import List, Optional, Union

REPO = os.environ.get("VL2_REPO", "/workspace/repos/VideoLLaMA2")
DATI = os.path.realpath(os.environ.get("DATI", "/workspace/dati"))
SRC = os.environ.get("SRC", "/workspace/tesi/src")   # cartella src/ del repository
CHIAVE = os.environ["CHIAVE_API"]
MAX_NEW_TOKENS = 32                       # come nel notebook: la risposta e' una lettera

sys.path.insert(0, SRC)

import numpy as np
import torch
import transformers
from fastapi import Depends, FastAPI, Header, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from hdepic import eval as H
from videollama2 import mm_infer, model_init

# ------------------------------------------------------------------ modello
model_path = snapshot_download("DAMO-NLP-SG/VideoLLaMA2-7B")
model, processor, tokenizer = model_init(model_path, device_map={"": "cuda:0"},
                                         attn_implementation="eager", use_flash_attn=False)
model.eval()
NUM_FRAMES = getattr(model.config, "num_frames", 8)
LOCK = threading.Lock()                   # una GPU, una richiesta alla volta
print(f"VideoLLaMA2-7B caricato - dtype {next(model.parameters()).dtype} | "
      f"num_frames {NUM_FRAMES} | dati in {DATI}", flush=True)


def sottocampiona(paths, n=NUM_FRAMES):
    """Liste di immagini -> n elementi uniformi (process_video ricampiona solo i path mp4)."""
    if len(paths) <= n:
        return paths
    sel = np.linspace(0, len(paths) - 1, n, dtype=int)
    return [paths[i] for i in sel]


# ------------------------------------------------------------------ server
app = FastAPI(title="VideoLLaMA2-7B (tesi)")


def controlla_chiave(x_chiave: str = Header(...)):
    if not secrets.compare_digest(x_chiave, CHIAVE):
        raise HTTPException(401, "chiave errata")


def sul_pod(p):
    """Accetta solo file esistenti sotto DATI (niente letture arbitrarie dal disco)."""
    r = os.path.realpath(p)
    if not r.startswith(DATI + os.sep):
        raise HTTPException(400, f"percorso fuori da {DATI}: {p}")
    if not os.path.exists(r):
        raise HTTPException(404, f"file non trovato sul pod: {p}")
    return r


@app.get("/salute", dependencies=[Depends(controlla_chiave)])
def salute():
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    enc = open(os.path.join(REPO, "videollama2", "model", "encoder.py")).read()
    return {"modello": "DAMO-NLP-SG/VideoLLaMA2-7B", "num_frames": NUM_FRAMES,
            "dtype": str(next(model.parameters()).dtype), "max_new_tokens": MAX_NEW_TOKENS,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__, "repo_commit": commit,
            "patch_eager": '"flash_attention_2"' not in enc,
            "hdepic.eval": H.VERSION, "dati": DATI}


_INDICE = None


@app.get("/indice", dependencies=[Depends(controlla_chiave)])
def indice(aggiorna: bool = False):
    """Stessa regola dei notebook: cartelle che iniziano con video/videos, senza 'panel'."""
    global _INDICE
    if _INDICE is None or aggiorna:
        roots = sorted(d for d in glob.glob(os.path.join(DATI, "*"))
                       if os.path.isdir(d) and "panel" not in os.path.basename(d).lower()
                       and os.path.basename(d).lower().startswith(("video", "videos")))
        idx = H.build_video_index(roots)
        leggibili = {}
        for tipo in ("sampled", "full"):
            for p in idx[tipo].values():
                H.is_readable(p, leggibili)
        _INDICE = {"cartelle": [os.path.basename(d) for d in roots],
                   "video_index": idx, "leggibili": leggibili}
    return _INDICE


class Richiesta(BaseModel):
    modal: str = "video"
    prompt: str
    visual: Optional[Union[str, List[str]]] = None


@app.post("/infer", dependencies=[Depends(controlla_chiave)])
def infer(r: Richiesta):
    if r.modal not in ("video", "text"):
        raise HTTPException(400, f"modal sconosciuto: {r.modal}")
    if r.modal == "video":
        if not r.visual:
            raise HTTPException(400, "modal=video senza visual")
        visual = (sottocampiona([sul_pod(p) for p in r.visual])
                  if isinstance(r.visual, list) else sul_pod(r.visual))
    with LOCK:
        try:
            if r.modal == "video":
                tensor = processor["video"](visual)
                if tensor.shape[0] != NUM_FRAMES:
                    raise RuntimeError(f"attesi {NUM_FRAMES} elementi visivi, "
                                       f"arrivati {tensor.shape[0]}")
                risposta = mm_infer(tensor, r.prompt, model, tokenizer, modal="video",
                                    do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
                n = int(tensor.shape[0])
            else:
                risposta = mm_infer(None, r.prompt, model, tokenizer, modal="text",
                                    do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
                n = 0
        except Exception as e:
            torch.cuda.empty_cache()
            raise HTTPException(500, f"{type(e).__name__}: {e}")
    return {"risposta": risposta, "n_frame": n}
