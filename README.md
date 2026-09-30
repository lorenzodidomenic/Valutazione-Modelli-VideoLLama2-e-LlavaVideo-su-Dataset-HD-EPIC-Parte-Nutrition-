# Valutazione di VideoLLaMA 2 e LLaVA-Video sul benchmark HD-EPIC (categoria Nutrition)

Progetto per il corso di Deep Learning. Il lavoro ha due obiettivi:

1. **riprodurre** i valori di baseline riportati nel paper HD-EPIC per due
   VLLM open da 7B sulla categoria *Nutrition* del suo benchmark VQA;
2. **valutare** il metodo di visual prompting *Video Panels* su quella stessa categoria,
   a parità di finestra di contesto.

---

## Risultati

### Riproduzione delle baseline

Accuratezza sulla categoria *Nutrition*, calcolata come media delle accuratezze per
prototipo (la metrica usata dal paper).

| Modello | Paper | Riprodotto | Δ |
|---|---|---|---|
| VideoLLaMA 2 7B | 32.7 | **36.5** | +3.8 |
| LLaVA-Video 7B  | 38.7 | **39.9** | +1.2 |
| Umano           | 85.0 | — | |
| Casuale         | 20.0 | — | |

Dettaglio per prototipo:

| Prototipo (N valutate) | VideoLLaMA 2 | LLaVA-Video |
|---|---|---|
| `video_nutrition_estimation` (46) | 69.6 | 71.7 |
| `nutrition_change` (50) | 16.0 | 22.0 |
| `image_nutrition_estimation` (88) | 23.9 | 26.1 |
| Media per prototipo | 36.5 | 39.9 |
| Aggregata sulle domande | 33.2 | 36.4 |

In tutte le esecuzioni il 100% delle risposte è stato interpretato al primo livello del
parser (lettera in apertura): nessuna accuratezza è stata persa per ragioni di formato.

### Video Panels

Video liscio contro pannelli 2×2, sulle stesse 96 domande dei due prototipi basati su
video e a parità di finestra di contesto.

| Prototipo | VideoLLaMA 2 base | panels | LLaVA-Video base | panels |
|---|---|---|---|---|
| `video_nutrition_estimation` | 69.6 | 65.2 | 71.7 | 76.1 |
| `nutrition_change` | 16.0 | 20.0 | 22.0 | 16.0 |
| **Media per prototipo** | **42.8** | **42.6** | **46.8** | **46.0** |

**Il metodo non produce alcun guadagno sulla categoria** (−0.2 e −0.8 punti, entrambi
dentro il rumore di campionamento). I due modelli si spostano però in direzioni opposte
sugli stessi prototipi, con variazioni di ampiezza simile e segno contrario: le medie
restano piatte perché i due movimenti si annullano.

Su `nutrition_change` entrambi i modelli sono al livello del caso o sotto, e il bias
posizionale mostra che concentrano le risposte su poche opzioni indipendentemente dal
contenuto del video — comportamento che i pannelli non modificano.

---

## Struttura del repository

```
preprocessing_folder/
  nutrition_*.json          annotazioni dei 3 prototipi (200 domande)
  id_videos_fetching.py     estrae gli id dei video dalle annotazioni
  download_missing.py       scarica in parallelo i video mancanti per prototipo
  sampling_script.py        sottocampiona ogni video a 128 frame uniformi

paper_mosaic_sampling/
  mosaic.py                 generazione dei pannelli 2x2 (Video Panels)

inference_folder/
  hdepic_eval.py            motore di valutazione condiviso: prompt, parsing, metriche
  01_videollama2_inference.ipynb
  02_llavavideo_inference.ipynb
  old/                      versioni precedenti dei notebook, prima del refactor
```

---

## Riprodurre gli esperimenti

**1. Annotazioni e video**

```bash
python3 preprocessing_folder/id_videos_fetching.py   # stampa il comando del downloader
python3 preprocessing_folder/download_missing.py     # recupera i mancanti (--dry-run per provare)
```

**2. Pre-processing**

```bash
python3 preprocessing_folder/sampling_script.py      # 128 frame uniformi per video, 1 fps
python3 paper_mosaic_sampling/mosaic.py              # 32 pannelli 2x2 per video
```

I parametri dei pannelli sono quelli indicati come migliori dagli autori del paper:
`C=32`, `alpha=beta=2`, bordo 0, stride temporale 1. Ogni video produce 32 pannelli che
coprono 128 fotogrammi, gli stessi 128 del video sottocampionato: i due input
dell'esperimento partono dalla stessa evidenza visiva e differiscono solo per come è
impacchettata.

**3. Inferenza**

I due notebook sono pensati per Google Colab con una **A100 80GB** e leggono i dati da
Google Drive (va aggiornata la variabile `BASE`). Ambienti richiesti:

| | VideoLLaMA 2 7B | LLaVA-Video 7B |
|---|---|---|
| checkpoint | `DAMO-NLP-SG/VideoLLaMA2-7B` | `lmms-lab/LLaVA-Video-7B-Qwen2` |
| transformers | 4.40.0 | 4.37.2 |
| precisione | float16 | bfloat16 |
| attention | eager | sdpa |
| frame visti dal modello | 8 (`num_frames` di default) | 32 |

Le due versioni di `transformers` sono incompatibili fra loro: i notebook vanno eseguiti
in sessioni separate. Per VideoLLaMA 2 serve inoltre una patch alla vision tower, che il
repository forza a usare FlashAttention senza supportarlo; la patch è applicata
automaticamente dal notebook.

---
---

## Riferimenti

- Perrett et al., *HD-EPIC: A Highly-Detailed Egocentric Video Dataset*, CVPR 2025.
- Doorenbos, Spurio, Gall, *Video Panels for Long Video Understanding*,
  [arXiv:2509.23724](https://arxiv.org/abs/2509.23724) —
  [codice ufficiale](https://github.com/FedeSpu/Video-Panels).
- [VideoLLaMA 2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) (DAMO-NLP-SG).
- [LLaVA-NeXT / LLaVA-Video](https://github.com/LLaVA-VL/LLaVA-NeXT) (lmms-lab).
