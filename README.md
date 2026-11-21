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
| VideoLLaMA 2 7B | 32.7 | **36.7** | +4.0 |
| LLaVA-Video 7B  | 38.7 | **38.7** | +0.0 |
| Umano           | 85.0 | — | |
| Casuale         | 20.0 | — | |

Dettaglio per prototipo:

| Prototipo (N valutate) | VideoLLaMA 2 | LLaVA-Video |
|---|---|---|
| `video_nutrition_estimation` (50) | 66.0 | 68.0 |
| `nutrition_change` (50) | 16.0 | 22.0 |
| `image_nutrition_estimation` (100) | 28.0 | 26.0 |
| Media per prototipo | 36.7 | 38.7 |
| Aggregata sulle domande | 34.5 | 35.5 |

Su `image_nutrition_estimation` entrambi i modelli restano vicini al caso (20%): ogni
domanda mostra 5 fotogrammi presi da 5 video diversi a timestamp precisi e chiede quale
ingrediente abbia più calorie, grassi, carboidrati o proteine. Le opzioni sono nomi di
ingredienti come in `video_nutrition_estimation`, ma qui la conoscenza linguistica non
basta. I `metadata` delle annotazioni lo spiegano: la risposta corretta è l'ingrediente con il
valore **assoluto** più alto, cioè riferito alla quantità usata nella ricetta (100/100 domande;
è anche quello con la quantità maggiore, 100/100), e **mai** quello con il valore più alto per
unità di quantità (`*_amount_ratio`, 0/100). Il prototipo è costruito in modo che la conoscenza
generale sugli ingredienti porti fuori strada: la quantità si può stimare solo dalle immagini.
L'immagine *i* mostra sempre l'ingrediente dell'opzione *i*, ma il prompt (come nel paper) non
lo dice; in VideoLLaMA 2 le 5 immagini vengono completate con 3 frame neri e il connettore STC
(convoluzione 3D, passo temporale 2) le fonde a coppie negli stessi token visivi.


### Video Panels

Video naturale contro pannelli 2×2, sulle stesse 99 domande dei due prototipi basati su
video e a parità di finestra di contesto.

| Prototipo | VideoLLaMA 2 base | panels | LLaVA-Video base | panels |
|---|---|---|---|---|
| `video_nutrition_estimation` | 66.0 | 62.0 | 68.0 | 72.0 |
| `nutrition_change` | 16.3 | 20.4 | 20.4 | 14.3 |
| **Media per prototipo** | **41.2** | **41.2** | **44.2** | **43.2** |

**Il metodo non produce alcun guadagno sulla categoria** (0.0 e −1.0 punti, entrambi
dentro il rumore di campionamento: con 50 domande per prototipo una risposta vale 2 punti). I due modelli si spostano però in direzioni opposte
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
  extract_image_frames.py   estrae i fotogrammi di image_nutrition_estimation dal server
  extract_clip_frames.py    estrae le clip [start_time, end_time] di nutrition_change dal server

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
python3 preprocessing_folder/extract_image_frames.py # 132 fotogrammi per il prototipo a immagini
python3 preprocessing_folder/extract_clip_frames.py  # 128 fotogrammi per ciascuna delle 50 clip di nutrition_change (richiede ffmpeg)
```

`image_nutrition_estimation` non usa i video sottocampionati: ogni domanda richiede 5
fotogrammi a timestamp esatti, che nel video a 128 frame non esistono. Lo script li estrae
dai video integrali leggendoli direttamente dal server del dataset (solo il tratto attorno
al timestamp, senza scaricare i video) e li salva con il nome atteso dalla cache dei
notebook; la cartella va caricata su Drive come `frames_image_nutrition_estimation`.

Anche `nutrition_change` non usa i video sottocampionati: ogni domanda indica un intervallo
`start_time`–`end_time` di 1–27 s dentro un video di decine di minuti, e l'input del benchmark
è la **clip**, non il video intero (nel video a 128 frame l'intervallo cade quasi sempre fra due
frame). `extract_clip_frames.py` decodifica con ffmpeg solo quel tratto, sempre dal server, e
salva 128 fotogrammi uniformi a 768 px per clip; la cartella va caricata su Drive come
`frames_nutrition_change`. Le clip sono tutte sotto i 128 s oltre i quali il paper applica i
pannelli, quindi `nutrition_change` è escluso dall'esperimento Video Panels.

I parametri dei pannelli sono quelli indicati come migliori dagli autori del paper:
`C=32`, `alpha=beta=2`, bordo 0, stride temporale 1. Ogni video produce 32 pannelli che
coprono 128 fotogrammi, gli stessi 128 del video sottocampionato: i due input
dell'esperimento partono dalla stessa evidenza visiva e differiscono solo per come è
impacchettata.

**3. Inferenza**

I due notebook sono pensati per Google Colab con una **L4 24GB** e Python 3.12 (il baseline
era stato eseguito su A100 80GB, con le stesse versioni) e leggono i dati da Google Drive
(va aggiornata la variabile `BASE`). Ogni notebook va eseguito in una sessione Colab nuova:
le due versioni di `transformers` sono incompatibili. Su L4, LLaVA-Video calcola i logits
solo sull'ultima posizione, altrimenti il prefill con 32 fotogrammi va in out of memory.
Ambienti richiesti:

| | VideoLLaMA 2 7B | LLaVA-Video 7B |
|---|---|---|
| checkpoint | `DAMO-NLP-SG/VideoLLaMA2-7B` | `lmms-lab/LLaVA-Video-7B-Qwen2` |
| transformers | 4.40.0 | 4.37.2 |
| precisione | float16 | bfloat16 |
| attention | eager | sdpa |
| frame visti dal modello | 8 (`num_frames` di default; anche i 32 pannelli vengono sottocampionati a 8) | 32 |

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
