# Ipotesi 1: il modello usa davvero il video?

**Osservazione di partenza.** Su `video_nutrition_estimation` VideoLLaMA 2 e LLaVA-Video
arrivano al 66–68% (caso: 20%). Le opzioni però sono quasi sempre ingredienti della stessa
ricetta (192 distrattori su 200), e la domanda ("quale ha più carboidrati?") si può spesso
risolvere con la sola conoscenza del mondo. Il dubbio è che i modelli rispondano dal testo
e diano poco peso al video.

**Esperimento.** Stesse 50 domande, stesse opzioni, stesso prompt. Cambia solo l'input visivo:

| Condizione | Input visivo | Da dove viene |
|---|---|---|
| `C0_video` | il video vero | **riusato** dal baseline del corso |
| `C1_testo` | nessuno: solo domanda e opzioni | nuovo run |
| `C3_scambio` | il video di un'altra ricetta, in cui **nessuna** delle 5 opzioni compare | nuovo run |

I frame neri (C2) sono stati considerati e scartati.

## Criterio, fissato prima dei run

Metrica principale: **accordo con C0**, cioè la % di domande con la stessa lettera che col
video vero. Un modello che guarda il video deve cambiare risposta quando il video sparisce
o è di un'altra ricetta.

> Il modello **ignora il video** se in `C3_scambio`
> - l'accordo con C0 è **≥ 80%**, e
> - l'accuratezza non cambia in modo significativo (McNemar esatto, p ≥ 0.05).

Le soglie sono in `src/hdepic/ablazione.py` (`SOGLIA_ACCORDO`, `ALPHA`).
**Soglia da confermare col professore prima dei run.**

Metriche di supporto:
- **kappa di Cohen**, l'accordo corretto per il caso;
- **IC 95% di Wilson**: con 50 domande sono larghi circa ±13 punti;
- accuratezza **per nutriente**;
- **bias posizionale**.

## Struttura

Il codice sta in `src/hdepic/` e i dati in `data/`, condivisi con il resto del progetto:

```
src/hdepic/ablazione.py       condizioni, costruzione degli scambi, metriche, grafico
                              (MANCANTE: vedi sotto)
data/hd-epic/                 annotazioni originali e ricette (complete_recipes.json)
data/derived/ipotesi_video/
  swap_map.json               domanda -> video originale, video di scambio, ricette
  annotazioni_swap/           annotazioni del benchmark con SOLO l'id del video sostituito
notebooks/02_ipotesi_video/
  00_preparazione.ipynb       locale, senza GPU: costruisce e verifica gli scambi
  01_videollama2.ipynb        Colab L4: C1 e C3 per VideoLLaMA 2
  02_llavavideo.ipynb         Colab L4: C1 e C3 per LLaVA-Video
  03_analisi.ipynb            locale o Colab senza GPU: tabella, verdetto, grafico
results/ipotesi_video/        un .jsonl per modello e condizione, figure/
```

Nomi dei risultati, nello schema del motore (`<modello>_<condizione>_<task>.jsonl`):

| File | Prodotto da |
|---|---|
| `videollama2_C0_video_video_nutrition_estimation.jsonl` | copia del baseline (vedi sotto) |
| `videollama2_C1_testo_video_nutrition_estimation.jsonl` | `01_videollama2` |
| `videollama2_C3_scambio_video_nutrition_estimation.jsonl` | `01_videollama2` |
| `videollama2_C0_controllo_...` e `videollama2_prova_...` | `01_videollama2` (controlli, non entrano nell'analisi) |
| idem con `llavavideo_` | `02_llavavideo` |

## Regole dello scambio (C3)

Il video di scambio di ogni domanda:
1. è uno dei 14 video già usati dal prototipo, quindi è già su Drive;
2. mostra una ricetta diversa (in 44 domande su 50 è anche di un altro partecipante);
3. non contiene nessuna delle 5 opzioni fra i suoi ingredienti. Il confronto è per parole intere ed è volutamente prudente.

Ogni domanda ha da 3 a 13 video validi. Si sceglie quello usato meno volte finora, e i
pareggi si rompono con seed 0, così ogni video viene usato da 2 a 5 volte.

## Passi

- [x] Costruzione e verifica degli scambi (`00_preparazione`)
- [ ] **Soglia confermata col professore**
- [ ] **C0 al suo posto.** Su Drive, in `Progetto Deep Learning 1/results/baseline/`, trovare per
      ciascun modello il `.jsonl` del baseline di `video_nutrition_estimation`. Deve avere
      50 domande, con accuratezza 66% per VideoLLaMA 2 e 68% per LLaVA-Video. Copiarlo in
      `results/ipotesi_video/` con il nome `<modello>_C0_video_video_nutrition_estimation.jsonl`.
- [ ] `scripts/sync_drive.sh` lanciato dal Mac (copia `src/`, `data/`, `notebooks/` su Drive)
- [ ] `01_videollama2` su Colab: prova rapida, controllo C0 (5/5 identiche), run completi
- [ ] `02_llavavideo` su Colab, in una sessione nuova (le versioni di `transformers` sono incompatibili)
- [ ] `03_analisi`

**Se il controllo C0 non dà 5/5 risposte identiche**, l'ambiente Colab è cambiato rispetto
al baseline. In quel caso C0 va rifatto in questa sessione, prima del confronto. Basta
chiamare `A.esegui(MODELLO, "C0_video", infer_video, ANN_ORIG, VIDEO_INDEX, OUT_DIR,
readable_cache=READABLE)` dopo aver rinominato il vecchio file.

## Eseguire 03_analisi in locale

Il Python di sistema ha un conflitto fra numpy 2 e matplotlib. Conviene un ambiente dedicato:

```bash
python3 -m venv .venv && .venv/bin/pip install pandas matplotlib jupyter
.venv/bin/jupyter notebook notebook/03_analisi.ipynb
```

In alternativa, il notebook gira su Colab senza GPU: monta Drive da solo.
