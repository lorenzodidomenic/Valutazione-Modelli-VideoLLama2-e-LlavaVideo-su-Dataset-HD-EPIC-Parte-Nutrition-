# Bug e punti deboli da sistemare

Revisione del 23/10/2026. Ordinati per gravità. Spuntare man mano.

> **Verifica del 13/11/2026: nessun punto risulta risolto.**
> - Ultimo commit (locale e su `origin/master`): `43904af` del 10/10/2026, cioè prima di questa revisione.
>   Nessun file tracciato modificato, nessuno stash, nessun altro branch, nessun file della cartella più recente di questo documento.
> - Controllo a campione sul codice, tutti ancora presenti:
>   comando shell in `id_videos_fetching.py:20` (#10), `dest.touch()` nel dry-run e nessun `timeout`/MD5 in `download_missing.py` (#9),
>   `num_frames=124` e `videos_3` in `sampling_script.py` (#11), frame neri silenziosi e solo `nutrition_estimation` attivo in `mosaic.py` (#6),
>   regex case-insensitive nel parser (#12), nessun uso di `start_time`/`end_time` (#2), nessun sottocampionamento dei pannelli
>   con `linspace` nel notebook VideoLLaMA2 (#1), nessun IC/McNemar in `accuracy_report`/`compare_runs` (#7), nessun baseline solo testo (#8),
>   "non verificata" ancora in README:40 e relazione.tex:370 (#4), `relazione/figure.jpg` assente e doppio `---` nel README (#14).
> - Se le correzioni sono state fatte altrove (Colab/Drive o un'altra copia del progetto), vanno riportate in questa cartella e committate.

---

## 🔴 Critici (cambiano numeri o conclusioni)

### 1. VideoLLaMA2 + Panels: input non a parità di contesto
- [x] **Codice sistemato (13/11/2026).** Confermato sul sorgente (`videollama2/mm_utils.py:183-194`): una lista di path
  viene caricata tutta e troncata a `MAX_FRAMES=32`. Nel notebook: `sottocampiona()` + `assert` sul numero di elementi
  visivi (cella 19), stampa dei pannelli realmente visti (cella 25), nuovo tag `videollama2_panels_8` per non riprendere
  il vecchio run (cella 26). Corretti relazione.tex (descrizione del preprocessore) e README (riga "frame visti").
- [ ] **Resta da fare:** rilanciare il run panels di VideoLLaMA2 e aggiornare i numeri in README (tabella Video Panels),
  relazione.tex (risultati e discussione) e nelle celle 28/30.
- **Dove:** `videollama2_infer` in [01_videollama2_inference.ipynb](inference_folder/01_videollama2_inference.ipynb) (cella 19), cella 25
- **Problema:** `process_video` del repo VideoLLaMA2 ricampiona a `num_frames=8` **solo** quando riceve un path video.
  Con una lista di path di immagini le carica **tutte** e tronca a `MAX_FRAMES=32`.
  Quindi baseline = 8 frame, panels = **32 pannelli** (128 frame coperti, 4× token visivi).
  L'output della cella 25 ("il processor ne campiona 8") è sbagliato.
- **Fix:** in `videollama2_infer`, se `visual` è una lista di pannelli, sottocampionarla a
  `model.config.num_frames` con `np.linspace` (come fa `_immagini` in LLaVA). In alternativa portare la
  baseline a 32 frame. Rifare il run panels.
- **Da aggiornare dopo:** README (tabella "frame visti dal modello"),
  [relazione.tex:270-273](relazione/relazione.tex#L270-L273), [:386-393](relazione/relazione.tex#L386-L393),
  [:449-453](relazione/relazione.tex#L449-L453) (l'interpretazione "VideoLLaMA2 vede solo 8 frame" non regge).

### 2. `nutrition_change`: la finestra temporale della domanda viene ignorata
- [x] **Codice sistemato (13/11/2026).** Confermato dal README ufficiale HD-EPIC ("inputs: one or more videos,
  one or more clips"): con `start_time`/`end_time` l'input è la clip.
  - Nuovo [extract_clip_frames.py](preprocessing_folder/extract_clip_frames.py): ffmpeg decodifica solo
    `[start, end)` dal server (seek preciso al frame, niente download dei video interi) e salva 128 JPEG
    uniformi a 768 px per clip in `frames_nutrition_change/<video_id>_<start ms>_<end ms>/`. Provato su una
    clip (q. 25, 1.03 s → 31 frame decodificati a 30 fps, contenuto coerente con la domanda): ~66 s a clip.
  - [hdepic_eval.py](inference_folder/hdepic_eval.py) v1.1: `CLIP_TASKS`, `clip_name`, `clip_frames`,
    parametro `clip_cache` in `preflight`/`run_task`/`run_all`; senza clip la domanda viene esclusa.
    Testato in locale con un `infer_fn` finto.
  - Notebook: `CLIP_DIR`, `clip_cache=CLIP_DIR`, nuovi tag `*_baseline_clip`, `%%writefile` riallineati al `.py`.
  - **Decisione:** `nutrition_change` è escluso dai Video Panels (`H.PANEL_TASKS`): le clip (max 27 s) sono sotto
    la soglia `γ·C·α·β = 128 s` del paper, sotto cui il metodo ricade sul campionamento normale. Un run panels
    su quel task ora dà `ValueError`.
  - **Decisione:** i `<TIME …>` della domanda restano assoluti, come nel benchmark (vedi limite in relazione).
  - Aggiornati README (preprocessing, struttura) e relazione (nuova sottosezione sulle clip, ambito panels, limiti,
    sviluppi).
- [x] Clip estratte (50/50 complete, 128 frame ciascuna, 452 MB) e caricate su Drive come `frames_nutrition_change`.
- [ ] **Resta da fare:** rilanciare i baseline
  di entrambi i modelli e aggiornare i numeri: README (tabelle risultati e Video Panels, ora su 50 domande),
  relazione (abstract righe ~50-56, tabella per prototipo, bias posizionale di `nutrition_change`, tabella e
  discussione Video Panels).
- **Dove:** [hdepic_eval.py:229-235](inference_folder/hdepic_eval.py#L229-L235), [:338-342](inference_folder/hdepic_eval.py#L338-L342)
- **Problema:** ogni domanda ha `start_time`/`end_time` negli `inputs` (finestra mediana 6.5 s,
  min 1 s, max 27 s) in video lunghi più di 12 minuti. Si passa invece il video intero sottocampionato:
  128 frame vuol dire un frame ogni 6 s o più, e il modello ne vede 8 o 32, quindi l'intervallo non arriva quasi mai.
  In più il `_sampled` dura 128 s a 1 fps, per cui i tag `<TIME 00:13:00.074>` non esistono nella sua timeline.
- **Fix:** campionare i frame **dentro `[start_time, end_time]`** del video integrale.
  - VideoLLaMA2: `process_video` accetta già `s, e`.
  - LLaVA: campionare in `sample_video_frames` solo nell'intervallo.
  - Per non scaricare i video interi: leggere la clip dal server via HTTP come in
    [extract_image_frames.py](preprocessing_folder/extract_image_frames.py) e salvarla in cache
    (es. `clips_nutrition_change/<key>.mp4`).
  - Valutare se riscrivere i `<TIME …>` della domanda come relativi all'inizio della clip.

### 3. `nutrition_change`: scorciatoia nelle annotazioni (da documentare)
- [ ] **Dove:** relazione, sezione dataset / risultati
- **Fatto:** le opzioni sbagliate sono i valori corretti × moltiplicatori casuali, quindi hanno decimali.
  In **20/50** domande una sola opzione ha tutti i valori `.0`, ed è **sempre** quella corretta (20/20).
  Una strategia solo testuale farebbe circa 52% (20 + 30·0.2 su 50), contro il 16–22% dei modelli.
- **Azione:** riportarlo nella relazione (dice che i modelli non sfruttano nemmeno indizi testuali banali).

### 4. `image_nutrition_estimation`: l'ipotesi "non verificata" è verificabile (e confermata)
- [x] **Sistemato (13/11/2026).** Ricontrollato sui `metadata`: corretta = argmax valore assoluto 100/100 (nessun
  pareggio), = argmax `amount` 100/100, = argmax `*_amount_ratio` 0/100. Aggiornati README (sezione
  `image_nutrition_estimation`) e relazione (paragrafo nei risultati per prototipo). Il "non verificata" rimasto
  in relazione (sviluppi) riguarda il baseline testuale, cioè il #8.
- **Dove:** README, [relazione.tex:369-372](relazione/relazione.tex#L369-L372)
- **Fatto (dai `metadata`):** la risposta corretta è l'argmax del valore **assoluto** (quantità usata
  nella ricetta) in **100/100** domande, e l'argmax del valore **per grammo** (`*_amount_ratio`) in **0/100**.
  Il dataset è costruito apposta contro la conoscenza linguistica.
- **Azione:** togliere "non verificata" e citare il controllo.

---

## 🟠 Importanti

### 5. `image_nutrition_estimation`: input poco informativo
- [x] **Documentato (13/11/2026), nessuna modifica al protocollo** (decisione: restare fedeli al paper).
  Trovato in più: lo `stc_connector` di VideoLLaMA2-7B (Conv3d kernel/stride temporale 2, padding 1) fonde gli
  8 elementi a coppie `[pad,img1] [img2,img3] [img4,img5] [nero,nero] [nero,pad]`. Aggiunti in relazione una frase
  nel setup di VideoLLaMA 2 e il limite "Input a immagini"; una nota nel README.
- [x] L'immagine *i* corrisponde sempre all'opzione *i* (verificato 100/100), ma il prompt non lo dice
  e le 5 immagini arrivano al modello come "video".
- [x] VideoLLaMA2 aggiunge 3 frame neri per arrivare a `num_frames=8` (comportamento di `process_video`).
- **Fix possibili:** aggiungere al prompt "Image 1 → option A, …", oppure usare la modalità immagine
  invece di quella video. Almeno documentarlo nella relazione.

### 6. Pannelli incompleti / potenzialmente corrotti
- [x] ~~1 domanda di `nutrition_change` senza pannelli~~: non serve più, `nutrition_change` è escluso dai pannelli (vedi #2).
- [x] Dopo il riscaricamento dei 4 video danneggiati vanno **rigenerati** anche `_sampled` e pannelli.
  **Confermato dall'utente (13/11/2026): già rigenerati.**
  **Verificato (13/11/2026):** dei 4 (`P06-20240510-090328`, `P06-20240510-140459`, `P07-20240529-131737`,
  `P07-20240529-134410`) solo **`P06-20240510-140459`** serve a `_sampled`/pannelli (4 domande di
  `video_nutrition_estimation`); gli altri sono solo in `image_nutrition_estimation`, che usa frame estratti dal
  server. **Azione utente:** riscaricarlo, verificare l'MD5 (`d417ae46e51dfdb6960c2f5be0954169`), rigenerare
  `_sampled` e pannelli, sostituirli su Drive.
- [x] **Sistemato (13/11/2026):** i frame illeggibili vengono contati; fino al 2% sono sostituiti con il frame
  valido più vicino e segnalati, oltre si alza `IOError`, la cartella del video viene rimossa e la domanda
  risulta esclusa nel pre-flight. `save_panels_to_disk` svuota la cartella prima di scrivere (niente file vecchi
  mescolati). Testato su video sintetici: un mp4 troncato che prima dava 57/128 frame neri ora viene scartato.
- [x] ~~[mosaic.py:124-132](paper_mosaic_sampling/mosaic.py#L124-L132): i frame non leggibili diventano
  frame neri **senza avviso**.~~ Pannelli fatti da file troncati passano il pre-flight. Contare e stampare i
  frame falliti, e alzare un errore oltre una soglia.
- [x] ~~[mosaic.py:261-271](paper_mosaic_sampling/mosaic.py#L261-L271): abilitare `nutrition_change` nel `__main__`~~:
  non serve più (vedi #2). Resta solo da togliere il path assoluto (#14).
- [x] Il fallback a campionamento normale ([mosaic.py:136](paper_mosaic_sampling/mosaic.py#L136)) mescola
  frame singoli e pannelli senza che l'inferenza lo sappia: stampare quanti video ci finiscono.
  **Sistemato:** il riepilogo elenca per nome i video in fallback, quelli riparati e quelli scartati, e
  `process_video_folders` li restituisce.

### 7. Statistica assente
- [ ] Con n=50 l'intervallo di confidenza al 95% di un prototipo è circa **±14 punti**.
  Aggiungere intervalli di confidenza (bootstrap o Wilson) a `accuracy_report`.
- [ ] Base contro panels sono dati **appaiati**: aggiungere un test di McNemar in `compare_runs`.
- [ ] Nella relazione presentare il "38.7 = 38.7" di LLaVA come compatibile, non come riproduzione esatta.

### 8. Manca il baseline senza video
- [ ] **In corso (14/11/2026)** come ipotesi 1 della tesi: condizioni solo testo e video scambiato in
  [tesi_ipotesi1_video/](tesi_ipotesi1_video/README.md). Notebook pronti, run su Colab da fare.
- [ ] Un `infer_fn` che non passa input visivo (solo testo). Serve a capire quanto del 66–68% di
  `video_nutrition_estimation` viene dal video. È già indicato come sviluppo nella relazione: è poco costoso, farlo.

### 9. [download_missing.py](preprocessing_folder/download_missing.py)
- [ ] [:98-101](preprocessing_folder/download_missing.py#L98-L101): `--dry-run` esegue `dest.touch()` e
  **crea mp4 da 0 byte**, che al giro successivo contano come "presenti". Probabile origine dei file vuoti. Togliere il `touch()`.
- [ ] [:86](preprocessing_folder/download_missing.py#L86): considera presente qualunque file, anche vuoto
  o troncato. Controllare la dimensione o, meglio, l'MD5.
- [ ] `urlopen` senza `timeout`.
- [ ] `data/md5.txt` c'è ma nessuno script lo usa, mentre la relazione dice che i video sono stati verificati con MD5.
  Aggiungere la verifica MD5 dopo il download.
- [ ] Variabili `total` e `downloaded` calcolate ma mai usate.

---

## 🟡 Minori, igiene e riproducibilità

### 10. [id_videos_fetching.py](preprocessing_folder/id_videos_fetching.py)
- [ ] [Riga 20](preprocessing_folder/id_videos_fetching.py#L20): comando shell incollato nel file Python, quindi
  **SyntaxError**. È il primo comando del README.
- [ ] Legge solo il JSON di `image_nutrition_estimation`, con path relativo e output `./HD-EPIC` fisso.

### 11. [sampling_script.py](preprocessing_folder/sampling_script.py)
- [ ] [Riga 6](preprocessing_folder/sampling_script.py#L6): default `num_frames=124`, contro 128 alla riga 45.
- [ ] [Righe 79-81](preprocessing_folder/sampling_script.py#L79-L81): il `__main__` punta a `videos_3`, non alle
  cartelle del README. Usare argparse.
- [ ] Nessun controllo su `out.isOpened()`: senza codec `avc1` (es. opencv su Linux) non scrive nulla e non avvisa.
- [ ] I frame non letti vengono saltati, e il video esce con meno di 128 frame senza errore.
- [ ] Il seek con `CAP_PROP_POS_FRAMES` su H.264 può essere impreciso, e la ricodifica H.264 aggiunge
  artefatti (rilevante per leggere le cifre della bilancia). Valutare lettura sequenziale e/o salvataggio in JPEG.

### 12. Parser ([hdepic_eval.py](inference_folder/hdepic_eval.py)): l'articolo "a" viene letto come opzione A
- [ ] Livelli [149](inference_folder/hdepic_eval.py#L149), [154](inference_folder/hdepic_eval.py#L154),
  [168](inference_folder/hdepic_eval.py#L168) sono case-insensitive:
  - `"A carrot has more carbs"` → A
  - `"The answer is a mix…"` → A
  - `"I think it is a good choice"` → A
- **Impatto attuale: zero** (tutte le 598 risposte lette con `leading_letter`), ma va corretto:
  accettare solo lettere maiuscole nei livelli 2 e 4, e nel livello 1 escludere "A" seguita da una parola.

### 13. Ripresa dei run ([hdepic_eval.py:363-373](inference_folder/hdepic_eval.py#L363-L373))
- [ ] Non verifica la configurazione: se cambi `input_mode`, `system` o il numero di frame con lo stesso `tag`,
  riusa risultati vecchi senza avvisare. Salvare un hash della configurazione nel record e controllarlo.
- [ ] Le risposte in errore entrano nel DataFrame finale come **sbagliate** finché non si rilancia.
  Escluderle dal report o segnalarle in modo esplicito.

### 14. Riproducibilità e repository
- [ ] `git clone` di VideoLLaMA2 e LLaVA-NeXT senza commit fissato; diversi `pip install` senza versione
  (`peft`, `timm`, `einops`, `accelerate` lato LLaVA).
- [ ] `hdepic_eval` è in 3 copie (`.py` + due `%%writefile`). Oggi la logica è identica, ma è fragile:
  meglio copiarlo da Drive o da git invece di riscriverlo nella cella.
- [ ] Path assoluti fissi (`/Users/lorenzodidomenico/...`) in `sampling_script.py` e `mosaic.py`.
- [ ] `relazione/figure.jpg` non è nel repo, quindi `relazione.tex` non compila.
- [ ] `hd-epic-annotations/` è un repo git annidato e non tracciato, con JSON identici a quelli in
  `preprocessing_folder/`. Decidere se rimuoverlo, aggiungerlo come submodule o metterlo in `.gitignore`.
- [ ] LLaVA-Video: non usa la `time_instruction` dell'inferenza ufficiale ("The video lasts for X seconds…").
  Valutare se aggiungerla, insieme al fix del punto 2.
- [ ] README: doppio `---` prima di "Riferimenti".
