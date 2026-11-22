
## Ipotesi

Su `video_nutrition_estimation` i due modelli hanno come percentuale di risposte giuste nel baseline del 66% e 68% (5 opzioni, caso al 20%), molto più che negli altri due prototipi.

Il test aggiunge una sesta opzione, l'oracolo: un ingrediente con almeno il doppio del nutriente chiesto, ma **assente dalla ricetta del video**. Un modello che guarda il video dovrebbe scartarlo; un modello che ragiona solo sul testo dovrebbe sceglierlo spesso. La prova vera è il confronto fra le due condizioni: `O0_video` (video più opzioni) e `O1_testo`.

## Risultati (49 domande, caso 16,7%)

| | Oracolo con video | Oracolo solo testo | Accuratezza con video | Accuratezza solo testo | Stessa lettera nelle due condizioni |
|---|---|---|---|---|---|
| **VideoLLaMA 2** | 32,7% | 28,6% | 38,8% | 40,8% | **39/49** |
| **LLaVA-Video** | 16,3% | 46,9% | 51,0% | 34,7% | 23/49 |



## Cosa si può concludere

**VideoLLaMA 2: ipotesi verificata.**
- Con e senza video dà la stessa lettera in 4 domande su 5, e l'accuratezza è uguale (38,8% contro 40,8%).
- Con il video davanti sceglie l'ingrediente assente nel 33% dei casi, il doppio del caso (p ≈ 0,005 col test binomiale).
- Basta aggiungere un'opzione assente per farlo scendere dal 66% al 39%. Il suo 66% veniva in gran parte dal testo.

**LLaVA-Video: ipotesi smentita**
- Solo col testo sceglie l'oracolo quasi una volta su due (46,9%, IC 34–61%), quindi la conoscenza linguistica lo spinge verso l'opzione più calorica.
- Con il video quella scelta scende al livello del caso (16,3%): usa il video almeno per capire quali ingredienti ci sono.
- L'accuratezza sale da 34,7% a 51,0%, ma con 49 domande la differenza non è significativa (p ≈ 0,10).
- Però il 34,7% solo testo è il doppio del caso, quindi una parte del 68% si spiega comunque col testo.

