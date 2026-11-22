"""
L'oracolo e' un ingrediente che:
  1. ha un valore nutrizionale (calorie, carboidrati, proteine o grassi, quello
     chiesto dalla domanda) almeno OGNI_MOLTIPLICATORE volte piu' alto del
     valore migliore fra le 5 opzioni originali;
  2. NON compare nella ricetta del video (controllo sul nome, parola per parola);
  3. non e' gia' fra le 5 opzioni.

Tra tutti gli ingredienti che rispettano le regole, si prende quello con il
valore PIU' BASSO. Cosi' l'oracolo e' "molto migliore" ma non assurdo.
Lo script e' deterministico: rieseguendolo si ottiene lo stesso file.

Input  (dataset):
  data/hd-epic/nutrition_video_nutrition_estimation.json
  data/hd-epic/complete_recipes.json
Output:
  data/derived/oracolo/oracoli.json
"""
import json
import os
import re

# ----------------------------------------------------------- percorsi
# radice del progetto: la cartella sopra scripts/
RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATI_ORACOLO = os.path.join(RADICE, "data", "derived", "oracolo")
DOMANDE = os.path.join(RADICE, "data", "hd-epic", "nutrition_video_nutrition_estimation.json")
RICETTE = os.path.join(RADICE, "data", "hd-epic", "complete_recipes.json")
OUTPUT = os.path.join(DATI_ORACOLO, "oracoli.json")

# ----------------------------------------------------------- regole
OGNI_MOLTIPLICATORE = 2.0   # l'oracolo vale almeno 2 volte la migliore delle 5 opzioni
NUTRIENTI = ["calories", "carbs", "protein", "fat"]
PAROLE_VUOTE = {"of", "and", "the", "a", "in", "with", "&"}


def nutriente_della_domanda(testo):
    """Trova quale nutriente chiede la domanda, es. 'highest carbs' -> 'carbs'."""
    testo = testo.lower()
    for n in NUTRIENTI:
        if n in testo:
            return n
    return None


def parole(nome):
    """Parole di un nome, in minuscolo, senza parole vuote."""
    return {w for w in re.findall(r"[a-z]+", nome.lower()) if w not in PAROLE_VUOTE}


def valore(ingrediente, nutriente):
    """Valore numerico del nutriente, oppure None se non c'e' (N/A, testo, ...)."""
    v = ingrediente.get(nutriente)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def indice_ingredienti(ricette):
    """Un dizionario ing_id -> dati dell'ingrediente, su tutte le ricette."""
    indice = {}
    for chiave, ricetta in ricette.items():
        for cattura in ricetta.get("captures", []):
            for iid, ing in cattura.get("ingredients", {}).items():
                indice.setdefault(iid, ing)
    return indice


def ricetta_di(ing_id):
    """'P03_R01_I02' -> 'P03_R01'."""
    return ing_id.rsplit("_I", 1)[0]


def costruisci():
    with open(DOMANDE, encoding="utf-8") as f:
        domande = json.load(f)
    with open(RICETTE, encoding="utf-8") as f:
        ricette = json.load(f)

    indice = indice_ingredienti(ricette)

    # tutti gli ingredienti di ciascuna ricetta, per il controllo "non nel video"
    nomi_per_ricetta = {}
    for chiave, ricetta in ricette.items():
        nomi = set()
        for cattura in ricetta.get("captures", []):
            for ing in cattura.get("ingredients", {}).values():
                nomi |= parole(ing.get("name", ""))
        nomi_per_ricetta[chiave] = nomi

    # candidati: tutti gli ingredienti con un nome "semplice" e un valore numerico
    candidati = {n: [] for n in NUTRIENTI}
    for iid, ing in indice.items():
        nome = ing.get("name", "")
        if len(nome.split()) > 3 or re.search(r"[,&()/]", nome):
            continue   # niente nomi lunghi o con punteggiatura: l'oracolo deve sembrare un'opzione normale
        for n in NUTRIENTI:
            v = valore(ing, n)
            if v is not None:
                candidati[n].append((v, nome.lower(), iid))

    risultati, scartate = {}, []
    for chiave in sorted(domande):
        dom = domande[chiave]
        nutr = nutriente_della_domanda(dom["question"])
        if nutr is None:
            scartate.append({"domanda": chiave, "motivo": "nutriente non riconosciuto"})
            continue

        # ricetta della domanda: quella del distrattore "same recipe" (la maggioranza)
        ids_opzioni = [m["ing_id"] for m in dom["metadata"]]
        ricetta = max(set(map(ricetta_di, ids_opzioni)),
                      key=lambda r: sum(ricetta_di(i) == r for i in ids_opzioni))

        # valori delle 5 opzioni originali (dal loro ing_id)
        valori_opzioni = []
        for m in dom["metadata"]:
            ing = indice.get(m["ing_id"])
            v = valore(ing, nutr) if ing else None
            valori_opzioni.append(v)
        if any(v is None for v in valori_opzioni):
            scartate.append({"domanda": chiave, "motivo": "valori mancanti fra le opzioni"})
            continue
        massimo = max(valori_opzioni)
        if massimo <= 0:
            scartate.append({"domanda": chiave, "motivo": "massimo uguale a zero"})
            continue

        soglia = OGNI_MOLTIPLICATORE * massimo
        nomi_opzioni = {c.lower() for c in dom["choices"]}
        vietate = nomi_per_ricetta.get(ricetta, set())

        eleggibili = []
        for v, nome, iid in candidati[nutr]:
            if v < soglia or nome in nomi_opzioni:
                continue
            if parole(nome) & vietate:       # una parola in comune con la ricetta: scartato
                continue
            eleggibili.append((v, nome, iid))

        if not eleggibili:
            scartate.append({"domanda": chiave, "motivo": "nessun ingrediente eleggibile"})
            continue

        eleggibili.sort()                    # il valore piu' basso fra quelli sopra soglia
        v, nome, iid = eleggibili[0]
        risultati[chiave] = {
            "domanda": dom["question"],
            "nutriente": nutr,
            "ricetta": ricetta,
            "oracolo": nome,
            "oracolo_ing_id": iid,
            "valore_oracolo": v,
            "massimo_opzioni": massimo,
            "rapporto": round(v / massimo, 2),
            "opzioni_originali": dom["choices"],
            "lettera_gold_originale": "ABCDE"[dom["correct_idx"]],
        }

    os.makedirs(DATI_ORACOLO, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"regole": {"moltiplicatore_minimo": OGNI_MOLTIPLICATORE,
                              "scelta": "valore piu' basso sopra soglia"},
                   "oracoli": risultati, "scartate": scartate},
                  f, ensure_ascii=False, indent=2)

    print(f"Domande totali:  {len(domande)}")
    print(f"Oracoli creati:  {len(risultati)}")
    print(f"Scartate:        {len(scartate)}")
    for s in scartate:
        print("   ", s)
    print("File:", OUTPUT)


if __name__ == "__main__":
    costruisci()
