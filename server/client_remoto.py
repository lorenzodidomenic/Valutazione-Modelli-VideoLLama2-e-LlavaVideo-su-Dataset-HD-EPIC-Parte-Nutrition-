"""
client_remoto - chiama dal Mac il server del modello acceso sul pod RunPod.

Solo libreria standard: non serve installare nulla.

Nel notebook prende il posto delle celle di installazione e caricamento del modello:

    import client_remoto as R
    srv = R.Remoto("https://<POD_ID>-8001.proxy.runpod.net", chiave="...")
    print(srv.salute())                       # versioni e GPU, da annotare
    VIDEO_INDEX, READABLE = srv.indice()      # i video stanno sul pod
    infer_video, infer_testo = srv.infer_video, srv.infer_testo

infer_video(visual, prompt) e infer_testo(prompt) hanno la stessa firma delle funzioni dei
notebook, quindi hdepic.eval e hdepic.oracolo non cambiano. Un errore del server solleva
un'eccezione: run_task la registra come risposta fallita e la rifa al rilancio.

Prova veloce da terminale:
    python3 server/client_remoto.py https://<POD_ID>-8001.proxy.runpod.net <CHIAVE>
"""
import json
import sys
import time
import urllib.error
import urllib.request


class ErroreServer(RuntimeError):
    pass


class Remoto:
    def __init__(self, url, chiave, timeout=95, tentativi=3):
        # timeout sotto i ~100 s dopo cui il proxy di RunPod chiude la richiesta
        self.url = url.rstrip("/")
        self.chiave = chiave
        self.timeout = timeout
        self.tentativi = tentativi

    def _chiama(self, metodo, percorso, corpo=None):
        dati = None if corpo is None else json.dumps(corpo).encode()
        for i in range(self.tentativi):
            req = urllib.request.Request(
                self.url + percorso, data=dati, method=metodo,
                headers={"X-Chiave": self.chiave, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                dettaglio = e.read().decode(errors="replace")[:500]
                # 502/503/504: proxy di RunPod o server ancora in caricamento -> si riprova
                if e.code in (502, 503, 504) and i < self.tentativi - 1:
                    time.sleep(5 * (i + 1))
                    continue
                raise ErroreServer(f"HTTP {e.code}: {dettaglio}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if i < self.tentativi - 1:
                    time.sleep(5 * (i + 1))
                    continue
                raise ErroreServer(f"server non raggiungibile: {e}") from None

    def salute(self):
        return self._chiama("GET", "/salute")

    def indice(self, aggiorna=False):
        """-> (VIDEO_INDEX, READABLE), con i percorsi dei video sul pod."""
        d = self._chiama("GET", "/indice" + ("?aggiorna=true" if aggiorna else ""))
        idx, ok = d["video_index"], d["leggibili"]
        rotti = sum(1 for v in ok.values() if not v)
        print(f"cartelle sul pod: {d['cartelle']}")
        print(f"video campionati: {len(idx['sampled'])} | integrali: {len(idx['full'])} | "
              f"non decodificabili: {rotti}")
        return idx, ok

    def infer_video(self, visual, prompt):
        """visual: percorso di un mp4 sul pod, oppure lista di percorsi di immagini."""
        return self._chiama("POST", "/infer",
                            {"modal": "video", "visual": visual, "prompt": prompt})["risposta"]

    def infer_testo(self, prompt):
        """Solo testo: nessun input visivo."""
        return self._chiama("POST", "/infer", {"modal": "text", "prompt": prompt})["risposta"]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("uso: python3 client_remoto.py <URL> <CHIAVE>")
    srv = Remoto(sys.argv[1], sys.argv[2])

    print("== salute")
    print(json.dumps(srv.salute(), indent=2))

    print("\n== indice")
    idx, ok = srv.indice()

    print("\n== solo testo")
    t = time.time()
    print(repr(srv.infer_testo("Which has more protein, lentils or cucumber?\n"
                               "A. lentils\nB. cucumber\nAnswer with the option's letter.")),
          f"({time.time() - t:.1f} s)")

    video = next((p for p in idx["sampled"].values() if ok.get(p)), None)
    if video:
        print(f"\n== video ({video})")
        t = time.time()
        print(repr(srv.infer_video(video, "Describe briefly what the person is cooking.")),
              f"({time.time() - t:.1f} s)")
    else:
        print("\nnessun video decodificabile sul pod: controlla la cartella DATI")
