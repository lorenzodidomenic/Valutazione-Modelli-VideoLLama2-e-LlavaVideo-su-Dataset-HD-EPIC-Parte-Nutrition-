import os
import cv2
import numpy as np
import sys

def campiona_e_salva_video(video_path, output_path, num_frames=124, fps_output=1):
    print(f"🎬 Apertura video: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    totale_frame_originali = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    larghezza = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altezza = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Frame totali originari: {totale_frame_originali}")
    
    if totale_frame_originali <= 0:
        print("❌ Errore: Impossibile leggere i frame del video.")
        cap.release()
        return

    indici_frame = np.linspace(0, totale_frame_originali - 1, num=num_frames, dtype=int)
    
    # 'avc1' (H.264) risolve il problema del video verde su macOS con mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps_output, (larghezza, altezza))
    
    print(f"⏳ Estrazione di {num_frames} frame in corso...")
    
    count_salvati = 0
    for idx in indici_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            out.write(frame)
            count_salvati += 1
        else:
            print(f"⚠️ Impossibile leggere il frame all'indice {idx}")
            
    cap.release()
    out.release()
    
    print(f"✅ Salvati {count_salvati}/{num_frames} frame -> {output_path}")


def campiona_cartella(cartella_input, cartella_output, num_frames=128, fps_output=1):
    estensioni_video = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    if not os.path.isdir(cartella_input):
        print(f"❌ Cartella non trovata: {cartella_input}")
        return

    os.makedirs(cartella_output, exist_ok=True)

    video_files = [
        f for f in os.listdir(cartella_input)
        if os.path.splitext(f)[1].lower() in estensioni_video
    ]

    if not video_files:
        print("❌ Nessun video trovato nella cartella.")
        return

    print(f"📁 Trovati {len(video_files)} video in '{cartella_input}'")
    print(f"💾 Output in '{cartella_output}'\n")

    for i, nome_file in enumerate(sorted(video_files), 1):
        video_path = os.path.join(cartella_input, nome_file)
        nome_puro, estensione = os.path.splitext(nome_file)
        output_path = os.path.join(cartella_output, f"{nome_puro}_sampled{estensione}")

        print(f"[{i}/{len(video_files)}]", end=" ")
        campiona_e_salva_video(video_path, output_path, num_frames, fps_output)
        print()

    print(f"🏁 Completato! {len(video_files)} video campionati.")


if __name__ == "__main__":
    BASE = "/Users/lorenzodidomenico/Desktop/PROGETTO DEEP LEARNING 1"
    cartella_input  = os.path.join(BASE, "videos_3")
    cartella_output = os.path.join(BASE, "videos_3_sampled")

    campiona_cartella(cartella_input, cartella_output)