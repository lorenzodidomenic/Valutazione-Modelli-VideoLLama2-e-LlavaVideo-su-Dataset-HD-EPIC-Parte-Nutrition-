import os
import cv2
import numpy as np
import sys

def campiona_e_salva_video(video_path, output_path, num_frames=128, fps_output=1):
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
    import argparse
    # radice del progetto: <radice>/src/hdepic/preprocessing/sampling.py
    RADICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    parser = argparse.ArgumentParser(
        description="Campiona 128 fotogrammi uniformi da ogni video di una cartella (1 fps in uscita).")
    parser.add_argument("input", nargs="?", default="videos_nutrition_estimation",
                        help="cartella dei video integrali (default: videos_nutrition_estimation)")
    parser.add_argument("output", nargs="?", default=None,
                        help="cartella di destinazione (default: <input>_sampled)")
    parser.add_argument("--num-frames", type=int, default=128)
    args = parser.parse_args()

    # percorsi relativi = relativi alla radice del progetto
    cartella_input = args.input if os.path.isabs(args.input) else os.path.join(RADICE, args.input)
    cartella_output = args.output or cartella_input.rstrip("/") + "_sampled"
    if not os.path.isabs(cartella_output):
        cartella_output = os.path.join(RADICE, cartella_output)

    campiona_cartella(cartella_input, cartella_output, num_frames=args.num_frames)