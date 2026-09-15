"""
Video Panels for Long Video Understanding (CVPR 2026)
Doorenbos, Spurio, Gall - https://arxiv.org/abs/2509.23724

Implementazione basata sul codice ufficiale: https://github.com/FedeSpu/Video-Panels
Adattata per pre-processare i video del dataset HD-EPIC in pannelli salvati su disco.

Output per ogni video: una sequenza di C immagini pannellate (griglie 2x2),
ognuna contenente panel_width * panel_height frame consecutivi.
"""

import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def stack_frames_grid(video, panel_width, panel_height, border_px):
    """
    Reshapes a video by stacking (panel_width x panel_height) frames into a grid.
    Riproduce fedelmente la funzione del paper.

    Args:
        video (np.ndarray): input video array di shape (N, H, W, C)
        panel_width (int): numero di frame per riga (alpha nel paper)
        panel_height (int): numero di frame per colonna (beta nel paper)
        border_px (int): bordo nero in pixel tra i frame (0 per riprodurre i numeri del paper)

    Returns:
        stacked_video (np.ndarray): video di shape (N // (w*h), H, W, C)
    """
    w, h = panel_width, panel_height
    D, H, W, Channels = video.shape
    frames_per_grid = w * h

    assert D % frames_per_grid == 0, \
        f"Numero di frame ({D}) deve essere divisibile per w*h ({w*h})"

    # Calcola lo spazio occupato dai bordi
    total_border_w = (w + 1) * border_px
    total_border_h = (h + 1) * border_px

    # Dimensione di ogni sotto-frame nella griglia
    panel_W = (W - total_border_w) // w
    panel_H = (H - total_border_h) // h

    new_num_frames = D // frames_per_grid
    stacked_video = np.zeros((new_num_frames, H, W, Channels), dtype=video.dtype)

    for i in range(new_num_frames):
        grid_frames = video[i * frames_per_grid: (i + 1) * frames_per_grid]
        frame_canvas = np.zeros((H, W, Channels), dtype=video.dtype)

        for idx in range(frames_per_grid):
            row = idx // w
            col = idx % w

            resized = cv2.resize(grid_frames[idx], (panel_W, panel_H),
                                 interpolation=cv2.INTER_AREA)

            y = (row + 1) * border_px + row * panel_H
            x = (col + 1) * border_px + col * panel_W

            frame_canvas[y:y + panel_H, x:x + panel_W] = resized

        stacked_video[i] = frame_canvas

    return stacked_video


def load_video_paneled(video_path, C=32, panel_width=2, panel_height=2,
                       border_px=0, fps_limit=1, verbose=False):
    """
    Carica un video e applica il Video Panels method.

    Se il video è abbastanza lungo (D > offset * C), produce C frame pannellati.
    Altrimenti, fa campionamento uniforme normale di C frame.

    Args:
        video_path (str): percorso al file video
        C (int): numero massimo di frame finali (context window del VLM)
        panel_width (int): alpha nel paper (default 2)
        panel_height (int): beta nel paper (default 2)
        border_px (int): bordo in pixel (default 0 come nel paper)
        fps_limit (int): gamma proporzionale - secondi minimi tra frame campionati
        verbose (bool): stampa info di debug

    Returns:
        frames (np.ndarray): array di shape (C, H, W, 3)
        was_paneled (bool): True se il paneling è stato applicato
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire il video: {video_path}")

    D = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    C_original = C
    C_total = C * panel_width * panel_height  # frame totali da estrarre per il paneling

    offset = fps_limit * fps

    def read_frames_sequential(cap, indices, total_frames):
        """
        Legge i frame scorrendo il video sequenzialmente (molto più veloce del seek random).
        Salta i frame non necessari con grab() invece di read().
        """
        indices_set = set(indices)
        idx_to_pos = {idx: i for i, idx in enumerate(indices)}
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames = [None] * len(indices)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        max_idx = max(indices)

        for frame_num in range(max_idx + 1):
            if frame_num in indices_set:
                ret, frame = cap.read()
                if ret:
                    frames[idx_to_pos[frame_num]] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frames[idx_to_pos[frame_num]] = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                cap.grab()  # salta il frame senza decodificarlo

        # Riempi eventuali None rimasti
        for i in range(len(frames)):
            if frames[i] is None:
                frames[i] = np.zeros((h, w, 3), dtype=np.uint8)

        return np.array(frames)

    if (D > offset * C_total) and (panel_width * panel_height > 1):
        # PANELING: il video è lungo abbastanza
        uniform_sampled_frames = np.linspace(0, D - 1, C_total, dtype=int)
        spare_frames = read_frames_sequential(cap, uniform_sampled_frames.tolist(), D)

        spare_frames = stack_frames_grid(spare_frames, panel_width, panel_height, border_px)

        if verbose:
            print(f"  Paneled: {spare_frames.shape}, grid={panel_width}x{panel_height}, "
                  f"D={D}, fps={fps:.1f}, durata={D/fps:.1f}s")
        cap.release()
        return spare_frames, True
    else:
        # SAMPLING NORMALE: video troppo corto per il paneling
        uniform_sampled_frames = np.linspace(0, D - 1, C_original, dtype=int)
        spare_frames = read_frames_sequential(cap, uniform_sampled_frames.tolist(), D)

        if verbose:
            print(f"  Normal sampling: {spare_frames.shape}, D={D}, "
                  f"durata={D/fps:.1f}s (troppo corto per paneling)")
        cap.release()
        return spare_frames, False


def save_panels_to_disk(frames, output_dir, video_name, was_paneled):
    """
    Salva i frame pannellati come immagini JPEG su disco.

    Se il video è stato pannellato, salva C immagini (una per panel).
    Se è stato campionato normalmente, salva i frame singoli.

    Args:
        frames (np.ndarray): array di shape (N, H, W, 3)
        output_dir (str): cartella di output
        video_name (str): nome base del video (senza estensione)
        was_paneled (bool): se il paneling è stato applicato
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        suffix = "panel" if was_paneled else "frame"
        filename = f"{video_name}_{suffix}_{i:04d}.jpg"
        output_path = os.path.join(output_dir, filename)

        img = Image.fromarray(frame)
        img.save(output_path, "JPEG", quality=95)


def process_video_folders(tasks_dict, C=32, panel_width=2, panel_height=2,
                          border_px=0, fps_limit=1, verbose=True):
    """
    Processa tutti i video nelle cartelle specificate.

    Args:
        tasks_dict (dict): dizionario {task_name: {"video_dir": path}}
        C (int): numero di frame finali per il VLM
        panel_width (int): alpha (default 2)
        panel_height (int): beta (default 2)
        border_px (int): bordo (default 0)
        fps_limit (int): stride temporale (default 1)
        verbose (bool): output dettagliato
    """
    stats = {"paneled": 0, "normal": 0, "errors": 0}

    for task_name, paths in tasks_dict.items():
        input_dir = paths["video_dir"]
        output_base = input_dir.rstrip("/") + "_panels"
        os.makedirs(output_base, exist_ok=True)

        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        video_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(video_extensions) and not f.startswith('.')
        ])

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_base}")
        print(f"Video trovati: {len(video_files)}")
        print(f"Parametri: C={C}, grid={panel_width}x{panel_height}, "
              f"fps_limit={fps_limit}, border_px={border_px}")
        print(f"{'='*60}")

        for video_file in tqdm(video_files, desc=f"{task_name}"):
            video_path = os.path.join(input_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_output_dir = os.path.join(output_base, video_name)

            try:
                frames, was_paneled = load_video_paneled(
                    video_path, C=C,
                    panel_width=panel_width,
                    panel_height=panel_height,
                    border_px=border_px,
                    fps_limit=fps_limit,
                    verbose=verbose
                )

                save_panels_to_disk(frames, video_output_dir, video_name, was_paneled)

                if was_paneled:
                    stats["paneled"] += 1
                else:
                    stats["normal"] += 1

            except Exception as e:
                print(f"\n  ERRORE con {video_file}: {e}")
                stats["errors"] += 1

    print(f"\n{'='*60}")
    print(f"RIEPILOGO:")
    print(f"  Pannellati: {stats['paneled']}")
    print(f"  Sampling normale (video corti): {stats['normal']}")
    print(f"  Errori: {stats['errors']}")
    print(f"{'='*60}")


# =====================================================================
# CONFIGURAZIONE
# =====================================================================
if __name__ == "__main__":
    BASE_PATH = "/Users/lorenzodidomenico/Desktop/PROGETTO DEEP LEARNING 1/"

    # Cartelle dei video NON sampled
    my_dataset_tasks = {
        # "image_nutrition_estimation": {
        #     "video_dir": os.path.join(BASE_PATH, "videos_image_nutrition_estimation"),
        # },
        # "nutrition_change": {
        #     "video_dir": os.path.join(BASE_PATH, "videos_nutrition_change"),
        # },
        "nutrition_estimation": {
            "video_dir": os.path.join(BASE_PATH, "videos_nutrition_estimation"),
        },
    }

    # Parametri dal paper (Tabella 1, configurazione migliore):
    # C = 32 (context window tipico per LLaVA-OneVision, Qwen2.5-VL, etc.)
    # panel_width = 2 (alpha)
    # panel_height = 2 (beta)
    # fps_limit = 1 (gamma)
    # border_px = 0 (per riprodurre i risultati del paper)
    process_video_folders(
        my_dataset_tasks,
        C=32,
        panel_width=2,
        panel_height=2,
        border_px=0,
        fps_limit=1,
        verbose=True
    )