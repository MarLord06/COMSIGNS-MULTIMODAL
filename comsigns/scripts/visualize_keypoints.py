
# Fix: imports necesarios antes de usarlos
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Fix: imports necesarios antes de usarlos

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from core.data.datasets.aec import AECDataset
from services.preprocessing import KeypointExtractor

# Cambia estos paths según el video que quieras analizar

import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualiza keypoints de un video de signos.")
    parser.add_argument('-v', '--video', type=str, default="data/raw/lsp_aec/videos/SEGMENTED_SIGN/ira_alegria/no_144.mp4",
                        help="Ruta al archivo de video a analizar")
    parser.add_argument('--no-show', action='store_true', help='No mostrar ventanas de matplotlib; imprimir shapes en consola')
    args = parser.parse_args()
    video_path = args.video
    extractor = KeypointExtractor()
    feature_clip = extractor.extract_from_video(video_path)

    print(f"Frames: {len(feature_clip.frames)}")
    for i, frame in enumerate(feature_clip.frames):
        hand = np.array(frame.hand_keypoints, dtype=float)
        body = np.array(frame.body_keypoints, dtype=float)
        face = np.array(frame.face_keypoints, dtype=float)

        def plot_kp(ax, arr, title):
            if arr.size == 0:
                ax.text(0.5, 0.5, 'No keypoints', ha='center')
                return
            # Ensure we only plot x,y,z channels (first 3 columns)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 3)
            if arr.shape[1] > 3:
                arr = arr[:, :3]
            # Plot each coordinate separately
            for c in range(arr.shape[1]):
                ax.plot(arr[:, c], label=['x', 'y', 'z'][c])
            ax.legend(loc='upper right')
            ax.set_title(title)

        plt.figure(figsize=(12, 3))
        plt.subplot(1, 3, 1)
        plot_kp(plt.gca(), hand, f"Hand {i}")
        plt.subplot(1, 3, 2)
        plot_kp(plt.gca(), body, f"Body {i}")
        plt.subplot(1, 3, 3)
        plot_kp(plt.gca(), face, f"Face {i}")
        plt.tight_layout()
        if args.no_show:
            print(f"Frame {i}: hand={hand.shape}, body={body.shape}, face={face.shape}")
        else:
            plt.show()
        if i >= 4:
            break

if __name__ == "__main__":
    main()
