#!/usr/bin/env python3
"""
Script para descargar modelos de MediaPipe manualmente
"""

import urllib.request
from pathlib import Path
import sys

MODELS = {
    "hand_landmarker.task": [
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ],
    "pose_landmarker_lite.task": [
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ],
    "face_landmarker.task": [
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ]
}

def download_model(model_name: str, urls: list, output_dir: Path) -> bool:
    """Intenta descargar un modelo desde múltiples URLs"""
    output_path = output_dir / model_name
    
    if output_path.exists():
        print(f"✓ {model_name} ya existe en {output_path}")
        return True
    
    print(f"Descargando {model_name}...")
    for url in urls:
        try:
            print(f"  Intentando: {url}")
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ {model_name} descargado exitosamente")
            return True
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"✗ No se pudo descargar {model_name} desde ninguna URL")
    return False

def main():
    """Función principal"""
    # Directorio de salida
    models_dir = Path(__file__).parent.parent / "models" / "mediapipe"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Descargando modelos de MediaPipe a: {models_dir}\n")
    
    success_count = 0
    for model_name, urls in MODELS.items():
        if download_model(model_name, urls, models_dir):
            success_count += 1
        print()
    
    print(f"Resultado: {success_count}/{len(MODELS)} modelos descargados")
    
    if success_count < len(MODELS):
        print("\nAlgunos modelos no se pudieron descargar.")
        print("Alternativas:")
        print("1. Descarga manualmente desde: https://developers.google.com/mediapipe/solutions/vision")
        print(f"2. Coloca los archivos .task en: {models_dir}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

