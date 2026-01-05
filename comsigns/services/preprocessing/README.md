# Módulo de Preprocessing

Módulo para extracción de keypoints usando MediaPipe.

## Uso

### Extraer keypoints de un video

```python
from comsigns.services.preprocessing import process_video_clip

feature_clip = process_video_clip(
    video_path="video.mp4",
    fps=30.0,
    normalize=True,
    format="json"
)
```

### Extraer keypoints manualmente

```python
from comsigns.services.preprocessing import KeypointExtractor

extractor = KeypointExtractor()
feature_clip = extractor.extract_from_video("video.mp4", fps=30.0)
```

## Funcionalidades

- Extracción de keypoints de manos (21 puntos por mano, hasta 2 manos)
- Extracción de keypoints del cuerpo (33 puntos)
- Extracción de keypoints del rostro (468 puntos)
- Normalización de keypoints (relativa o absoluta)
- Guardado en formato JSON o Parquet

## MediaPipe

Este módulo utiliza:
- `mediapipe.solutions.hands` para detección de manos
- `mediapipe.solutions.pose` para detección de pose corporal
- `mediapipe.solutions.face_mesh` para detección facial

