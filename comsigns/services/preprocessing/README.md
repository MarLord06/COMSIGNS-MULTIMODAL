# 🔧 Módulo de Preprocessing

> Extracción de keypoints usando MediaPipe para manos, cuerpo y rostro.

---

## 📖 Ver También

| Documento | Descripción |
|-----------|-------------|
| [🔧 Setup MediaPipe](../../MODELS_SETUP.md) | Descarga de modelos `.task` |
| [🧠 Arquitectura del Modelo](../../../docs/MODEL_ARCHITECTURE.md) | Cómo se usan los keypoints en el encoder |
| [⚙️ Servicios](../../../services/SERVICES_TECH_DOC.md) | Docs técnicos detallados |
| [📹 Ingestion](../ingestion/README.md) | Paso anterior: captura de video |
| [🧠 Encoder](../encoder/README.md) | Paso siguiente: encoder multimodal |

---

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

---

## Funcionalidades

| Funcionalidad | Detalle |
|---------------|---------|
| Manos | 21 keypoints por mano, hasta 2 manos |
| Cuerpo | 33 keypoints |
| Rostro | 468 keypoints |
| Normalización | Relativa o absoluta |
| Formatos de salida | JSON o Parquet |

---

## MediaPipe

Este módulo utiliza:
- `mediapipe.solutions.hands` para detección de manos
- `mediapipe.solutions.pose` para detección de pose corporal
- `mediapipe.solutions.face_mesh` para detección facial

> [!TIP]
> Si los modelos no se descargan automáticamente, ver [MODELS_SETUP.md](../../MODELS_SETUP.md).

---

## 📚 Docs Relacionados

- [🔧 Setup MediaPipe](../../MODELS_SETUP.md) — Descarga de modelos
- [🧠 Arquitectura](../../../docs/MODEL_ARCHITECTURE.md) — Shapes y diagramas
- [📹 Ingestion](../ingestion/README.md) — Paso anterior
- [🧠 Encoder](../encoder/README.md) — Paso siguiente
