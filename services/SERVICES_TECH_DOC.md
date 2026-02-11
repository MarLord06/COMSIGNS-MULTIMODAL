# ⚙️ Documentación Técnica — Carpeta `services`

> Guía técnica de los módulos runtime: preprocessing, encoder, inference y API.

---

## 📖 Navegación

| Documento | Descripción |
|-----------|-------------|
| [🧠 Arquitectura del Modelo](../docs/MODEL_ARCHITECTURE.md) | Encoder multimodal y clasificador |
| [📘 Documentación Técnica](../docs/MODEL_TECHNICAL.md) | I/O, inferencia, limitaciones |
| [🏋️ Entrenamiento](../docs/TRAINING.md) | Trainer, métricas, checkpointing |
| [🏗️ Arquitectura General](../comsigns/docs/ARCHITECTURE.md) | Pipeline + dataset + resultados |
| [🌐 Inferencia Web](../comsigns/docs/WEB_INFERENCE.md) | API REST + Frontend |

### READMEs de Servicios

| Servicio | README |
|----------|--------|
| API | [api/README.md](../comsigns/services/api/README.md) |
| Encoder | [encoder/README.md](../comsigns/services/encoder/README.md) |
| Ingestion | [ingestion/README.md](../comsigns/services/ingestion/README.md) |
| Preprocessing | [preprocessing/README.md](../comsigns/services/preprocessing/README.md) |

---

## Módulos Inspeccionados

| Archivo | Propósito |
|---------|-----------|
| `services/preprocessing/extract_keypoints.py` | Extracción MediaPipe |
| `services/preprocessing/process_clip.py` | Normalización y guardado |
| `services/encoder/model.py` | `MultimodalEncoder` y ramas |
| `services/encoder/utils.py` | Conversión `FeatureClip` → tensores |
| `services/inference/loader.py` | Carga de checkpoints y mapeos |
| `services/inference/predictor.py` | Wrapper de inferencia y postprocesado |
| `services/api/main.py` | API FastAPI (REST + WebSocket) |
| `services/schemas.py` | Pydantic schemas y contratos |

---

## 1. Preprocessing — Extracción de Keypoints (MediaPipe)

**Clase principal:** `KeypointExtractor`

### API Pública

```python
KeypointExtractor.__init__(model_paths: Optional[Dict[str,str]] = None)
extract_from_frame(frame: np.ndarray, timestamp_ms: int = 0) -> FrameKeypoints
extract_from_video(video_path: str, fps: Optional[float] = None) -> FeatureClip
```

### Snippet

```python
def _extract_hand_keypoints(self, mp_image, timestamp_ms: int) -> List[List[float]]:
    if hasattr(self.hand_landmarker, 'detect_for_video'):
        res = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    else:
        res = self.hand_landmarker.detect(mp_image)
    keypoints = []
    if res.hand_landmarks:
        for hand_landmarks in res.hand_landmarks:
            for landmark in hand_landmarks:
                keypoints.append([landmark.x, landmark.y, landmark.z])
    return keypoints
```

> [!NOTE]
> Los landmarks están normalizados [0,1] (x,y), z es relativo. El extractor NO incluye canal de confidence.

**Setup:** [MODELS_SETUP.md](../comsigns/MODELS_SETUP.md)

---

## 2. Preprocessing — Normalización y Guardado

### API Pública

```python
normalize_keypoints(keypoints, method='relative'|'absolute')
process_video_clip(video_path, output_path=None, fps=None, normalize=True, format='json') -> FeatureClip
```

Output: `FeatureClip.to_dict()` para JSON; DataFrame por frame para Parquet.

---

## 3. Schemas — Contrato de Datos (Pydantic)

### Objetos Principales

| Schema | Descripción |
|--------|-------------|
| `Keypoint` | Validación x,y ∈ [0,1], z opcional, `confidence` ∈ [0,1] |
| `FrameKeypoints` | `t`, `hand_keypoints`, `body_keypoints`, `face_keypoints` |
| `FeatureClip` | `clip_id`, `fps`, `frames: List[FrameKeypoints]`, `meta` |

> [!IMPORTANT]
> Los servicios asumen que `FeatureClip.frames` tiene al menos 1 frame. Si produce 0, manejar como error upstream.

---

## 4. Encoder — `MultimodalEncoder`

**Archivo:** [`comsigns/services/encoder/model.py`](../comsigns/services/encoder/model.py)

### API Pública

```python
MultimodalEncoder.forward(hand_keypoints, body_keypoints, face_keypoints) -> Tensor
# Output: (batch, seq_len, output_dim)

create_encoder(config_path=None, **kwargs) -> MultimodalEncoder
```

### Forward (Fusión)

```python
hand_emb = self.hand_branch(hand_keypoints)
body_emb = self.body_branch(body_keypoints)
face_emb = self.face_branch(face_keypoints)
fused = torch.cat([hand_emb, body_emb, face_emb], dim=-1)
output = self.fusion(fused)
return output
```

**Arquitectura detallada:** [MODEL_ARCHITECTURE.md](../docs/MODEL_ARCHITECTURE.md)

---

## 5. Encoder Utils — FeatureClip → Tensores

**Archivo:** `comsigns/services/encoder/utils.py`

### API Pública

```python
keypoints_to_tensor(keypoints, expected_size=None, pad_value=0.0) -> torch.Tensor
feature_clip_to_tensors(feature_clip) -> Dict['hand','body','face']
# Cada tensor es (seq_len, input_dim)
```

> [!IMPORTANT]
> Garantiza vectores de tamaño fijo: `hand=126`, `body=99`, `face=1404`. El encoder depende de estas formas.

---

## 6. Inference — Carga de Checkpoint y Predictor

### `InferenceLoader`

```python
from services.encoder.model import MultimodalEncoder
from training.classifier import SignLanguageClassifier

encoder = MultimodalEncoder()
model = SignLanguageClassifier(encoder=encoder, num_classes=num_classes)
model.load_state_dict(model_state)
model = model.to(self.device).eval()
```

### `Predictor`

```python
logits = self.model(hand, body, face, lengths)
scores = F.softmax(logits.flatten(), dim=0)
topk_scores, topk_indices = torch.topk(scores, k)
```

> [!NOTE]
> `Predictor.predict` asume batch size 1. Para batch: `predict_batch`.

---

## 7. API — FastAPI REST + WebSocket

**Archivo:** [`comsigns/services/api/main.py`](../comsigns/services/api/main.py)

### Endpoints

| Endpoint | Descripción |
|----------|-------------|
| `POST /infer/video` | Recibe video → extrae keypoints → encoder → embeddings |
| `WebSocket /ws/infer` | Frames base64 en tiempo real → glosador → traductor |

### Flujo de `/infer/video`

```
client → UploadFile → save temp → validate_video() → process_video_clip() → feature_clip_to_tensors() → encoder() → return embeddings
```

> [!WARNING]
> - **Lazy loading:** Los modelos se cargan la primera vez; manejar concurrencia con múltiples workers.
> - **WebSocket:** Guarda frames en temp files (simple pero ineficiente para producción).

---

## 8. Flujo de Datos Integrado

```
Entrada: video (archivo) o frames base64 (WebSocket)
    ↓
KeypointExtractor → FeatureClip (Pydantic) → feature_clip_to_tensors → encoder tensors
    ↓
MultimodalEncoder → embeddings (batch, seq_len, output_dim)
    ↓
SignLanguageClassifier → logits   ─── ó ───   Glosador + Translator → texto
    ↓
Predictor → top-k predicciones
```

---

## 9. Configuraciones Relevantes

| Key | Uso |
|-----|-----|
| `preprocessing.mediapipe.min_detection_confidence` | Umbral de detección |
| `preprocessing.mediapipe.static_image_mode` | Modo VIDEO vs IMAGE |
| `encoder.hidden_dim`, `encoder.output_dim` | Dimensiones del encoder |
| `api.host`, `api.port` | Configuración de uvicorn |

---

## 10. Errores Comunes y Mitigaciones

| Error | Mitigación |
|-------|------------|
| MediaPipe no instalado / modelos faltantes | Añadir `*.task` en `models/mediapipe` — ver [MODELS_SETUP.md](../comsigns/MODELS_SETUP.md) |
| `FeatureClip` vacío (0 frames) | Validar antes de proseguir (HTTP 500) |
| Desajuste de dimensiones (126/99/1404) | `keypoints_to_tensor` pad/trunca; mejor asegurar extracción completa |
| Carga concurrente de modelos | Usar bloqueo o cargar en proceso maestro |

---

## 11. Ejemplos Rápidos

### Extraer Features

```python
from comsigns.services.preprocessing.process_clip import process_video_clip

fc = process_video_clip('video.mp4', format='json')
print(fc.clip_id, len(fc.frames))
```

### Inferencia con Checkpoint

```python
from comsigns.services.inference.loader import load_checkpoint_for_inference

model, class_names, other_id, info = load_checkpoint_for_inference(
    Path('experiments/run_001/checkpoints/best.pt')
)
```

---

## 📚 Documentos Relacionados

- [🧠 Arquitectura del Modelo](../docs/MODEL_ARCHITECTURE.md) — Diagramas detallados
- [📘 Documentación Técnica](../docs/MODEL_TECHNICAL.md) — I/O, inferencia
- [🏋️ Entrenamiento](../docs/TRAINING.md) — Trainer, métricas
- [🌐 Inferencia Web](../comsigns/docs/WEB_INFERENCE.md) — API + Frontend
- [📜 Scripts](../comsigns/docs/SCRIPTS_USAGE.md) — CLI flags
