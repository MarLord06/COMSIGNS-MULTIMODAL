# Documentación Técnica de la Carpeta `services`

La carpeta `services` contiene la implementación runtime de los pipelines multimodales del proyecto: extracción y preprocesado de keypoints, conversión a tensores, encoder multimodal, carga de checkpoints y APIs de inferencia/streaming.

Este documento reemplaza la descripción superficial con una guía técnica accionable: lista de módulos reales, APIs públicas, snippets de código relevantes, flujos de datos (entrada → salida), configuraciones clave y puntos de fallo conocidos.

**Archivos / módulos inspeccionados**
- `comsigns/services/preprocessing/extract_keypoints.py` — extracción MediaPipe
- `comsigns/services/preprocessing/process_clip.py` — normalización y guardado
- `comsigns/services/encoder/model.py` — implementa `MultimodalEncoder` y ramas
- `comsigns/services/encoder/utils.py` — conversión `FeatureClip` → tensores
- `comsigns/services/inference/loader.py` — carga de checkpoints y mapeos de clases
- `comsigns/services/inference/predictor.py` — wrapper de inferencia y postprocesado
- `comsigns/services/api/main.py` — API FastAPI (endpoints REST + WebSocket)
- `comsigns/services/schemas.py` — Pydantic schemas y contratos de datos

**Cómo leer este documento**
- Cada sección incluye: propósito, API pública (clases/funciones), snippet representativo y ejemplos de uso.
- Los snippets están extraídos del código real en la carpeta `comsigns/services`.

---

**1) Preprocessing — extracción de keypoints (MediaPipe)**

- Propósito: leer frames (imagen/archivo), ejecutar MediaPipe Tasks (hands, pose, face) y devolver un `FeatureClip` validado para downstream.
- Principal clase: `KeypointExtractor`
- API pública relevante:
	- `KeypointExtractor.__init__(model_paths: Optional[Dict[str,str]] = None)` — inicializa landmarkers.
	- `extract_from_frame(frame: np.ndarray, timestamp_ms: int = 0) -> FrameKeypoints`
	- `extract_from_video(video_path: str, fps: Optional[float] = None) -> FeatureClip`

Snippet (extracción de mano / cuerpo / rostro, truncado):

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

Notas técnicas y recomendaciones:
- Los landmarks devueltos están normalizados en [0,1] (x,y) y z es relativo; el extractor NO incluye un canal de confidence en la entrada a los modelos (se descarta para el encoder).
- `KeypointExtractor` intenta descargar modelos de MediaPipe si no se proporcionan rutas; en entornos sin internet conviene montar los `*.task` en `models/mediapipe`.
- Modo `VIDEO` vs `IMAGE`: `running_mode` depende de `preprocessing.mediapipe.static_image_mode` en la config.

---

**2) Preprocessing — normalización y guardado**

- Propósito: normalizar keypoints por frame y almacenar en `json` o `parquet`.
- Funciones clave:
	- `normalize_keypoints(keypoints, method='relative'|'absolute')`
	- `process_video_clip(video_path, output_path=None, fps=None, normalize=True, format='json') -> FeatureClip`

Snippet (normalización relativa/absoluta):

```python
def normalize_keypoints(keypoints, method='relative'):
		if method == 'relative':
				# clamp x,y a [0,1] y devolver [x,y,z,confidence]
				return [[max(0, min(1, kp[0])), max(0, min(1, kp[1])), kp[2] if len(kp)>2 else 0.0, kp[3] if len(kp)>3 else 1.0] for kp in keypoints]
		elif method == 'absolute':
				# centrar en el centroide y escalar por la distancia máxima
				...
```

Consideraciones:
- `process_video_clip` delega en `KeypointExtractor` y luego aplica `normalize_keypoints` por frame.
- Output: `FeatureClip.to_dict()` cuando se guarda a JSON; para Parquet se convierte a un DataFrame por frame.

---

**3) Schemas — contrato de datos (Pydantic)**

- Archivos: `comsigns/services/schemas.py`
- Objetos principales:
	- `Keypoint` : validación de x,y en [0,1], z opcional y `confidence` [0,1]
	- `FrameKeypoints` : `t`, `hand_keypoints`, `body_keypoints`, `face_keypoints`
	- `FeatureClip` : `clip_id`, `fps`, `frames: List[FrameKeypoints]`, `meta`

Snippet (validador de frames ordenados):

```python
@field_validator('frames')
def validate_frames_ordered(cls, v):
		times = [f.t for f in v]
		if times != sorted(times):
				raise ValueError('Los frames deben estar ordenados por tiempo (t)')
		return v
```

Importante:
- Los servicios asumen que `FeatureClip.frames` tiene al menos 1 frame (`min_length=1`). Si un pipeline produce 0 frames, se debe manejar como error upstream.

---

**4) Encoder — `MultimodalEncoder` (arquitectura)**

- Archivo: `comsigns/services/encoder/model.py`
- Diseño: 3 ramas (`HandBranch`, `BodyBranch`, `FaceBranch`) que producen embeddings de dimensión `hidden_dim` cada una; se concatenan y proyectan a `output_dim`.
- API pública:
	- `MultimodalEncoder.forward(hand_keypoints, body_keypoints, face_keypoints) -> Tensor` (batch, seq_len, output_dim)
	- `create_encoder(config_path=None, **kwargs) -> MultimodalEncoder`

Snippet (forward fusion):

```python
hand_emb = self.hand_branch(hand_keypoints)
body_emb = self.body_branch(body_keypoints)
face_emb = self.face_branch(face_keypoints)
fused = torch.cat([hand_emb, body_emb, face_emb], dim=-1)
output = self.fusion(fused)
return output
```

Puntos técnicos:
- Las ramas usan LSTM (batch_first=True) y una proyección inicial `Linear` para reducir/expandir dimensiones.
- `FaceBranch` aplica una reducción inicial debido a la alta dimensionalidad (468*3 = 1404).
- El encoder espera tensores con forma `(batch, seq_len, input_dim)`. Para inferencia en endpoints que usan clips, el código añade dimensión de batch antes de pasar al encoder.

---

**5) Encoder utils — conversión FeatureClip → tensores**

- Archivo: `comsigns/services/encoder/utils.py`
- Funciones clave:
	- `keypoints_to_tensor(keypoints, expected_size=None, pad_value=0.0) -> torch.Tensor`
	- `feature_clip_to_tensors(feature_clip) -> Dict['hand','body','face']` (cada tensor es `(seq_len, input_dim)`)

Snippet (padding/truncado y empaquetado):

```python
def keypoints_to_tensor(keypoints, expected_size=None, pad_value=0.0):
		kp_array = np.array(keypoints, dtype=np.float32)
		# keep only first 3 columns (x,y,z)
		kp_array = kp_array[:, :3] if kp_array.shape[1] >= 3 else pad_columns(kp_array)
		kp_flat = kp_array.flatten()
		if expected_size:
				if len(kp_flat) < expected_size:
						kp_flat = np.concatenate([kp_flat, np.full(expected_size - len(kp_flat), pad_value)])
				else:
						kp_flat = kp_flat[:expected_size]
		return torch.from_numpy(kp_flat)
```

Relevancia:
- Este módulo garantiza que cada frame produce vectores de tamaño fijo: `hand=126`, `body=99`, `face=1404`. El encoder depende de estas formas fijas.
- Si los keypoints vienen en formato plano (flattened) o con un canal de confidence extra, la función los normaliza/trunca.

---

**6) Inference — carga de checkpoint (`InferenceLoader`) y predictor (`Predictor`)**

- `InferenceLoader` (archivo `inference/loader.py`) se encarga de:
	- cargar checkpoint (torch.load),
	- inferir `num_classes`,
	- reconstruir arquitectura (`MultimodalEncoder` + `SignLanguageClassifier`) y cargar `state_dict`,
	- exponer `load_all()` → `(model, class_names, other_class_id)`.

Snippet (reconstrucción y carga):

```python
from services.encoder.model import MultimodalEncoder
from training.classifier import SignLanguageClassifier
encoder = MultimodalEncoder()
model = SignLanguageClassifier(encoder=encoder, num_classes=num_classes)
model.load_state_dict(model_state)
model = model.to(self.device).eval()
```

- `Predictor` (archivo `inference/predictor.py`) encapsula:
	- movimiento de tensores a device,
	- forward pass (`logits = model(hand, body, face, lengths)`),
	- softmax → top-k, y empaqueta `PredictionResult`.

Snippet (postprocesado top-k):

```python
logits = self.model(hand, body, face, lengths)
scores = F.softmax(logits.flatten(), dim=0)
topk_scores, topk_indices = torch.topk(scores, k)
for rank in range(k):
		topk_predictions.append(TopKPrediction(rank=rank+1, class_id=topk_indices[rank].item(), score=topk_scores[rank].item()))
```

Consideraciones:
- `Predictor.predict` asume batch size 1 por simplicidad; para batch se usa `predict_batch`.
- `InferenceLoader.get_num_classes()` intenta inferir `num_classes` desde el `state_dict` si no está explícito en el checkpoint.

---

**7) API — `FastAPI` REST + WebSocket**

- Archivo: `comsigns/services/api/main.py`
- Endpoints importantes:
	- `POST /infer/video` — recibe un archivo `UploadFile`, ejecuta `process_video_clip`, convierte a tensores con `feature_clip_to_tensors`, pasa por el encoder y devuelve embeddings en JSON.
	- `WebSocket /ws/infer` — recibe frames base64, decodifica, procesa (usa `process_video_clip` como simplificación), encoda con `encoder`, pasa por `glosador` y `translator`, y envía respuestas en tiempo real.

Flujo simplificado de `/infer/video`:

```text
client -> UploadFile -> save temp -> validate_video() -> process_video_clip() -> feature_clip_to_tensors() -> encoder() -> return embeddings
```

Snippet (parte de infer_video):

```python
tensors = feature_clip_to_tensors(feature_clip)
hand_t = tensors['hand'].unsqueeze(0)
body_t = tensors['body'].unsqueeze(0)
face_t = tensors['face'].unsqueeze(0)
embeddings = encoder(hand_t, body_t, face_t)
```

Puntos críticos de producción:
- Carga perezosa (lazy loading) de modelos (`get_encoder()`): evita costos en arranque, pero hay que manejar concurrencia si múltiples workers intentan cargar a la vez.
- WebSocket: el ejemplo guarda frames en archivos temporales y reusa `process_video_clip` — eso es simple pero ineficiente. Para producción conviene una ruta de procesamiento de frames en memoria que evite I/O.

---

**8) Integración y flujo de datos**

- Entrada esperada: video (archivo) o frames base64 (WebSocket).
- Preprocesado: `KeypointExtractor` → `FeatureClip` (Pydantic) → `feature_clip_to_tensors` → encoder tensors.
- Modelo: `MultimodalEncoder` produce secuencia de embeddings `(batch, seq_len, output_dim)` → (opcional) `SignLanguageClassifier` produce logits.
- Post-procesado: `Predictor` para clasificación (top-k) o `Glosador` + `Translator` para pipelines de glosa → texto.

---

**9) Configuraciones relevantes (keys observadas en código)**

- `preprocessing.mediapipe`: `min_detection_confidence`, `min_tracking_confidence`, `static_image_mode`
- `preprocessing.output_dir`: directorio de salida por defecto para features
- `encoder.hidden_dim`, `encoder.output_dim`, `encoder.num_layers`, `encoder.dropout`
- `api.host`, `api.port` (usado por `uvicorn.run`)

---

**10) Errores y modos de fallo comunes (y mitigaciones)**

- MediaPipe no instalado / modelos no disponibles → `KeypointExtractor.__init__` lanza `ImportError` o `RuntimeError` al no descargar modelos.
	- Mitigación: añadir modelos `*.task` en `models/mediapipe` o setear `model_paths` al crear `KeypointExtractor`.
- `FeatureClip` vacío (0 frames) → endpoints devuelven HTTP 500 o error; validar antes de proseguir.
- Desajuste de dimensiones (esperado 126/99/1404) → `keypoints_to_tensor` pad/trunca, pero truncar puede perder información; mejor asegurar extracción completa.
- Carga concurrente de modelos en entornos con múltiples workers → usar mecanismo de bloqueo o cargar modelo en proceso maestro y servir vía worker pool.

---

**11) Ejemplos de uso rápidos**

- Extraer y guardar features desde un video (CLI interno):

```python
from comsigns.services.preprocessing.process_clip import process_video_clip
fc = process_video_clip('video.mp4', format='json')
print(fc.clip_id, len(fc.frames))
```

- Inferencia local con checkpoint:

```python
from comsigns.services.inference.loader import load_checkpoint_for_inference
model, class_names, other_id, info = load_checkpoint_for_inference(Path('experiments/run_001/checkpoints/best.pt'))
```

---

Si quieres, puedo:
- añadir diagramas (mermaid) del flujo de datos,
- extraer y añadir más snippets de `glosador`, `translator` o `ingestion` (si los quieres documentados),
- o generar una versión en inglés.

Actualicé la lista de tareas: marcaré la extracción de snippets como completada y continuaré con la mejora/validación si me indicas qué módulos profundizar. 
