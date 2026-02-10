**ComSigns — Documentación Técnica del Modelo de IA**

**Visión general**
- **Propósito:** Este documento describe con detalle el modelo de inteligencia artificial contenido en este repositorio, su arquitectura interna, módulos relevantes, formato de entradas/salidas y la forma correcta de ejecutar inferencia. Está dirigido a desarrolladores, investigadores e integradores técnicos.
- **Dominio:** Interpretación de Lengua de Señas (visión por computador + series temporales).
- **Framework principal:** PyTorch.

**1. Visión general del modelo**
- **Qué problema resuelve:** Reconocimiento y clasificación de signos a partir de video. Convierte secuencias de frames en secuencias de keypoints (características) y produce predicciones de clases (vocabulario de signos) con scores de confianza.
- **Qué NO intenta resolver:** No realiza traducción en lenguaje natural, no provee servicios web ni UI; no resuelve ambigüedades semánticas del lenguaje humano fuera del vocabulario entrenado.
- **Flujo general de datos:**
  - Entrada: Video o representación intermedia de keypoints (por ejemplo archivos serializados .pkl).
  - Preprocesado: Extracción de keypoints (MediaPipe) y aplanado/normalización por frame.
  - Encoder: Ramas separadas para `hand`, `body` y `face` que generan embeddings temporales.
  - Fusion + Clasificador: Embeddings fusionados y pasada a una cabeza de clasificación que devuelve logits/probabilidades.
  - Salida: Predicciones top-K, secuencias temporales de embeddings y métricas (confianza, scores).

**2. Estructura del repositorio (modelo)**
- **`comsigns/`**: Código fuente principal del modelo y utilidades. Contiene submódulos críticos como `services`, `training`, `scripts`.
  - See: [comsigns/](comsigns/)
- **`comsigns/services/`**: Componentes reutilizables para preprocesado, encoder y utilidades.
  - `encoder/`: Implementación del encoder multimodal.
  - `preprocessing/`: Extracción de keypoints y transformaciones de features.
- **`comsigns/training/`**: Definición de la cabeza de clasificación, rutinas de entrenamiento y métricas.
  - `classifier.py`: Clase `SignLanguageClassifier` que encapsula encoder + cabeza de clasificación.
- **`comsigns/scripts/`**: Scripts de uso directo (inferencia por video, entrenamiento rápido, extracción de features). Ej.: `infer_video.py` (script de inferencia localizado en [comsigns/scripts/infer_video.py](comsigns/scripts/infer_video.py)).
- **`comsigns/experiments/`**: Resultados de experimentos, checkpoints y mappings (ej.: `micro_v1_retrained/best.pt`, `class_mapping.json`).
- **`data/`**: Datasets y ejemplos de video para pruebas.

**3. Documentación por módulo (módulos críticos)**
Nota: A continuación se documentan los módulos clave que forman la "API interna de inferencia". Para cada módulo se indica propósito, entradas, salidas, dependencias y errores comunes.

**3.1 `preprocessing`** (ubicación: `comsigns/services/preprocessing`)
- **Propósito:** Extraer keypoints de video, normalizar y serializar features.
- **Componentes principales:** `KeypointExtractor` (encapsula MediaPipe), utilidades de normalización y guardado.
- **Entradas:** Ruta a video (`.mp4`, etc.) o frames; parámetros de extracción (resolución, fps opcional).
- **Salidas:** `FeatureClip` (objeto interno con `frames`), arrays por frame con keypoints para `hand`, `body`, `face`. También puede exportar `.pkl` o estructuras numpy.
- **Dependencias internas:** MediaPipe, OpenCV, numpy.
- **Supuestos:** Video con una o pocas personas visibles; manos y rostro detectables por MediaPipe; fps suficiente para capturar gestos.
- **Errores comunes:** No encontrar landmarks (frames vacíos), formatos de keypoints inesperados (4 elementos vs 3). Solución: verificar extractor y usar la versión del script de inferencia actualizada que espera 3 valores por keypoint.

**3.2 `encoder`** (ubicación: `comsigns/services/encoder`) — núcleo del modelo
- **Propósito:** Convertir secuencias de keypoints en embeddings temporales por rama y fusionarlos.
- **Archivos claves:** [comsigns/services/encoder/model.py](comsigns/services/encoder/model.py)
- **Estructura:** Tres ramas:
  - `HandBranch`: proyección lineal → LSTM → LayerNorm. Entrada esperada: `(batch, seq_len, 126)` (2 manos × 21 × 3).
  - `BodyBranch`: proyección lineal → LSTM → LayerNorm. Entrada: `(batch, seq_len, 99)` (33 × 3).
  - `FaceBranch`: proyección (reducción de dimensionalidad) → LSTM → LayerNorm. Entrada: `(batch, seq_len, 1404)` (468 × 3).
  - `MultimodalEncoder`: concatena las salidas de las 3 ramas y aplica una fusión (Linear → ReLU → Dropout → LayerNorm) produciendo `output_dim` por paso temporal.
- **Entradas:** Tensores PyTorch: `hand_keypoints`, `body_keypoints`, `face_keypoints` con shapes indicadas anteriormente.
- **Salidas:** Tensor `(batch, seq_len, output_dim)` con embeddings temporales fusionados.
- **Dependencias internas:** PyTorch, configuración del proyecto (`comsigns/config.yaml` si aplica).
- **Supuestos:** Los keypoints han sido normalizados de manera consistente (mismo orden y unidades). Las longitudes de secuencia no deben romper las expectativas del modelo (seq_len > 0).
- **Errores comunes:** Mismatch de dimensiones (e.g., vectores con 4 valores por keypoint). Ver `infer_video.py` para ejemplo de preprocesado compatible.

**3.3 `temporal`** (ubicación estimada: `comsigns/services/temporal`)
- **Propósito:** Módulos para agregar lógica temporal (pooling, attention temporal, decodificadores de secuencia) que pueden postprocesar embeddings.
- **Entradas:** Embeddings `(batch, seq_len, dim)` producidos por `encoder`.
- **Salidas:** Representaciones agregadas (por ejemplo, vector global, o secuencia transformada) para la cabeza de clasificación o para tareas secuenciales.
- **Dependencias internas:** PyTorch.
- **Errores comunes:** Desincronización between seq_len usados en entrenamiento y en inferencia; asegurarse de usar mismos padding/truncamiento.

**3.4 `inference`** (patrón: `comsigns/scripts/infer_video.py`)
- **Propósito:** Orquestar la extracción de features, carga de checkpoint y ejecución de modelo para obtener predicciones.
- **Entradas:** Ruta a video o path a features serializadas; checkpoint del modelo; mapping de clases.
- **Salidas:** Top-K predicciones con `class_id`, `class_name` y `confidence`.
- **Dependencias internas:** `services.encoder.model.MultimodalEncoder`, `training.classifier.SignLanguageClassifier`, `services.preprocessing.KeypointExtractor`.
- **Supuestos:** Checkpoint compatible con la arquitectura esperada; mapping de clases contiene `vocabulary_size` y `new_class_names`.
- **Errores comunes:** Checkpoint con formato distinto (state dict anidado), shapes incompatibles al cargar estado (ver `torch.load` y manejo en `infer_video.py`).

**3.5 `training`** (ubicación: `comsigns/training`)
- **Propósito:** Definir la cabeza de clasificación y procedimientos para entrenar y evaluar el modelo.
- **Componente clave:** `SignLanguageClassifier` — encapsula `encoder` + `classification head` (linear(s) → softmax).
- **Entradas:** Embeddings o keypoints (según flujo); targets (ids de clase).
- **Salidas:** Logits, pérdidas y métricas (accuracy, top-k, etc.).
- **Errores comunes:** Discrepancias entre `num_classes` en `class_mapping.json` y el `num_classes` usado al instanciar la cabeza.

**3.6 `utils` / helpers**
- **Propósito:** Funciones utilitarias para serialización, métricas, visualización de keypoints y conversión entre formatos.
- **Errores comunes:** Rutas relativas mal resueltas; usar `PROJECT_ROOT` y verificaciones de existencia de archivos antes de operar.

**4. Inferencia del modelo (uso correcto)**
- **Preparación de datos de entrada:**
  - Opción A: Proporcionar un `video` legible por OpenCV; el extractor interno (MediaPipe) generará keypoints por frame.
  - Opción B: Proporcionar `features serializadas` (p. ej. `.pkl` con arrays numpy) con las keys `hand`, `body`, `face` y shapes `(seq_len, dim)` donde `dim` corresponde a 126, 99 y 1404 respectivamente.
- **Formato aceptado:**
  - `hand`: numpy array float32, shape `(seq_len, 126)`
  - `body`: numpy array float32, shape `(seq_len, 99)`
  - `face`: numpy array float32, shape `(seq_len, 1404)`
  - Alternativamente, un archivo de video: el script `infer_video.py` llama al extractor y produce internamente esos arrays.
- **Flujo recomendado para inferencia:**
  1. Extraer keypoints (o verificar `features.pkl`).
  2. Normalizar/filtrar frames con detección fallida (opcionalmente interpolar si hay pocos frames perdidos).
  3. Convertir arrays a tensores PyTorch y añadir dimensión `batch` (ej. `unsqueeze(0)`).
  4. Cargar checkpoint del modelo y mapping de clases.
  5. Ejecutar `model.eval()` y `with torch.no_grad(): logits = model(hand, body, face)`.
  6. Aplicar `softmax` para obtener probabilidades y extraer top-K.
- **Ejemplo conceptual (sin backend):**
  - Input: `my_video.mp4` → extractor → `hand.npy`, `body.npy`, `face.npy` → cargar checkpoint `best.pt` → inferencia → `[{class_id, class_name, confidence}, ...]`.
- **Consejos prácticos:**
  - Forzar `--device cpu` si no se dispone de GPU compatible.
  - Verificar que `class_mapping.json` tenga `config.vocabulary_size` consistente con checkpoint.

**5. Modelos entrenados y artefactos**
- **Checkpoints (`.pt`):** Contienen típicamente un `state_dict` (o una estructura con `model_state_dict`) con parámetros del encoder y del clasificador. Algunos checkpoints pueden contener metadatos adicionales (optimizer, epoch, training config).
- **`class_mapping.json`:** Debe incluir al menos: `config.vocabulary_size` y `new_class_names` (mapa de índices → nombres). Es usado en inferencia para mapear índices a nombres legibles.
- **Cómo seleccionar un checkpoint:**
  - Priorizar `best.pt` (mejor según métrica registrada), revisar `training_summary.json` en el mismo experimento para entender la métrica usada.
  - Consistencia: elegir checkpoint entrenado con la misma arquitectura y `encoder` config (hidden_dim, output_dim) que el código actual espera.
- **Registry (si existe):** En este repositorio los experimentos están organizados en `comsigns/experiments/<run>/` con `best.pt`, `class_mapping.json` y `training_summary.json`.

**6. Limitaciones y supuestos del modelo**
- **Entradas que fallan o degradan rendimiento:**
  - Videos con manos parcialmente fuera de cuadro, o muy baja resolución para detectar landmarks.
  - Iluminación extrema, o vistas donde MediaPipe no detecta rostro/manos.
  - Personas múltiples superpuestas; el extractor puede devolver landmarks ambiguos.
- **Casos fuera de alcance:**
  - Traducción libre de lenguaje de señas a oraciones gramaticalmente correctas en lenguaje oral.
  - Signos fuera del vocabulario entrenado o dialectos regionales muy distintos a los del dataset.
- **Dependencias críticas:** MediaPipe para extracción de keypoints; PyTorch para cómputo; versiones específicas de TensorFlow/MediaPipe pueden afectar la extracción.
- **Riesgos operativos:** Desajustes de dimensiones (común al mezclar versiones antiguas del extractor), incompatibilidades de checkpoint.

**7. Referencias para manual de usuario técnico**
- **Secciones reutilizables directamente:**
  - **Inferencia paso a paso** (sección 4) → copiar a manual de usuario.
  - **Preparación de datos** y **Formas de entrada** → útil para guías de integración.
  - **Limitaciones** y **Errores comunes** → FAQ / Troubleshooting.
- **Glosario (términos clave):**
  - **Keypoint:** Punto de interés detectado en imagen (x, y, z) por MediaPipe.
  - **FeatureClip:** Estructura por frames que contiene keypoints para `hand`, `body`, `face`.
  - **Embedding:** Representación vectorial producida por el encoder.
  - **Checkpoint:** Archivo con pesos del modelo entrenado (`.pt`).
  - **Top-K:** Las K clases con mayor probabilidad.

**8. Buenas prácticas de uso**
- Validar shapes antes de pasar al modelo.
- Mantener `class_mapping.json` sincronizado con checkpoints.
- Utilizar `model.eval()` y `torch.no_grad()` para inferencia.
- Preprocesar para eliminar frames con detección nula o interpolarlos con criterio.

**Apéndice — Localizaciones útiles**
- Script de inferencia: [comsigns/scripts/infer_video.py](comsigns/scripts/infer_video.py)
- Encoder: [comsigns/services/encoder/model.py](comsigns/services/encoder/model.py)
- Classifier: [comsigns/training/classifier.py](comsigns/training/classifier.py)
- Experimentos y checkpoints: [comsigns/experiments/](comsigns/experiments/)
- Extractor de keypoints: [comsigns/services/preprocessing/](comsigns/services/preprocessing/)

---

Documento generado como base técnica. Si quieres, puedo:
- expandir secciones con diagramas de flujo (ASCII/mermaid),
- añadir snippets de ejemplo de validación de shapes en Python,
- o convertir esto en `README` o páginas `mkdocs` bajo `docs/`.
