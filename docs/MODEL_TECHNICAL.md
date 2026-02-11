# 📘 ComSigns — Documentación Técnica del Modelo

> Descripción detallada del modelo de IA: módulos internos, formato de entradas/salidas, inferencia paso a paso y limitaciones.

---

## 📖 Navegación

| Documento | Descripción |
|-----------|-------------|
| [🧠 Arquitectura del Modelo](MODEL_ARCHITECTURE.md) | Diagramas de encoder, ramas, fusión y clasificador |
| [🏋️ Entrenamiento](TRAINING.md) | Trainer, métricas, checkpointing, augmentation |
| [👤 Guía de Usuario](USER_GUIDE.md) | Instalación, ejecución, troubleshooting |
| [🏗️ Arquitectura General](../comsigns/docs/ARCHITECTURE.md) | Pipeline + dataset + resultados Phase 1 |
| [⚙️ Servicios](../services/SERVICES_TECH_DOC.md) | Docs técnicos de preprocessing, encoder, API |

---

## 1. Visión General del Modelo

- **Dominio:** Interpretación de Lengua de Señas (visión por computador + series temporales)
- **Framework:** PyTorch
- **Qué resuelve:** Reconocimiento y clasificación de signos a partir de video. Convierte secuencias de frames en keypoints y produce predicciones de clases con scores de confianza.
- **Qué NO resuelve:** Traducción en lenguaje natural, servicios web/UI, ambigüedades semánticas fuera del vocabulario entrenado.

### Flujo General de Datos

```
Entrada: Video o .pkl con keypoints
    ↓
Preprocesado: MediaPipe → keypoints aplanados/normalizados
    ↓
Encoder: Ramas hand/body/face → embeddings temporales
    ↓
Fusion + Clasificador: embeddings → logits/probabilidades
    ↓
Salida: Top-K predicciones (class_id, class_name, confidence)
```

> [!TIP]
> Para diagramas detallados de la arquitectura, consulta [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

---

## 2. Estructura del Repositorio (Modelo)

| Carpeta / Archivo | Descripción |
|-------------------|-------------|
| [`comsigns/`](../comsigns/) | Código fuente principal del modelo y utilidades |
| [`comsigns/services/encoder/`](../comsigns/services/encoder/) | Implementación del encoder multimodal |
| [`comsigns/services/preprocessing/`](../comsigns/services/preprocessing/) | Extracción de keypoints y transformaciones |
| [`comsigns/training/`](../comsigns/training/) | Clasificador, rutinas de entrenamiento y métricas |
| [`comsigns/scripts/`](../comsigns/scripts/) | Scripts de inferencia y entrenamiento |
| [`comsigns/experiments/`](../comsigns/experiments/) | Checkpoints, mappings y resultados |
| [`data/`](../data/) | Datasets y videos de ejemplo |

---

## 3. Documentación por Módulo

> [!NOTE]
> A continuación se documentan los módulos clave que forman la "API interna de inferencia". Para cada módulo se indica propósito, entradas, salidas y errores comunes.

### 3.1 `preprocessing` — Extracción de Keypoints

**Ubicación:** [`comsigns/services/preprocessing/`](../comsigns/services/preprocessing/)

- **Propósito:** Extraer keypoints de video, normalizar y serializar features.
- **Componentes:** `KeypointExtractor` (encapsula MediaPipe), utilidades de normalización.
- **Entradas:** Ruta a video (`.mp4`, etc.) o frames; parámetros de extracción.
- **Salidas:** `FeatureClip` con `frames`, arrays por frame con keypoints para `hand`, `body`, `face`. También `.pkl` o numpy.
- **Dependencias:** MediaPipe, OpenCV, numpy.

> [!WARNING]
> Errores comunes: No encontrar landmarks (frames vacíos), formatos de keypoints inesperados (4 elementos vs 3). Verificar extractor y usar `infer_video.py` actualizado que espera 3 valores por keypoint.

**Setup:** Ver [MODELS_SETUP.md](../comsigns/MODELS_SETUP.md)

### 3.2 `encoder` — Encoder Multimodal

**Ubicación:** [`comsigns/services/encoder/`](../comsigns/services/encoder/) · **Archivo clave:** [`model.py`](../comsigns/services/encoder/model.py)

- **Propósito:** Convertir secuencias de keypoints en embeddings temporales fusionados.
- **Estructura:** Tres ramas (`HandBranch`, `BodyBranch`, `FaceBranch`) → concat → fusión.
- **Entradas (shapes):**
  - `hand`: `(batch, seq_len, 126)` — 2 manos × 21 keypoints × 3
  - `body`: `(batch, seq_len, 99)` — 33 × 3
  - `face`: `(batch, seq_len, 1404)` — 468 × 3
- **Salidas:** `(batch, seq_len, 512)` embeddings temporales fusionados.

> [!IMPORTANT]
> Los keypoints deben estar normalizados de manera consistente (mismo orden y unidades). Las longitudes de secuencia deben ser > 0.

**Arquitectura detallada:** Ver [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)

### 3.3 `temporal` — Módulos Temporales

**Ubicación estimada:** `comsigns/services/temporal/`

- **Propósito:** Agregar lógica temporal (pooling, attention, decodificadores) que postprocesa embeddings.
- **Entradas:** Embeddings `(batch, seq_len, dim)` producidos por el encoder.
- **Salidas:** Representaciones agregadas para la cabeza de clasificación.

> [!WARNING]
> Errores comunes: Desincronización entre `seq_len` usados en entrenamiento y en inferencia; usar mismos padding/truncamiento.

### 3.4 `inference` — Scripts de Inferencia

**Patrón:** [`comsigns/scripts/infer_video.py`](../comsigns/scripts/infer_video.py)

- **Propósito:** Orquestar extracción de features, carga de checkpoint y predicción.
- **Entradas:** Ruta a video o features serializadas; checkpoint del modelo; mapping de clases.
- **Salidas:** Top-K predicciones con `class_id`, `class_name` y `confidence`.
- **Dependencias:** `MultimodalEncoder`, `SignLanguageClassifier`, `KeypointExtractor`.

**Referencia de flags:** Ver [SCRIPTS_USAGE.md](../comsigns/docs/SCRIPTS_USAGE.md)

### 3.5 `training` — Entrenamiento y Clasificación

**Ubicación:** [`comsigns/training/`](../comsigns/training/) · **Archivo clave:** [`classifier.py`](../comsigns/training/classifier.py)

- **Propósito:** Cabeza de clasificación y procedimientos de entrenamiento/evaluación.
- **Componente clave:** `SignLanguageClassifier` — encoder + classification head (linear → softmax).
- **Entradas:** Embeddings o keypoints; targets (ids de clase).
- **Salidas:** Logits, pérdidas y métricas (accuracy, top-k, etc.).

**Documentación completa:** Ver [TRAINING.md](TRAINING.md)

### 3.6 `utils` / helpers

- **Propósito:** Funciones utilitarias para serialización, métricas, visualización de keypoints.

> [!WARNING]
> Errores comunes: Rutas relativas mal resueltas. Usar `PROJECT_ROOT` y verificar existencia de archivos.

---

## 4. Inferencia del Modelo (Uso Correcto)

### Preparación de Datos

- **Opción A:** Proporcionar un video legible por OpenCV; el extractor (MediaPipe) generará keypoints.
- **Opción B:** Proporcionar features serializadas (`.pkl`) con keys `hand`, `body`, `face` y shapes `(seq_len, dim)`.

### Formato Aceptado

| Modalidad | Tipo | Shape |
|-----------|------|-------|
| `hand` | `float32` | `(seq_len, 126)` |
| `body` | `float32` | `(seq_len, 99)` |
| `face` | `float32` | `(seq_len, 1404)` |

### Flujo Recomendado

1. Extraer keypoints (o verificar `features.pkl`)
2. Normalizar/filtrar frames con detección fallida
3. Convertir arrays a tensores PyTorch + dimensión batch (`unsqueeze(0)`)
4. Cargar checkpoint del modelo y mapping de clases
5. Ejecutar `model.eval()` + `torch.no_grad()` → `logits = model(hand, body, face)`
6. Aplicar `softmax` para probabilidades y extraer top-K

### Ejemplo Conceptual

```
my_video.mp4 → extractor → hand.npy, body.npy, face.npy
    → cargar checkpoint best.pt → inferencia
    → [{class_id, class_name, confidence}, ...]
```

> [!TIP]
> - Usar `--device cpu` si no hay GPU compatible.
> - Verificar que `class_mapping.json` tenga `config.vocabulary_size` consistente con el checkpoint.

---

## 5. Modelos Entrenados y Artefactos

### Checkpoints (`.pt`)

Contienen `state_dict` con parámetros del encoder y clasificador. Algunos incluyen metadatos adicionales (optimizer, epoch, training config).

### `class_mapping.json`

Debe incluir al menos:
- `config.vocabulary_size` — número de clases
- `new_class_names` — mapa de índices a nombres legibles

### Cómo Seleccionar un Checkpoint

1. Priorizar `best.pt` (mejor según métrica registrada)
2. Revisar `training_summary.json` en el mismo experimento
3. Elegir checkpoint entrenado con la misma arquitectura y config que el código actual

### Organización de Experimentos

```
comsigns/experiments/<run>/
├── best.pt                 # Mejor checkpoint
├── class_mapping.json      # Mapeo de clases
└── training_summary.json   # Resumen de entrenamiento
```

---

## 6. Limitaciones y Supuestos

### ⚠️ Entradas que Degradan Rendimiento

- Videos con manos parcialmente fuera de cuadro o baja resolución
- Iluminación extrema donde MediaPipe no detecta rostro/manos
- Personas múltiples superpuestas (landmarks ambiguos)

### 🚫 Fuera de Alcance

- Traducción libre de lengua de señas a oraciones gramaticales
- Signos fuera del vocabulario entrenado o dialectos regionales distintos

### ⚡ Dependencias Críticas

- **MediaPipe** para extracción de keypoints
- **PyTorch** para cómputo
- Versiones específicas de TensorFlow/MediaPipe pueden afectar la extracción

### 🔴 Riesgos Operativos

- Desajustes de dimensiones (común al mezclar versiones del extractor)
- Incompatibilidades de checkpoint (arquitectura vs state_dict)

---

## 7. Glosario

| Término | Definición |
|---------|------------|
| **Keypoint** | Punto de interés detectado en imagen (x, y, z) por MediaPipe |
| **FeatureClip** | Estructura por frames que contiene keypoints para hand, body, face |
| **Embedding** | Representación vectorial producida por el encoder |
| **Checkpoint** | Archivo con pesos del modelo entrenado (`.pt`) |
| **Top-K** | Las K clases con mayor probabilidad |
| **Glosa** | Palabra en lengua de señas (representación escrita de un signo) |

---

## 8. Buenas Prácticas

- ✅ Validar shapes antes de pasar al modelo
- ✅ Mantener `class_mapping.json` sincronizado con checkpoints
- ✅ Utilizar `model.eval()` y `torch.no_grad()` para inferencia
- ✅ Preprocesar para eliminar frames con detección nula o interpolarlos

---

## 📍 Localizaciones Útiles

| Componente | Archivo |
|------------|---------|
| Script de inferencia | [scripts/infer_video.py](../comsigns/scripts/infer_video.py) |
| Encoder | [services/encoder/model.py](../comsigns/services/encoder/model.py) |
| Classifier | [training/classifier.py](../comsigns/training/classifier.py) |
| Experimentos | [experiments/](../comsigns/experiments/) |
| Extractor | [services/preprocessing/](../comsigns/services/preprocessing/) |

---

## 📚 Documentos Relacionados

- [🧠 Arquitectura del Modelo](MODEL_ARCHITECTURE.md) — Diagramas detallados
- [🏋️ Entrenamiento](TRAINING.md) — Trainer, métricas, checkpointing
- [👤 Guía de Usuario](USER_GUIDE.md) — Instalación y ejecución
- [📜 Referencia de Scripts](../comsigns/docs/SCRIPTS_USAGE.md) — CLI flags
- [🌐 Inferencia Web](../comsigns/docs/WEB_INFERENCE.md) — API REST + Frontend
