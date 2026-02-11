# 🏗️ ComSigns — Arquitectura y Sistema

> Visión general del pipeline, dataset AEC, resultados de entrenamiento y estructura del proyecto.

---

## 📖 Navegación

| Documento | Descripción |
|-----------|-------------|
| [🧠 Arquitectura del Modelo](../../docs/MODEL_ARCHITECTURE.md) | Diagramas detallados del encoder, fusión, clasificador |
| [📘 Documentación Técnica](../../docs/MODEL_TECHNICAL.md) | I/O, inferencia por módulo, limitaciones |
| [🏋️ Entrenamiento](../../docs/TRAINING.md) | Trainer, métricas, checkpointing |
| [👤 Guía de Usuario](../../docs/USER_GUIDE.md) | Instalación y ejecución |
| [📜 Referencia de Scripts](SCRIPTS_USAGE.md) | CLI flags y uso |
| [🌐 Inferencia Web](WEB_INFERENCE.md) | API REST + Frontend |

---

## 1. Visión General

ComSigns es un sistema de reconocimiento de lengua de señas que utiliza un **encoder multimodal** para procesar keypoints de manos, cuerpo y rostro, produciendo embeddings que se clasifican en glosas.

```
┌─────────────┐     ┌─────────────────────────────────────────────────────┐     ┌──────────────┐
│   VIDEO     │ ──▶ │              PIPELINE DE INFERENCIA                 │ ──▶ │    GLOSA     │
│  (frames)   │     │  MediaPipe → Encoder Multimodal → Clasificador      │     │  (palabra)   │
└─────────────┘     └─────────────────────────────────────────────────────┘     └──────────────┘
```

> [!TIP]
> Para diagramas mermaid detallados de cada componente, ver [MODEL_ARCHITECTURE.md](../../docs/MODEL_ARCHITECTURE.md).

---

## 2. Extracción de Keypoints (MediaPipe)

| Componente | Modelo | Keypoints | Dimensión |
|------------|--------|-----------|-----------|
| **Manos** | `hand_landmarker.task` | 21 × 2 manos | 126 (21 × 3 × 2) |
| **Cuerpo** | `pose_landmarker_lite.task` | 33 puntos | 99 (33 × 3) |
| **Rostro** | `face_landmarker.task` | 468 puntos | 1404 (468 × 3) |

Cada keypoint tiene 3 valores: `[x, y, z]`

**Setup:** [MODELS_SETUP.md](../MODELS_SETUP.md) · **Código:** [preprocessing/](../services/preprocessing/)

---

## 3. Encoder Multimodal

El encoder procesa secuencias temporales de keypoints y produce embeddings de 512 dimensiones.

```
     hand [B,T,126] → HandBranch (Linear → LSTM ×2 → LayerNorm) → [B,T,256] ─┐
     body [B,T,99]  → BodyBranch (Linear → LSTM ×2 → LayerNorm) → [B,T,256] ─┼─ CONCAT [B,T,768] → FUSION → [B,T,512]
     face [B,T,1404] → FaceBranch (MLP → LSTM ×2 → LayerNorm)  → [B,T,256] ─┘
```

| Parámetro | Valor |
|-----------|-------|
| `hidden_dim` | 256 |
| `output_dim` | 512 |
| `num_layers` | 2 (LSTM) |
| `dropout` | 0.1 |
| **Total params** | ~4.7M |

**Arquitectura detallada:** [MODEL_ARCHITECTURE.md](../../docs/MODEL_ARCHITECTURE.md) · **Código:** [encoder/model.py](../services/encoder/model.py)

---

## 4. Clasificador

```
Encoder Output [B,T,512] → Temporal Pooling [B,512] → Dropout(0.1) → Linear(512 → C) → [B, num_classes]
```

| Estrategia de pooling | Descripción |
|----------------------|-------------|
| `mean` | Promedio sobre timesteps (default) |
| `max` | Máximo sobre timesteps |
| `last` | Último timestep válido |

**Código:** [training/classifier.py](../training/classifier.py)

---

## 5. Dataset AEC

El dataset **AEC** (Asociación de Estudio del Conocimiento) de Lengua de Señas Peruana.

### Estructura

```
data/raw/lsp_aec/
├── dict.json           # Vocabulario: {gloss_id: {gloss, instances}}
├── Keypoints/pkl/      # Archivos .pkl con keypoints pre-extraídos
├── Videos/             # Videos originales
└── SRT/                # Subtítulos con timestamps
```

### Estadísticas

| Métrica | Valor |
|---------|-------|
| Total glosses | 506 |
| Total instances | ~2,278 |
| Splits | train: 1,757 / val: 521 |
| Source videos | 2 |

### Uso

```python
dataset = AECDataset(
    dataset_root=Path("data/raw/lsp_aec"),
    split_file=Path("data/splits/aec_stratified.json"),
    split="train"
)

sample = dataset[0]
# sample.hand_keypoints: [T, 126]
# sample.body_keypoints: [T, 99]
# sample.face_keypoints: [T, 1404]
# sample.gloss: "comer"
```

---

## 6. Pipeline de Entrenamiento

```
dict.json ──▶ AECDataset ──▶ MicroVocabDataset ──▶ DataLoader ──▶ Model ──▶ Loss ──▶ Optimizer
                   │              (filtrado)          (balanced)
                   ▼
              .pkl files
            (keypoints)
```

### Configuración Phase 1 (Micro Vocab)

| Parámetro | Valor |
|-----------|-------|
| Vocabulario | 6 palabras |
| Epochs | 100 (early stopping) |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Early stopping | patience=10 |

**Documentación completa:** [TRAINING.md](../../docs/TRAINING.md)

---

## 7. Resultados Phase 1

| Palabra | Precision | Recall | F1 | Estado |
|---------|-----------|--------|-----|--------|
| **comer** | 0.706 | 0.800 | **0.750** | ✅ Aprendido |
| **yo** | 0.714 | 0.625 | **0.667** | ✅ Aprendido |
| **tú** | 0.667 | 0.154 | 0.250 | ❌ |
| **sí** | 1.000 | 0.333 | **0.500** | ✅ Aprendido |
| **no** | 0.238 | 0.833 | 0.370 | ❌ |
| **dos** | 0.750 | 0.750 | **0.750** | ✅ Aprendido |

**Macro F1:** 0.5478 · **Palabras aprendidas (F1 ≥ 0.5):** 4 de 6

### Artefactos

```
experiments/micro_v1/
├── best.pt              # Checkpoint del mejor modelo
├── class_mapping.json   # Mapeo de clases
└── training_summary.json
```

---

## 8. Inferencia

### API

```bash
python run_api.py
# POST /predict con video → {"gloss": "comer", "confidence": 0.87}
```

### Script Directo

```bash
python -m scripts.infer --model experiments/micro_v1/best.pt --video input.mp4
```

**Referencia de scripts:** [SCRIPTS_USAGE.md](SCRIPTS_USAGE.md)

---

## 9. Estructura del Proyecto

```
comsigns/
├── core/data/              # Datasets y loaders
│   └── datasets/aec/       # Dataset AEC
├── services/
│   ├── encoder/            # MultimodalEncoder
│   ├── preprocessing/      # KeypointExtractor (MediaPipe)
│   ├── inference/          # Predictor y loader
│   └── api/                # FastAPI
├── training/               # Trainer, Classifier, Metrics
├── scripts/                # Scripts CLI
├── experiments/            # Modelos entrenados
└── docs/                   # Documentación
```

---

## 10. Dependencias Clave

| Paquete | Versión | Uso |
|---------|---------|-----|
| PyTorch | ≥2.0 | Framework de deep learning |
| MediaPipe | ≥0.10 | Extracción de keypoints |
| NumPy | ≥1.24 | Operaciones numéricas |
| FastAPI | ≥0.100 | API REST |

---

## 11. Próximos Pasos

1. **Expandir vocabulario** — Añadir más palabras al micro-vocab
2. **Analizar confusiones** — Entender por qué "tú" y "no" tienen bajo recall
3. **Data augmentation** — ver [TRAINING.md § augmentation](../../docs/TRAINING.md#augmentationpy--data-augmentation)
4. **Integrar con glosador** — Conectar encoder con módulo de traducción

---

## 📚 Documentos Relacionados

- [🧠 Arquitectura del Modelo](../../docs/MODEL_ARCHITECTURE.md) — Diagramas detallados
- [📘 Documentación Técnica](../../docs/MODEL_TECHNICAL.md) — I/O, limitaciones
- [🏋️ Entrenamiento](../../docs/TRAINING.md) — Trainer, métricas
- [📜 Scripts](SCRIPTS_USAGE.md) — CLI
- [🌐 Inferencia Web](WEB_INFERENCE.md) — API + Frontend
