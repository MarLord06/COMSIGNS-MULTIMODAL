# ComSigns - Arquitectura y Modelo

## Visión General

ComSigns es un sistema de reconocimiento de lengua de señas que utiliza un **encoder multimodal** para procesar keypoints de manos, cuerpo y rostro, produciendo embeddings que se clasifican en glosas (palabras en lengua de señas).

```
┌─────────────┐     ┌─────────────────────────────────────────────────────┐     ┌──────────────┐
│   VIDEO     │ ──▶ │              PIPELINE DE INFERENCIA                 │ ──▶ │    GLOSA     │
│  (frames)   │     │  MediaPipe → Encoder Multimodal → Clasificador      │     │  (palabra)   │
└─────────────┘     └─────────────────────────────────────────────────────┘     └──────────────┘
```

---

## 1. Extracción de Keypoints (MediaPipe)

El primer paso convierte frames de video en representaciones numéricas usando **MediaPipe Tasks**.

### Modelos utilizados

| Componente | Modelo | Keypoints | Dimensión |
|------------|--------|-----------|-----------|
| **Manos** | `hand_landmarker.task` | 21 × 2 manos | 168 (21 × 4 × 2) |
| **Cuerpo** | `pose_landmarker_lite.task` | 33 puntos | 132 (33 × 4) |
| **Rostro** | `face_landmarker.task` | 468 puntos | 1872 (468 × 4) |

Cada keypoint tiene 4 valores: `[x, y, z, confidence/visibility]`

### Salida por frame
```python
{
    "hand": np.array([168]),    # Ambas manos concatenadas
    "body": np.array([132]),    # Pose del cuerpo
    "face": np.array([1872])    # Landmarks faciales
}
```

---

## 2. Encoder Multimodal

El encoder procesa secuencias temporales de keypoints y produce embeddings de 512 dimensiones.

### Arquitectura

```
                    ┌──────────────────┐
                    │   HAND BRANCH    │
     hand [B,T,168] │ Linear → LSTM ×2 │ → [B,T,256]
                    │   LayerNorm      │
                    └────────┬─────────┘
                             │
                    ┌──────────────────┐
                    │   BODY BRANCH    │
     body [B,T,132] │ Linear → LSTM ×2 │ → [B,T,256]    ──────▶  CONCAT  ──────▶  FUSION  ──────▶  [B,T,512]
                    │   LayerNorm      │                         [B,T,768]      Linear+ReLU
                    └────────┬─────────┘                                        +Dropout
                             │                                                  +LayerNorm
                    ┌──────────────────┐
                    │   FACE BRANCH    │
    face [B,T,1872] │ MLP → LSTM ×2    │ → [B,T,256]
                    │   LayerNorm      │
                    └──────────────────┘
```

### Componentes

#### HandBranch / BodyBranch
```python
class HandBranch(nn.Module):
    input_proj: Linear(input_dim → 256)
    lstm: LSTM(256, 256, num_layers=2, dropout=0.1)
    layer_norm: LayerNorm(256)
```

#### FaceBranch
```python
class FaceBranch(nn.Module):
    # MLP para reducir alta dimensionalidad (1872 → 256)
    input_proj: Sequential(
        Linear(1872 → 512),
        ReLU, Dropout,
        Linear(512 → 256)
    )
    lstm: LSTM(256, 256, num_layers=2, dropout=0.1)
    layer_norm: LayerNorm(256)
```

#### Fusion Layer
```python
fusion: Sequential(
    Linear(768 → 512),  # 256*3 ramas
    ReLU,
    Dropout(0.1),
    LayerNorm(512)
)
```

### Parámetros del Encoder
| Parámetro | Valor |
|-----------|-------|
| `hidden_dim` | 256 |
| `output_dim` | 512 |
| `num_layers` | 2 (LSTM) |
| `dropout` | 0.1 |
| **Total params** | ~4.7M |

---

## 3. Clasificador

El clasificador toma los embeddings del encoder y produce logits para cada clase.

```
┌────────────────────┐      ┌──────────────┐      ┌────────────────┐      ┌──────────────┐
│  Encoder Output    │ ──▶  │   Temporal   │ ──▶  │    Dropout     │ ──▶  │   Linear     │
│   [B, T, 512]      │      │    Pooling   │      │     (0.1)      │      │ (512 → C)    │
└────────────────────┘      │  [B, 512]    │      └────────────────┘      └──────────────┘
                            └──────────────┘                                    ↓
                                                                          [B, num_classes]
```

### Temporal Pooling

Convierte secuencias de longitud variable a vectores de tamaño fijo:

| Estrategia | Descripción |
|------------|-------------|
| `mean` | Promedio sobre timesteps (default) |
| `max` | Máximo sobre timesteps |
| `last` | Último timestep válido |

### SignLanguageClassifier
```python
class SignLanguageClassifier(nn.Module):
    def __init__(self, encoder, num_classes, pooling="mean", dropout=0.1):
        self.encoder = MultimodalEncoder()
        self.dropout = Dropout(0.1)
        self.classifier = Linear(512, num_classes)
    
    def forward(self, hand, body, face, lengths=None):
        embeddings = self.encoder(hand, body, face)  # [B, T, 512]
        pooled = temporal_pool(embeddings, lengths)   # [B, 512]
        logits = self.classifier(self.dropout(pooled)) # [B, C]
        return logits
```

---

## 4. Dataset AEC

El dataset utilizado es el **AEC** (Asociación de Estudio del Conocimiento) de Lengua de Señas Peruana.

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

### AECDataset
```python
dataset = AECDataset(
    dataset_root=Path("data/raw/lsp_aec"),
    split_file=Path("data/splits/aec_stratified.json"),
    split="train"
)

sample = dataset[0]
# sample.hand_keypoints: [T, 168]
# sample.body_keypoints: [T, 132]
# sample.face_keypoints: [T, 1872]
# sample.gloss: "comer"
# sample.gloss_id: 288
```

---

## 5. Pipeline de Entrenamiento

### Flujo de datos
```
dict.json ──▶ AECDataset ──▶ MicroVocabDataset ──▶ DataLoader ──▶ Model ──▶ Loss ──▶ Optimizer
                   │              (filtrado)          (balanced)
                   ▼
              .pkl files
            (keypoints)
```

### Configuración de entrenamiento (Phase 1 - Micro Vocab)
| Parámetro | Valor |
|-----------|-------|
| Vocabulario | 6 palabras |
| Epochs | 100 (early stopping) |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Early stopping | patience=10 |

### Micro-Vocabulario V1
| Palabra | Soporte (train/val) | ID Original |
|---------|---------------------|-------------|
| comer | 57 / 15 | 288 |
| yo | 31 / 8 | 0 |
| tú | 50 / 13 | 96 |
| sí | 36 / 9 | 58 |
| no | 22 / 6 | 39 |
| dos | 31 / 8 | 65 |

---

## 6. Resultados Phase 1

### Métricas del modelo entrenado

| Palabra | Precision | Recall | F1 | Estado |
|---------|-----------|--------|-----|--------|
| **comer** | 0.706 | 0.800 | **0.750** | ✓ Aprendido |
| **yo** | 0.714 | 0.625 | **0.667** | ✓ Aprendido |
| **tú** | 0.667 | 0.154 | 0.250 | ✗ |
| **sí** | 1.000 | 0.333 | **0.500** | ✓ Aprendido |
| **no** | 0.238 | 0.833 | 0.370 | ✗ |
| **dos** | 0.750 | 0.750 | **0.750** | ✓ Aprendido |

**Macro F1:** 0.5478  
**Palabras aprendidas (F1 ≥ 0.5):** 4 de 6

### Artefactos
```
experiments/micro_v1/
├── best.pt              # Checkpoint del mejor modelo
├── class_mapping.json   # Mapeo de clases (old_id ↔ new_id)
└── training_summary.json
```

---

## 7. Inferencia

### API
```bash
python run_api.py
# POST /predict con video → {"gloss": "comer", "confidence": 0.87}
```

### Script directo
```bash
python -m scripts.infer --model experiments/micro_v1/best.pt --video input.mp4
```

---

## 8. Estructura del Proyecto

```
comsigns/
├── core/data/              # Datasets y loaders
│   └── datasets/aec/       # Dataset AEC
├── services/
│   ├── encoder/            # MultimodalEncoder
│   ├── preprocessing/      # KeypointExtractor (MediaPipe)
│   └── api/                # FastAPI
├── training/               # Trainer, Classifier, Checkpoints
├── scripts/                # Scripts de entrenamiento e inferencia
├── experiments/            # Modelos entrenados
└── docs/                   # Documentación
```

---

## 9. Dependencias Clave

| Paquete | Versión | Uso |
|---------|---------|-----|
| PyTorch | ≥2.0 | Framework de deep learning |
| MediaPipe | ≥0.10 | Extracción de keypoints |
| NumPy | ≥1.24 | Operaciones numéricas |
| FastAPI | ≥0.100 | API REST |

---

## 10. Próximos Pasos

1. **Expandir vocabulario** - Añadir más palabras al micro-vocab
2. **Analizar confusiones** - Entender por qué "tú" y "no" tienen bajo recall
3. **Data augmentation** - Aumentar variabilidad en entrenamiento
4. **Integrar con glosador** - Conectar encoder con módulo de traducción
