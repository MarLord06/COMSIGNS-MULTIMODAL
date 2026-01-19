# 🗺️ Roadmap: Capa de Batching para PyTorch

> **Fecha**: 19 de enero de 2026  
> **Estado**: ✅ Completado  
> **Módulo**: `comsigns.core.data.loaders`

---

## 📋 Contexto del Problema

El sistema ComSigns utiliza un encoder multimodal que procesa secuencias temporales de keypoints (manos, cuerpo, rostro). El dataset AEC ya expone instancias `EncoderReadySample` con la estructura:

```python
@dataclass
class EncoderReadySample:
    gloss: str
    hand_keypoints: np.ndarray   # shape: [T, 168]
    body_keypoints: np.ndarray   # shape: [T, 132]
    face_keypoints: np.ndarray   # shape: [T, 1872]
    gloss_id: Optional[int]
```

**Problema**: Las secuencias tienen longitud temporal `T` variable entre muestras, lo que impide crear batches directamente con PyTorch DataLoader.

---

## 🎯 Objetivos

| ID | Objetivo | Prioridad |
|----|----------|-----------|
| O1 | Implementar `collate_fn` para secuencias de longitud variable | Alta |
| O2 | Padding explícito y eficiente | Alta |
| O3 | Generar máscaras temporales para attention/LSTM | Media |
| O4 | Compatibilidad con cualquier dataset que implemente `BaseDataset` | Alta |
| O5 | No modificar el dataset existente | Crítica |

---

## 🏗️ Diseño Propuesto

### Arquitectura

```
comsigns/core/data/
├── datasets/           # Ya existe - NO MODIFICAR
│   ├── base.py
│   ├── sample.py
│   └── aec/
└── loaders/            # NUEVO
    ├── __init__.py
    └── collate.py      # collate_fn genérico
```

### Flujo de Datos

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   AECDataset    │────▶│  DataLoader  │────▶│  EncoderBatch   │
│ [EncoderReady   │     │  + collate   │     │  {hand, body,   │
│  Sample, ...]   │     │              │     │   face, labels, │
└─────────────────┘     └──────────────┘     │   lengths,mask} │
                                             └─────────────────┘
```

### Estructura del Batch de Salida

```python
EncoderBatch = {
    "hand":    Tensor[batch, T_max, 168],   # float32
    "body":    Tensor[batch, T_max, 132],   # float32
    "face":    Tensor[batch, T_max, 1872],  # float32
    "labels":  Tensor[batch],               # int64
    "lengths": Tensor[batch],               # int64 (longitudes originales)
    "mask":    Tensor[batch, T_max]         # bool (True=válido)
}
```

---

## 🔧 Decisiones Técnicas

### 1. Valor de Padding: `0.0`

**Razón**: Los keypoints normalizados tienen valores en rango [0, 1]. Usar `0.0` como padding:
- No distorsiona gradientes durante backprop
- Es distinguible de valores válidos
- Compatible con máscaras de attention

### 2. Inferencia de Dimensiones (no hardcodear)

**Razón**: Las dimensiones 168, 132, 1872 pueden cambiar si:
- Se agregan/quitan keypoints
- Se usa un subconjunto de landmarks
- Se integra otro dataset con formato diferente

```python
# ✅ Correcto: inferir del primer sample
feature_dim = arrays[0].shape[1]

# ❌ Incorrecto: hardcodear
feature_dim = 168
```

### 3. Función Pura (sin estado)

**Razón**: 
- Fácil de testear (input → output determinístico)
- Sin efectos secundarios
- Reutilizable entre datasets

### 4. Pre-allocación con `np.full()`

**Razón**: Más eficiente que concatenar arrays incrementalmente.

```python
# ✅ Eficiente: pre-allocar
padded = np.full((batch, T_max, dim), pad_value, dtype=np.float32)
for i, arr in enumerate(arrays):
    padded[i, :len(arr)] = arr

# ❌ Ineficiente: concatenar
padded = np.concatenate([...])  # múltiples allocations
```

### 5. Máscara de Attention Incluida

**Razón**: 
- LSTMs necesitan `pack_padded_sequence` → requiere lengths
- Transformers necesitan máscara de attention → `mask`
- Incluir ambos permite flexibilidad

---

## 📁 Archivos a Crear

| Archivo | Propósito | Estado |
|---------|-----------|--------|
| `loaders/__init__.py` | Exportar API pública | ✅ |
| `loaders/collate.py` | Implementación de `encoder_collate_fn` | ✅ |
| `tests/unit/test_collate.py` | Tests unitarios | ✅ |

---

## 🧪 Plan de Testing

### Tests Unitarios

| Test | Descripción | Estado |
|------|-------------|--------|
| `test_basic_padding` | Padding correcto a T_max | ✅ |
| `test_custom_pad_value` | Usar valor de padding personalizado | ✅ |
| `test_dtypes` | Verificar tipos de datos de salida | ✅ |
| `test_labels_correct` | Labels preservados correctamente | ✅ |
| `test_lengths_correct` | Longitudes originales preservadas | ✅ |
| `test_mask_correctness` | Máscara True/False correcta | ✅ |
| `test_empty_batch_raises` | Error con batch vacío | ✅ |
| `test_custom_dimensions` | Dimensiones inferidas, no hardcodeadas | ✅ |
| `test_dataloader_iteration` | Integración con DataLoader real | ✅ |

### Comando para Ejecutar Tests

```bash
python3 -m pytest tests/unit/test_collate.py -v
```

---

## 📊 Resultados de Implementación

```
23 passed in 1.23s ✅
```

---

## 💡 Uso

### Básico

```python
from torch.utils.data import DataLoader
from comsigns.core.data.loaders import encoder_collate_fn
from comsigns.core.data.datasets.aec import AECDataset

dataset = AECDataset(Path("data/raw/lsp_aec"))

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=encoder_collate_fn
)

for batch in loader:
    # batch["hand"].shape = [32, T_max, 168]
    # batch["mask"].shape = [32, T_max]
    ...
```

### Con Configuración Personalizada

```python
from comsigns.core.data.loaders import create_encoder_collate_fn

collate_fn = create_encoder_collate_fn(
    pad_value=-1.0,      # Padding con -1 en lugar de 0
    include_mask=False   # Sin máscara de attention
)

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

---

## 🔜 Próximos Pasos Sugeridos

1. **Integrar con MultimodalEncoder**: Modificar `forward()` para aceptar máscara
2. **Sampler balanceado**: Implementar sampler por clase para datasets desbalanceados
3. **Augmentation temporal**: Data augmentation en el collate (random crop, speed perturbation)

---

## 📚 Referencias

- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [pack_padded_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
- Arquitectura ComSigns: `read-cursor.md`
