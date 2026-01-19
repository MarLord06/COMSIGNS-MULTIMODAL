# 🗺️ Roadmap: Trainer Mínimo Funcional

> **Fecha**: 19 de enero de 2026  
> **Estado**: ✅ Completado  
> **Módulo**: `comsigns.training`

---

## 📋 Contexto del Problema

ComSigns tiene implementados:
- ✅ `MultimodalEncoder` - Procesa keypoints (manos, cuerpo, rostro) → embeddings [B, T, 512]
- ✅ `TemporalSegmenter` + `SegmentAggregator` - Segmenta y agrega embeddings temporales
- ✅ `AECDataset` - Carga muestras con `EncoderReadySample`
- ✅ `encoder_collate_fn` - Crea batches con padding y máscaras

**Falta**: Un Trainer que conecte todo para ejecutar entrenamiento end-to-end.

---

## 🎯 Objetivos

| ID | Objetivo | Prioridad |
|----|----------|-----------|
| O1 | Ejecutar training loop sin errores | Crítica |
| O2 | Verificar que loss disminuye | Alta |
| O3 | Validar backward funciona (gradientes no nulos) | Alta |
| O4 | Modo `overfit_single_batch` para debug | Media |
| O5 | Logging mínimo (step, loss) | Media |

### No-Objetivos (explícitamente excluidos)

- ❌ Métricas avanzadas (WER, accuracy, BLEU)
- ❌ Validación / test split
- ❌ Early stopping
- ❌ Checkpointing avanzado
- ❌ Integración con frameworks externos (Lightning, Ignite)

---

## 🏗️ Diseño Propuesto

### Arquitectura

```
comsigns/
├── training/                 # NUEVO
│   ├── __init__.py          # API pública
│   ├── config.py            # TrainerConfig (dataclass)
│   ├── trainer.py           # Trainer (orquestador)
│   └── loops.py             # train_one_epoch, train
└── services/
    └── encoder/model.py     # Ya existe - NO MODIFICAR
```

### Flujo de Datos

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  DataLoader  │───▶│   Encoder    │───▶│  Classifier  │───▶│   Loss   │
│  (batches)   │    │  [B,T,512]   │    │  [B, C]      │    │  scalar  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────┘
                                                                  │
                                                                  ▼
                                                            ┌──────────┐
                                                            │ Backward │
                                                            └──────────┘
```

### Componentes

#### 1. `TrainerConfig` (config.py)

```python
@dataclass
class TrainerConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 10
    device: str = "cuda"  # o "cpu", "mps"
    log_every_n_steps: int = 10
    overfit_single_batch: bool = False
    gradient_clip_val: Optional[float] = 1.0
```

#### 2. `Trainer` (trainer.py)

Responsabilidades:
- Mover modelo/datos a device
- Orquestar forward → loss → backward → step
- Logging básico
- Modo overfit para debug

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: TrainerConfig
    )
    
    def fit(self, train_loader: DataLoader) -> Dict[str, List[float]]
```

#### 3. `SignLanguageClassifier` (model wrapper)

El `MultimodalEncoder` produce embeddings `[B, T, 512]`. Necesitamos agregar una cabeza de clasificación:

```python
class SignLanguageClassifier(nn.Module):
    def __init__(self, encoder: MultimodalEncoder, num_classes: int):
        self.encoder = encoder
        self.classifier = nn.Linear(512, num_classes)
        self.pooling = "mean"  # o "last"
    
    def forward(self, hand, body, face, lengths=None):
        embeddings = self.encoder(hand, body, face)  # [B, T, 512]
        pooled = self._pool(embeddings, lengths)     # [B, 512]
        logits = self.classifier(pooled)             # [B, C]
        return logits
```

---

## 🔧 Decisiones Técnicas

### 1. Pooling temporal: Mean Pooling con máscara

**Razón**: Las secuencias tienen padding. Mean pooling naive incluiría los zeros.

```python
# ✅ Correcto: mean sobre longitud real
def masked_mean_pool(embeddings, lengths):
    mask = create_mask(lengths)  # [B, T]
    masked = embeddings * mask.unsqueeze(-1)
    return masked.sum(dim=1) / lengths.unsqueeze(-1)

# ❌ Incorrecto: incluye padding
embeddings.mean(dim=1)
```

### 2. Loss Function: CrossEntropyLoss

**Razón**: 
- Clasificación multi-clase estándar
- Labels son integers (gloss_id)
- Internamente aplica softmax

### 3. Optimizer: AdamW

**Razón**:
- Weight decay separado (mejor regularización)
- Estándar para transformers/LSTMs
- Buen default: `lr=1e-4, weight_decay=0.01`

### 4. Gradient Clipping

**Razón**: LSTMs pueden tener gradientes explosivos. Clip a `max_norm=1.0` por seguridad.

### 5. Modo Overfit Single Batch

**Razón**: Técnica de debug estándar:
- Si el modelo no puede memorizar 1 batch, hay bug
- Loss debe ir a ~0 en pocas iteraciones

```python
if config.overfit_single_batch:
    batch = next(iter(train_loader))
    for epoch in range(epochs):
        # Usar siempre el mismo batch
        loss = train_step(batch)
```

---

## 📁 Archivos a Crear

| Archivo | Propósito | Estado |
|---------|-----------|--------|
| `training/__init__.py` | Exportar API pública | ✅ |
| `training/config.py` | `TrainerConfig` dataclass | ✅ |
| `training/trainer.py` | Clase `Trainer` | ✅ |
| `training/loops.py` | `train_one_epoch`, `train` | ✅ |
| `training/classifier.py` | `SignLanguageClassifier` wrapper | ✅ |
| `tests/unit/test_trainer.py` | Tests unitarios | ✅ |

---

## 🧪 Plan de Testing

### Tests Unitarios

| Test | Descripción | Estado |
|------|-------------|--------|
| `test_trainer_config_defaults` | Config tiene valores por defecto | ✅ |
| `test_trainer_single_step` | Un step de training sin crash | ✅ |
| `test_overfit_single_batch` | Loss disminuye en overfit mode | ✅ |
| `test_gradients_not_zero` | Gradientes son no-nulos | ✅ |
| `test_classifier_forward` | Classifier produce logits correctos | ✅ |
| `test_masked_pooling` | Pooling ignora padding | ✅ |

### Resultados de Tests

```
22 passed in 3.25s ✅
```

### Validaciones Mínimas

```python
# 1. Training sin crash
trainer.fit(train_loader)  # No exceptions

# 2. Loss disminuye en overfit mode
losses = trainer.fit(train_loader)
assert losses[-1] < losses[0]

# 3. Gradientes no nulos
for param in model.parameters():
    assert param.grad is not None
    assert param.grad.abs().sum() > 0
```

---

## 💡 Ejemplo de Uso Final

```python
from pathlib import Path
from torch.utils.data import DataLoader

from comsigns.core.data.datasets.aec import AECDataset
from comsigns.core.data.loaders import encoder_collate_fn
from comsigns.services.encoder import MultimodalEncoder
from comsigns.training import (
    Trainer,
    TrainerConfig,
    SignLanguageClassifier
)

# 1. Dataset y DataLoader
dataset = AECDataset(Path("data/raw/lsp_aec"))
loader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=encoder_collate_fn
)

# 2. Modelo
encoder = MultimodalEncoder()
num_classes = len(dataset.gloss_to_id)
model = SignLanguageClassifier(encoder, num_classes)

# 3. Training
config = TrainerConfig(
    epochs=10,
    learning_rate=1e-4,
    device="cuda",
    overfit_single_batch=False  # True para debug
)

trainer = Trainer(model, config)
history = trainer.fit(loader)

# 4. Verificar
print(f"Loss inicial: {history['loss'][0]:.4f}")
print(f"Loss final: {history['loss'][-1]:.4f}")
```

---

## 📊 Métricas de Éxito

| Métrica | Criterio |
|---------|----------|
| Training sin crash | 3 epochs completos |
| Loss decrease | `loss[-1] < loss[0]` en overfit mode |
| Gradientes | Todos non-zero después de backward |
| Tiempo | < 1 min para 100 steps en CPU |

---

## 🔜 Próximos Pasos (fuera de scope)

1. Validation loop
2. Checkpointing (save/load model)
3. Learning rate scheduler
4. Early stopping
5. Métricas (accuracy, WER)
6. TensorBoard logging

---

## 📚 Referencias

- [PyTorch Training Loop](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- MultimodalEncoder: `services/encoder/model.py`
- Collate function: `core/data/loaders/collate.py`
