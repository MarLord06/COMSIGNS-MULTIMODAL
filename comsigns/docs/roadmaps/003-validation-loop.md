# 🗺️ Roadmap: Validación y Diagnóstico de Generalización

> **Fecha**: 20 de enero de 2026  
> **Estado**: ✅ Completado  
> **Módulo**: `comsigns.training` + `comsigns.core.data`

---

## 📋 Contexto del Problema

El sistema ComSigns tiene un trainer funcional que demostró:
- ✅ Loss disminuye en modo overfit (~6.3 → ~0.001)
- ✅ Gradientes no nulos
- ✅ Shapes correctos

**Problema**: No sabemos si el modelo **generaliza** o solo **memoriza**.

**Solución**: Implementar train/validation split y medir ambos losses.

---

## 🎯 Objetivos

| ID | Objetivo | Prioridad |
|----|----------|-----------|
| O1 | Crear split train/validation (80/20) | Alta |
| O2 | Implementar validation loop (forward-only) | Alta |
| O3 | Loggear train_loss y val_loss por epoch | Alta |
| O4 | Mantener backward compatibility (validate=False) | Media |

### No-Objetivos (explícitamente excluidos)

- ❌ Test set
- ❌ Métricas avanzadas (accuracy, WER, BLEU)
- ❌ Early stopping
- ❌ Checkpoints
- ❌ Learning rate schedulers
- ❌ Data augmentation
- ❌ Cambios de arquitectura

---

## 🏗️ Diseño Propuesto

### Arquitectura de Archivos

```
comsigns/
├── core/data/
│   └── splits.py           # NUEVO: funciones de split
├── training/
│   ├── config.py           # MODIFICAR: agregar validate flag
│   ├── loops.py            # MODIFICAR: agregar validate_one_epoch
│   ├── trainer.py          # MODIFICAR: integrar validación
│   └── validation.py       # NUEVO (opcional): lógica de validación
└── scripts/
    └── train.py            # MODIFICAR: usar split
```

### Flujo de Datos

```
┌─────────────┐     random_split      ┌─────────────┐
│  AECDataset │ ───────────────────▶  │ train_set   │ (80%)
│   (full)    │                       │ val_set     │ (20%)
└─────────────┘                       └─────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       
            ┌─────────────┐         ┌─────────────┐
            │train_loader │         │ val_loader  │
            └─────────────┘         └─────────────┘
                    │                       │
                    ▼                       ▼
            train_one_epoch()       validate_one_epoch()
                    │                       │
                    └───────────────────────┘
                                │
                                ▼
                    Epoch X | Train: Y.YY | Val: Z.ZZ
```

### Componentes

#### 1. `create_train_val_split()` (splits.py)

```python
def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: Optional[int] = 42
) -> Tuple[Subset, Subset]:
    """Split dataset into train and validation sets."""
```

**Decisiones**:
- Usar `torch.utils.data.random_split` (estándar PyTorch)
- NO estratificar (simplificación inicial)
- Seed configurable para reproducibilidad

#### 2. `validate_one_epoch()` (loops.py)

```python
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> float:
    """Forward-only validation pass."""
```

**Reglas**:
- `model.eval()`
- `torch.no_grad()`
- NO backward, NO optimizer
- Retorna loss promedio

#### 3. `TrainerConfig` actualizado

```python
@dataclass
class TrainerConfig:
    # ... existing fields ...
    validate: bool = True
    val_ratio: float = 0.2
```

#### 4. `Trainer.fit()` modificado

```python
def fit(self, dataset_or_loader, val_loader=None):
    if self.config.validate and val_loader is None:
        # Auto-split if dataset passed
        train_set, val_set = create_train_val_split(dataset)
        train_loader = DataLoader(train_set, ...)
        val_loader = DataLoader(val_set, ...)
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(...)
        val_loss = validate_one_epoch(...) if val_loader else None
        log(train_loss, val_loss)
```

---

## 🔧 Decisiones Técnicas

### 1. Split fuera del Dataset

**Razón**: Clean Architecture - el Dataset no debe saber sobre splits.

```python
# ✅ Correcto: split externo
train_set, val_set = random_split(dataset, [0.8, 0.2])

# ❌ Incorrecto: split dentro del dataset
dataset = AECDataset(split="train")  # NO
```

### 2. No estratificar inicialmente

**Razón**: 
- Simplificación para diagnóstico inicial
- El dataset AEC puede tener distribución desbalanceada
- Estratificación es optimización prematura aquí

### 3. Validation loop separado

**Razón**: Single Responsibility Principle

```python
# ✅ Correcto: función separada
val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

# ❌ Incorrecto: flag en train_one_epoch
train_one_epoch(..., is_validation=True)  # NO
```

### 4. Backward compatibility con `validate=False`

**Razón**: No romper el modo overfit existente.

```python
if config.validate:
    val_loss = validate_one_epoch(...)
else:
    val_loss = None  # Comportamiento anterior
```

---

## 📁 Archivos a Crear / Modificar

| Archivo | Acción | Propósito | Estado |
|---------|--------|-----------|--------|
| `core/data/splits.py` | CREAR | Funciones de split | ⏳ |
| `core/data/__init__.py` | MODIFICAR | Exportar splits | ⏳ |
| `training/loops.py` | MODIFICAR | Agregar `validate_one_epoch` | ⏳ |
| `training/config.py` | MODIFICAR | Agregar `validate`, `val_ratio` | ⏳ |
| `training/trainer.py` | MODIFICAR | Integrar validación en `fit()` | ⏳ |
| `scripts/train.py` | MODIFICAR | Usar split train/val | ⏳ |
| `tests/unit/test_validation.py` | CREAR | Tests de validación | ⏳ |

---

## 🧪 Plan de Testing

### Tests Unitarios

| Test | Descripción | Estado |
|------|-------------|--------|
| `test_split_ratios` | Split respeta 80/20 | ⏳ |
| `test_split_reproducibility` | Mismo seed = mismo split | ⏳ |
| `test_validate_one_epoch_no_grad` | No hay gradientes en validación | ⏳ |
| `test_validate_returns_float` | Retorna loss promedio | ⏳ |
| `test_trainer_with_validation` | Trainer loggea train y val loss | ⏳ |
| `test_trainer_validate_false` | Trainer funciona sin validación | ⏳ |

### Validaciones de Diagnóstico

| Escenario | train_loss | val_loss | Interpretación |
|-----------|------------|----------|----------------|
| Sano | ↓ | ↓ | ✅ Modelo generaliza |
| Overfitting | ↓ | ↑ | ⚠️ Necesita regularización |
| Underfitting | ≈ | ≈ | ⚠️ Modelo muy simple o LR bajo |

---

## 💡 Ejemplo de Uso Final

```python
# Opción 1: Trainer hace el split automáticamente
trainer = Trainer(model, TrainerConfig(validate=True, val_ratio=0.2))
history = trainer.fit(dataset)  # Pasa dataset completo

# Opción 2: Usuario hace el split manualmente
train_set, val_set = create_train_val_split(dataset)
train_loader = DataLoader(train_set, ...)
val_loader = DataLoader(val_set, ...)
history = trainer.fit(train_loader, val_loader=val_loader)
```

### Output Esperado

```
============================================================
Epoch 1/5
============================================================
Epoch 1 | Step 10/50 | Loss: 6.2341
Epoch 1 | Step 20/50 | Loss: 5.8923
...
Epoch 1 | Train Loss: 5.4321 | Val Loss: 5.6789

============================================================
Epoch 5/5
============================================================
...
Epoch 5 | Train Loss: 1.2345 | Val Loss: 1.8765

Training complete!
  Train Loss: 6.34 → 1.23
  Val Loss: 6.12 → 1.88
  ✅ Modelo está aprendiendo
```

---

## 📊 Métricas de Éxito

| Métrica | Criterio | Estado |
|---------|----------|--------|
| Split correcto | `len(train) + len(val) == len(dataset)` | ✅ |
| Reproducibilidad | Mismo seed → mismo split | ✅ |
| Validación sin gradientes | `param.grad is None` después de val | ✅ |
| Logging correcto | Train y Val loss aparecen | ✅ |
| Backward compatible | `validate=False` funciona como antes | ✅ |

---

## ✅ Resultados de Implementación

**Tests creados**: 19 tests en `tests/unit/test_validation.py`

**Archivos creados/modificados**:
- `core/data/splits.py` - NUEVO: funciones de split
- `core/data/__init__.py` - NUEVO: exports
- `training/config.py` - MODIFICADO: validate, val_ratio
- `training/loops.py` - MODIFICADO: validate_one_epoch
- `training/trainer.py` - MODIFICADO: integración validación
- `scripts/train.py` - MODIFICADO: --no-validate, --val-ratio

**Total tests pasando**: 41 (22 trainer + 19 validation)

---

## 🔜 Próximos Pasos (fuera de scope)

1. Estratificación del split por clase
2. Test set separado
3. Métricas (accuracy, top-k)
4. Early stopping basado en val_loss
5. Checkpointing del mejor modelo

---

## 📚 Referencias

- [PyTorch random_split](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)
- Trainer actual: `training/trainer.py`
- Loops actuales: `training/loops.py`
