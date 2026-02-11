# 🏋️ ComSigns — Módulo de Entrenamiento

> Documentación completa del módulo `comsigns/training`: trainer, métricas, checkpointing, remapeo de clases, augmentation y evaluación.

---

## 📖 Navegación

| Documento | Descripción |
|-----------|-------------|
| [🧠 Arquitectura del Modelo](MODEL_ARCHITECTURE.md) | Encoder multimodal, ramas, fusión, clasificador |
| [📘 Documentación Técnica](MODEL_TECHNICAL.md) | I/O del modelo, inferencia, limitaciones |
| [👤 Guía de Usuario](USER_GUIDE.md) | Instalación, ejecución, troubleshooting |
| [🏗️ Arquitectura General](../comsigns/docs/ARCHITECTURE.md) | Pipeline + dataset + resultados |

---

## Resumen Rápido

El módulo [`comsigns/training/`](../comsigns/training/) contiene la lógica de entrenamiento, métricas, checkpointing, remapeo de clases, limpieza de glosas, augmentation y evaluación final. Sus piezas funcionan juntas mediante `Trainer` y funciones puras de `loops`.

---

## Archivos y Descripción Detallada

### `config.py` — Configuración de Entrenamiento

- **Propósito:** Definición tipada de la configuración (`TrainerConfig` dataclass).
- **Entradas:** Parámetros (epochs, lr, batch_size, device, seed, etc.).
- **Salidas:** `TrainerConfig` con validación en `__post_init__` y `get_torch_device()`.
- **Supuestos:** `device='auto'` resuelve a `cuda`/`mps`/`cpu`.

> [!WARNING]
> Pasar `val_ratio` fuera de (0,1) o `batch_size` ≤ 0 producirá excepciones inmediatas.

### `classifier.py` — SignLanguageClassifier

**Archivo:** [`comsigns/training/classifier.py`](../comsigns/training/classifier.py)

- **Propósito:** Envuelve el [`MultimodalEncoder`](../comsigns/services/encoder/model.py) y añade la cabeza de clasificación.
- **Entradas:** Tensores `hand`, `body`, `face` de shape `[B, T, dim]`, opcional `lengths`/`mask`.
- **Salidas:** Logits `[B, num_classes]`; método `get_embeddings()` para features pooled.
- **Soporta:** Pooling temporal (`mean`, `max`, `last`) con manejo de máscaras.

> [!IMPORTANT]
> Verificar que `num_classes` coincida con `class_mapping.json` del experimento.

**Arquitectura detallada:** Ver [MODEL_ARCHITECTURE.md § Clasificador](MODEL_ARCHITECTURE.md#4-clasificador)

### `trainer.py` — Orquestador de Entrenamiento

- **Propósito:** Clase `Trainer` de alto nivel que orquesta todo el entrenamiento.
- **Entradas:** `model`, `TrainerConfig`, `optimizer` opcional, `loss_fn`.
- **Salidas:** Historial con `train_loss`, `val_loss`, `epoch` y `final_eval`.
- **Funcionalidades:**
  - Crea/gestiona optimizador si no se proporciona
  - Maneja semilla, device y metrics tracking (`MetricsTracker`)
  - `fit()` realiza bucle por épocas con validación y callbacks
  - Soporta `run_final_eval` con `FinalEvaluator`

> [!WARNING]
> `train_loader` y `val_loader` deben devolver batches con keys `hand`, `body`, `face`, `labels`, `lengths`.

### `loops.py` — Funciones Puras de Entrenamiento

- **Propósito:** Funciones para `train_one_epoch`, `validate_one_epoch`, `train`, `validate_gradients`.
- **Ventaja:** Separa la lógica de bucle del `Trainer` para testear o reutilizar.

### `metrics.py` — MetricsTracker

- **Propósito:** Acumulador y computación de métricas (top-K, accuracy, precision/recall/f1 macro).
- **Entradas:** `logits` (2D) y `labels` (1D) por batch a `update()`.
- **Salidas:** `compute()` → dict con `accuracy`, `topK`, `precision_macro`, `recall_macro`, `f1_macro`.
- **Dependencia opcional:** `scikit-learn` para métricas robustas.

### `evaluation.py` — FinalEvaluator

- **Propósito:** Evaluación final tras entrenamiento — métricas por clase, matriz de confusión, artefactos (CSV/PNG/JSON).
- **Entradas:** `model` en `eval()` y `dataloader` de validación.
- **Salidas:** `EvaluationResult` con `y_true`, `y_pred`, `y_logits` y métodos para guardar artifacts.
- **Dependencias opcionales:** `sklearn`, `matplotlib`.

> [!WARNING]
> Llamar `save_artifacts()` antes de `evaluate()` lanza `RuntimeError`.

### `checkpointing.py` — CheckpointManager

- **Propósito:** Guardar checkpoints por época, seleccionar `best.pt` según criterios, soportar resumption.
- **Entradas:** `model`, `optimizer`, métricas, `output_dir`.
- **Salidas:** Archivos en disco (`epoch_XXX.pt`, `best.pt`, `best_model.json`).
- **Criterios de selección:** Prioriza `learned_words_count` → `f1_macro` → `val_loss`.

### `remapping.py` — Remapeo de Clases

- **Propósito:** Colapsar long-tail a `OTHER` o excluir TAIL con `ClassRemapper` y `RemapConfig`.
- **Entradas:** Soporte por clase (dict class_id → count), `class_names` opcional.
- **Salidas:** Mappings `old_to_new`, `new_to_old`, `new_class_names`.

> [!WARNING]
> Usar `transform()` antes de `fit()` produce `RuntimeError`.

### `gloss_cleaner.py` — Limpieza de Glosas

- **Propósito:** Normalizar/limpiar glosas antes de construir datasets (remover `???_...`, deletreos, prefijos inválidos).
- **Salidas:** `GlossCleaningReport` con detalles y `CleanedGlossDataset` wrapper.

### `augmentation.py` — Data Augmentation

- **Propósito:** `KeypointAugmenter` para aumentos geométricos y temporales (ruido, shift, mirror).
- **Entradas:** Arrays de keypoints con estructura `(T, N, 4)`.

> [!WARNING]
> Aplicar `mirror` con datos no normalizados X∈[0,1] puede producir artefactos.

### `experiment_metrics.py` — Resumen de Experimentos

- **Propósito:** Guardar `training_summary.json`, agregación de métricas, selección de checkpoint por criterio compuesto.

### `rebalance.py` — Re-balanceo

- **Propósito:** Oversample/undersample para clases desbalanceadas.
- **Salidas:** Sampler o índices para `DataLoader`.

### `semantic_closure.py` — Cierre Semántico

- **Propósito:** Agrupar glosas relacionadas (sinónimos) antes del mapeo o evaluación.

### `advanced_metrics.py` — Métricas Avanzadas

- **Propósito:** Learned words report, rejection metrics, composite scoring.

### `analysis/` — Análisis Post-Entrenamiento

| Archivo | Propósito |
|---------|-----------|
| `bucket_analysis.py` | Análisis por buckets (HEAD/MID/TAIL) |
| `confusion.py` | Matrices de confusión |
| `coverage.py` / `coverage_metrics.py` | Métricas de vocabulary coverage |
| `learned_words.py` | Reportes de palabras aprendidas |
| `remapped_metrics.py` | Métricas para datasets remapeados |

---

## Recomendaciones Prácticas

### Entrenamiento con Control Total

```python
from comsigns.training.trainer import Trainer
from comsigns.training.config import TrainerConfig
from comsigns.training.classifier import SignLanguageClassifier
from comsigns.services.encoder.model import MultimodalEncoder

encoder = MultimodalEncoder()
model = SignLanguageClassifier(encoder=encoder, num_classes=NUM_CLASSES)
config = TrainerConfig(epochs=20, learning_rate=3e-4, batch_size=16)
trainer = Trainer(model, config)

trainer.fit(train_loader, val_loader=val_loader)
```

### Checkpointing por Época

```python
from comsigns.training.checkpointing import CheckpointManager

exp_dir = Path("experiments/run_001")
ckpt_manager = CheckpointManager(output_dir=exp_dir)

def epoch_end_callback(epoch, model, optimizer, metrics):
    ckpt_manager.save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, metrics=metrics)
    if ckpt_manager.is_best(metrics):
        ckpt_manager.save_best(model, metrics, optimizer=optimizer)

trainer.fit(
    train_loader,
    val_loader=val_loader,
    epoch_end_callback=epoch_end_callback,
    run_final_eval=True,
    eval_output_dir=exp_dir / "eval"
)
```

### Reanudar Entrenamiento

```python
import torch
from comsigns.training.evaluation import FinalEvaluator

state = torch.load(exp_dir / "checkpoints/epoch_005.pt", map_location="cpu")
model.load_state_dict(state["model_state"])
optimizer.load_state_dict(state.get("optimizer_state", {}))

trainer = Trainer(model)
trainer.fit(train_loader, val_loader=val_loader, start_epoch=state.get("epoch", 0) + 1)

# Evaluación final independiente
evaluator = FinalEvaluator(model, val_loader, num_classes=NUM_CLASSES, class_names=CLASS_NAMES)
result = evaluator.evaluate()
evaluator.save_artifacts(output_dir=exp_dir / "final_eval")
```

---

## Chequeos Recomendados

- ✅ Verificar shapes de input: `hand (B,T,126)`, `body (B,T,99)`, `face (B,T,1404)`
- ✅ Confirmar que `class_mapping.json` y `num_classes` coinciden con `SignLanguageClassifier`
- ✅ Instalar `scikit-learn` y `matplotlib` para métricas avanzadas y visualizaciones

---

## 📚 Documentos Relacionados

- [🧠 Arquitectura del Modelo](MODEL_ARCHITECTURE.md) — Diagramas del encoder y clasificador
- [📘 Documentación Técnica](MODEL_TECHNICAL.md) — I/O, inferencia, limitaciones
- [👤 Guía de Usuario](USER_GUIDE.md) — Instalación y ejecución
- [🏗️ Arquitectura General](../comsigns/docs/ARCHITECTURE.md) — Pipeline + dataset + resultados
- [📜 Referencia de Scripts](../comsigns/docs/SCRIPTS_USAGE.md) — CLI flags para scripts de entrenamiento
