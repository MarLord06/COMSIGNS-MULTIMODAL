**ComSigns — Documentación del Módulo `training`**

Propósito: describir en detalle cada archivo dentro de `comsigns/training` para que un ingeniero o investigador entienda responsabilidades, entradas, salidas, supuestos y errores comunes.

**Resumen rápido**
- `comsigns/training` contiene la lógica de entrenamiento, métricas, checkpointing, remapeo de clases, limpieza de glosas, aumentación y evaluación final.
- Sus piezas están pensadas para funcionar juntas mediante `Trainer` y funciones puras de `loops`.

**Listado de archivos y descripción detallada**

- `__init__.py`
  - Propósito: inicializa el paquete `comsigns.training` y exporta objetos públicos cuando proceda.
  - No contiene lógica crítica; sirve para importaciones relativas.

- `config.py`
  - Propósito: definición tipada y segura de la configuración de entrenamiento (`TrainerConfig` dataclass).
  - Entradas: parámetros de instancia (epochs, lr, batch_size, device, seed, validate, etc.).
  - Salidas: objetos `TrainerConfig` con validación en `__post_init__` y método `get_torch_device()`.
  - Supuestos: valores razonables; `device='auto'` resuelve a `cuda`/`mps`/`cpu`.
  - Errores comunes: pasar `val_ratio` fuera de (0,1) o `batch_size` ≤ 0 producirá excepciones inmediatas.

- `classifier.py`
  - Propósito: implementación de `SignLanguageClassifier`, que envuelve el `MultimodalEncoder` y añade la cabeza de clasificación.
  - Entradas: tensores `hand`, `body`, `face` de shape `[B, T, dim]`, opcional `lengths`/`mask`.
  - Salidas: logits de clasificación `[B, num_classes]`; también método `get_embeddings()` para extraer features pooled.
  - Componentes clave:
    - Soporta pooling temporal (`mean`, `max`, `last`) con manejo de máscaras/longitudes.
    - `dropout` y `Linear` final para logits.
  - Supuestos: el encoder tiene atributo `output_dim`; `num_classes` > 0.
  - Errores comunes: mismatch en shapes de entrada (seq_len/dim) o `num_classes` inconsistente con `class_mapping`.

- `trainer.py`
  - Propósito: clase `Trainer` de alto nivel que orquesta toda la tanda de entrenamiento (optimizador, device, loops, validación, callbacks y evaluación final).
  - Entradas: `model` (p. ej. `SignLanguageClassifier`), `TrainerConfig`, `optimizer` opcional, `loss_fn`.
  - Salidas: historial `history` con `train_loss`, `val_loss`, `epoch` y (opcional) `final_eval` y `eval_artifacts`.
  - Funcionalidades importantes:
    - Crea/gestiona optimizador si no se proporciona.
    - Maneja semilla, device y metrics tracking (`MetricsTracker`).
    - `fit()` realiza bucle por épocas, llamadas a `train_one_epoch`, validación con `_validate_with_metrics`, logging y callbacks de fin de época (útil para checkpointing).
    - Soporta `run_final_eval` que ejecuta `FinalEvaluator` y guarda artefactos.
  - Supuestos: `train_loader` y `val_loader` devuelven batches con keys `hand`, `body`, `face`, `labels`, `lengths`.
  - Errores comunes: no proporcionar `num_classes` puede deshabilitar `MetricsTracker` (warning); pasar `eval_output_dir=None` con `run_final_eval=True` lanzará error.

- `loops.py`
  - Propósito: funciones puras para entrenamiento/validación (`train_one_epoch`, `validate_one_epoch`, `train`, `validate_gradients`).
  - Entradas: `model`, `dataloader`, `optimizer`, `loss_fn`, `device`, `config`.
  - Salidas: métricas por época o historial.
  - Ventaja: separa la lógica de bucle del `Trainer` para testear o reutilizar.
  - Errores comunes: batches sin las keys esperadas; recordar usar `model.train()` / `model.eval()` apropiadamente.

- `metrics.py`
  - Propósito: `MetricsTracker`, acumulador y computación de métricas de clasificación (top-K, accuracy, precision/recall/f1 macro).
  - Entradas: `logits` (2D) y `labels` (1D) por batch a `update()`.
  - Salidas: `compute()` devuelve dict con `accuracy`, `topK` y (si sklearn instalado) `precision_macro`, `recall_macro`, `f1_macro`.
  - Dependencias: opcional `scikit-learn` para métricas robustas; si no está presente, devuelve zeros para esas métricas y emite warning.
  - Errores comunes: llamar `compute()` sin datos acumulados (retorna ceros con warning); incompatibilidad en shapes de batch.

- `evaluation.py`
  - Propósito: evaluación final tras entrenamiento (`FinalEvaluator`, `EvaluationResult`) que genera métricas por clase, matriz de confusión y artefactos (CSV/PNG/JSON).
  - Entradas: `model` en `eval()` y `dataloader` de validación; `num_classes`, `class_names` opcional.
  - Salidas: `EvaluationResult` con arrays `y_true`, `y_pred`, `y_logits` y métodos para guardar artifacts en disco.
  - Dependencias: `sklearn` y `matplotlib` opcionales para análisis y visualización; si faltan, algunas salidas se omiten con warnings.
  - Errores comunes: llamar `save_artifacts()` antes de `evaluate()` lanza RuntimeError.

- `checkpointing.py`
  - Propósito: `CheckpointManager` para guardar checkpoints por época, seleccionar `best.pt` según criterios y soportar resumption.
  - Entradas: `model`, `optimizer`, métricas (`val_loss`, `f1_macro`, `learned_words_count`), `output_dir`.
  - Salidas: archivos en disco (`epoch_XXX.pt`, `best.pt`, `best_model.json`) y utilidades para cargar estados.
  - Criterios de selección: prioriza `learned_words_count`, luego `f1_macro`, luego `val_loss`.
  - Supuestos: las métricas provistas incluyen al menos las claves requeridas; estructura de checkpoint guarda `model_state`, `optimizer_state`, `metrics`.
  - Errores comunes: mismatch en claves de `metrics` al usar `is_best()`; no encontrar `best_model.json` en resumption (se ignora con warning).

- `remapping.py`
  - Propósito: herramientas para remapear clases (colapsar long-tail a `OTHER` o excluir TAIL) con `ClassRemapper` y `RemapConfig`.
  - Entradas: soporte por clase (dict class_id → count), `class_names` opcional.
  - Salidas: mappings `old_to_new`, `new_to_old`, `new_class_names` y estadística (clases colapsadas, samples in OTHER).
  - Uso típico: transformar datasets o labels antes de entrenar experimentos TAIL→OTHER.
  - Errores comunes: usar `transform()` antes de `fit()` produce RuntimeError; no proporcionar `class_support` correcto conduce a remapeos inesperados.

- `gloss_cleaner.py`
  - Propósito: normalizar/limpiar glosas antes de construir datasets (remover `???_...`, deletreos, prefijos inválidos, glosas no en diccionario).
  - Entradas: conjunto de glosses y `dict.json` opcional para validación semántica.
  - Salidas: `GlossCleaningReport` con detalles y `CleanedGlossDataset` wrapper que filtra muestras.
  - Errores comunes: si `dict.json` no está disponible, el cleaner asume válido; revisar `removals` en el reporte.

- `augmentation.py`
  - Propósito: `KeypointAugmenter` para aplicar aumentos geométricos y temporales sobre arrays de keypoints (ruido, shift temporal, mirror).
  - Entradas: arrays de keypoints con estructura esperada (T, N, 4) — el code preserva canal de confianza al no perturbarlo.
  - Salidas: sample modificado con keypoints aumentados.
  - Supuestos: coordenadas normalizadas (para `mirror`), formato con 4 elementos por keypoint cuando se desea preservar confidence.
  - Errores comunes: aplicar `mirror` cuando datos no normalizados X∈[0,1] puede producir artefactos.

- `experiment_metrics.py`
  - Propósito: utilidades de resumen para experimentos (guardar training_summary.json, agregación de métricas por epoch, selección de checkpoint por criterio compuesto).
  - Entradas: resultados de `Trainer` y `CheckpointManager`.
  - Salidas: JSON/CSV con métricas listas para análisis.
  - Errores comunes: inconsistencia entre nombres de métricas almacenadas y las esperadas por visualizadores externos.

- `rebalance.py`
  - Propósito: funciones para re-balanceo y sampling (oversample/undersample) para lidiar con clases desbalanceadas.
  - Entradas: dataset indices, desired strategy.
  - Salidas: sampler o índices reorder que se usan en `DataLoader`.
  - Errores comunes: usar oversampling sin ajustar weights en loss puede causar overfitting.

- `semantic_closure.py`
  - Propósito: herramientas para agrupar o cerrar semánticamente glosas relacionadas (p. ej. sinónimos) antes del mapeo o evaluación.
  - Entradas: listas de glosas, reglas de cierre semántico.
  - Salidas: diccionarios de agrupamiento usados en análisis de métricas semánticas.
  - Errores comunes: reglas demasiado amplias pueden agrupar clases distintas y falsear métricas.

- `advanced_metrics.py`
  - Propósito: cálculos de métricas avanzadas (learned words report, rejection metrics, composite scoring para selección de modelos).
  - Entradas: matriz de confusión, logits, predicciones y configuración de criterios.
  - Salidas: reportes estructurados (`LearnedWordsReport`, `RejectionMetrics`, etc.) y JSONs para selección de mejor modelo.
  - Errores comunes: depender de `other_class_id` inexistente o no sincronizado entre remapper y experiment.

- `analysis/` (submódulos)
  - Propósito: scripts y funciones auxiliares para análisis post-entrenamiento.
  - Archivos principales:
    - `bucket_analysis.py`: análisis por buckets (HEAD/MID/TAIL) — usa `ClassRemapper` y estadística de soporte.
    - `confusion.py`: utilidades para análisis y visualización de matrices de confusión.
    - `coverage.py` / `coverage_metrics.py`: métricas de coverage (qué fracción del vocabulario es alcanzada por predicciones).
    - `learned_words.py`: herramientas que usan `AdvancedMetricsCalculator` para crear reportes de palabras "aprendidas".
    - `remapped_metrics.py`: métricas específicas para datasets remapeados.
  - Uso: invocados durante evaluación final o en análisis offline de experimentos.

**Recomendaciones prácticas (cómo usar `training` como referencia)**
- Para entrenar con control total, instanciar `Trainer` con `SignLanguageClassifier` y `TrainerConfig`, pasar `DataLoader` con batches que contengan `hand`, `body`, `face`, `labels`, `lengths`.
- Para debug rápido: activar `config.overfit_single_batch=True` o `TrainerConfig(seed=...)` para reproducibilidad.
- Para checkpointing robusto: usar `CheckpointManager` en el callback `epoch_end_callback` y llamar `save_best()` cuando `is_best()` retorne True.
- Para evaluación completa: después de `fit(..., run_final_eval=True, eval_output_dir=...)` revisar `eval_artifacts` guardados en el directorio del experimento.

**Errores y chequeos recomendados**
- Verificar shapes de input (`hand: (B,T,126)`, `body: (B,T,99)`, `face: (B,T,1404)`) antes de pasar al model.
- Confirmar que `class_mapping.json` y `num_classes` coinciden con `SignLanguageClassifier`.
- Instalar `scikit-learn` y `matplotlib` para métricas avanzadas y visualizaciones.

---
Documento añadido a `docs/TRAINING.md`. Si quieres, puedo:
- insertar ejemplos de código para instanciar `Trainer` y `CheckpointManager`,
- o agregar diagramas de flujo de entrenamiento y checkpointing al documento.
---

## Ejemplos de uso

A continuación hay dos snippets prácticos que muestran flujos comunes: (1) entrenar con `CheckpointManager` y callback; (2) reanudar entrenamiento desde un checkpoint y ejecutar evaluación final.

1) Entrenamiento con `CheckpointManager` (callback por época)

```python
from pathlib import Path
import torch

from comsigns.training.trainer import Trainer
from comsigns.training.config import TrainerConfig
from comsigns.training.checkpointing import CheckpointManager
from comsigns.training.classifier import SignLanguageClassifier
from comsigns.services.encoder.model import MultimodalEncoder

# Construir modelo y trainer
encoder = MultimodalEncoder()
model = SignLanguageClassifier(encoder=encoder, num_classes=NUM_CLASSES)
config = TrainerConfig(epochs=20, learning_rate=3e-4, batch_size=16)
trainer = Trainer(model, config)

# Checkpoint manager
exp_dir = Path("experiments/run_001")
ckpt_manager = CheckpointManager(output_dir=exp_dir)

def epoch_end_callback(epoch, model, optimizer, metrics):
  # Guardar checkpoint por época
  ckpt_manager.save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, metrics=metrics)
  # Guardar mejor modelo según criterio compuesto
  if ckpt_manager.is_best(metrics):
    ckpt_manager.save_best(model, metrics, optimizer=optimizer)

# Ejecutar entrenamiento (train_loader/val_loader deben estar definidos)
trainer.fit(
  train_loader,
  val_loader=val_loader,
  epoch_end_callback=epoch_end_callback,
  run_final_eval=True,
  eval_output_dir=exp_dir / "eval"
)
```

2) Reanudar entrenamiento desde un checkpoint y ejecutar evaluación final por separado

```python
import torch
from pathlib import Path

from comsigns.training.trainer import Trainer
from comsigns.training.checkpointing import CheckpointManager
from comsigns.training.evaluation import FinalEvaluator

device = torch.device("cpu")
exp_dir = Path("experiments/run_001")
ckpt_path = exp_dir / "checkpoints" / "epoch_005.pt"

# Cargar estado
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state["model_state"])  # asumir modelo ya instanciado
optimizer.load_state_dict(state.get("optimizer_state", {}))
start_epoch = state.get("epoch", 0) + 1

# Reanudar
trainer = Trainer(model)
trainer.fit(train_loader, val_loader=val_loader, start_epoch=start_epoch)

# Evaluación final independiente
evaluator = FinalEvaluator(model, val_loader, num_classes=NUM_CLASSES, class_names=CLASS_NAMES)
result = evaluator.evaluate()
evaluator.save_artifacts(output_dir=exp_dir / "final_eval")
```

Estas muestras son plantillas: reemplaza `NUM_CLASSES`, `CLASS_NAMES`, `train_loader`, `val_loader`, `optimizer` y rutas por tus objetos y rutas reales del experimento.
