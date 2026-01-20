# 🗺️ Roadmap: Módulo de Métricas Desacoplado

> **Fecha**: 20 de enero de 2026  
> **Estado**: ✅ Completado  
> **Módulo**: `comsigns.training.metrics`

---

## 📋 Contexto del Problema

El sistema ComSigns tiene:
- ✅ Training loop funcional
- ✅ Validation loop con loss
- ❌ Sin métricas de clasificación (accuracy, F1, etc.)

**Problema**: Solo medimos loss, no sabemos el rendimiento real del clasificador.

**Solución**: Implementar módulo de métricas desacoplado del trainer.

---

## 🎯 Objetivos

| ID | Objetivo | Prioridad |
|----|----------|-----------|
| O1 | Implementar Top-K Accuracy (K=1,5,10) | Alta |
| O2 | Implementar Accuracy global | Alta |
| O3 | Implementar Precision/Recall/F1 macro | Alta |
| O4 | Clase MetricsTracker acumulativa | Alta |
| O5 | Diseño extensible para futuras métricas | Media |

### No-Objetivos (explícitamente excluidos)

- ❌ Métricas por modalidad (hand/face/body)
- ❌ Métricas por source_video_name
- ❌ Exportación a CSV/JSON
- ❌ Visualización de métricas
- ❌ Modificar el training loop

---

## 🏗️ Diseño Propuesto

### Arquitectura de Archivos

```
comsigns/
└── training/
    ├── __init__.py          # MODIFICAR: exportar MetricsTracker
    ├── metrics.py           # NUEVO: módulo de métricas
    └── ...
```

### Interfaz Principal

```python
class MetricsTracker:
    """Acumula predicciones y calcula métricas por epoch."""
    
    def __init__(
        self,
        num_classes: int,
        topk: Tuple[int, ...] = (1, 5, 10),
        device: str = "cpu"
    ):
        ...
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Acumula batch de predicciones."""
        ...
    
    def compute(self) -> Dict[str, float]:
        """Calcula todas las métricas sobre datos acumulados."""
        ...
    
    def reset(self) -> None:
        """Reinicia el tracker para nuevo epoch."""
        ...
```

### Formato de Salida

```python
{
    "accuracy": 0.7234,
    "top1_acc": 0.7234,
    "top5_acc": 0.8912,
    "top10_acc": 0.9456,
    "precision_macro": 0.6821,
    "recall_macro": 0.6543,
    "f1_macro": 0.6678
}
```

### Flujo de Datos

```
┌─────────────────┐     update()      ┌─────────────────────┐
│  Validation     │ ────────────────▶ │  MetricsTracker     │
│  Loop           │    (logits,       │  - all_logits: []   │
│                 │     labels)       │  - all_labels: []   │
└─────────────────┘                   └─────────────────────┘
                                               │
                                      compute() (end of epoch)
                                               │
                                               ▼
                                      ┌─────────────────────┐
                                      │  Dict[str, float]   │
                                      │  accuracy, top-k,   │
                                      │  precision, recall  │
                                      └─────────────────────┘
```

---

## 📝 Especificaciones de Implementación

### Top-K Accuracy (PyTorch puro)

```python
def _compute_topk_accuracy(
    logits: torch.Tensor,  # [N, C]
    labels: torch.Tensor,  # [N]
    k: int
) -> float:
    _, topk_preds = logits.topk(k, dim=1)  # [N, k]
    correct = topk_preds.eq(labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()
```

### Precision/Recall/F1 (sklearn)

```python
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true=labels_np,
    y_pred=predictions_np,
    average='macro',
    zero_division=0  # Evitar warnings
)
```

### Manejo de Edge Cases

| Caso | Solución |
|------|----------|
| K > num_classes | Usar min(K, num_classes) |
| División por cero | zero_division=0 en sklearn |
| GPU tensors | Mover a CPU antes de sklearn |
| Empty tracker | Retornar 0.0 para todas las métricas |

---

## 🧪 Tests Requeridos

| Test | Descripción |
|------|-------------|
| `test_update_accumulates` | Verifica que update() acumula datos |
| `test_reset_clears_data` | Verifica que reset() limpia el estado |
| `test_top1_accuracy_correct` | Top-1 accuracy con datos conocidos |
| `test_top5_accuracy_correct` | Top-5 accuracy con datos conocidos |
| `test_perfect_predictions` | Todas las métricas = 1.0 |
| `test_random_predictions` | Métricas en rango esperado |
| `test_compute_returns_dict` | Formato de salida correcto |
| `test_handles_gpu_tensors` | Funciona con tensores CUDA |
| `test_handles_empty_tracker` | No crashea sin datos |

---

## 📊 Uso Esperado

```python
# En el validation loop (sin modificar trainer)
metrics = MetricsTracker(num_classes=505, topk=(1, 5, 10))

for batch in val_loader:
    with torch.no_grad():
        logits = model(batch)
        labels = batch["labels"]
        metrics.update(logits, labels)

results = metrics.compute()
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Top-5 Acc: {results['top5_acc']:.2%}")
print(f"F1 Macro: {results['f1_macro']:.4f}")

metrics.reset()  # Para siguiente epoch
```

---

## ✅ Criterios de Aceptación

- [ ] `training/metrics.py` implementado
- [ ] MetricsTracker con update/compute/reset
- [ ] Top-K accuracy funcional
- [ ] Precision/Recall/F1 macro funcional
- [ ] Tests unitarios pasan
- [ ] No modifica trainer existente
- [ ] Documentación clara

---

## 📚 Referencias

- PyTorch topk: `torch.topk()`
- sklearn metrics: `precision_recall_fscore_support`
- Clean Architecture: Métricas son responsabilidad separada del training
