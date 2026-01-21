# 🗺️ Roadmap: Métricas Avanzadas por Clase y Análisis de Cobertura

> **Fecha**: 20 de enero de 2026  
> **Estado**: 🔄 En progreso  
> **Módulo**: `comsigns.training.metrics` + `comsigns.analysis`

---

## 📋 Contexto del Problema

El sistema actual tiene:
- ✅ Métricas globales (accuracy, top-k, F1 macro)
- ❌ Sin métricas por clase individual
- ❌ Sin análisis de cobertura del dataset
- ❌ Sin matriz de confusión
- ❌ Sin identificación de clases problemáticas

**Problema**: No podemos diagnosticar qué clases están fallando ni por qué.

**Solución**: Implementar métricas granulares por clase + análisis de dataset.

---

## 🎯 Objetivos

| ID | Objetivo | Prioridad |
|----|----------|-----------|
| O1 | Métricas por clase (P/R/F1/Support) | Alta |
| O2 | Top-K accuracy por clase | Alta |
| O3 | Análisis de cobertura del dataset | Alta |
| O4 | Matriz de confusión exportable | Media |
| O5 | Logging estructurado por clase | Media |
| O6 | Identificación automática de clases problemáticas | Media |

### No-Objetivos

- ❌ Cambiar el flujo principal de entrenamiento
- ❌ Data augmentation o rebalanceo (solo diagnóstico)
- ❌ Métricas por signer (signer_id=-1)

---

## 🏗️ Diseño Propuesto

### Arquitectura de Archivos

```
comsigns/
├── training/
│   └── metrics.py              # EXTENDER: agregar per-class metrics
├── analysis/
│   ├── __init__.py            # NUEVO
│   ├── coverage.py            # NUEVO: análisis de cobertura
│   └── confusion.py           # NUEVO: matriz de confusión
└── scripts/
    └── analyze_dataset.py     # NUEVO: script de análisis
```

### Extensión de MetricsTracker

```python
class MetricsTracker:
    # ... métodos existentes ...
    
    def compute_per_class(self) -> Dict[str, Dict[str, float]]:
        """Retorna métricas por clase."""
        return {
            "yo": {"precision": 0.8, "recall": 0.7, ...},
            "hola": {"precision": 0.6, ...},
            ...
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Retorna matriz de confusión [C x C]."""
        ...
    
    def get_worst_classes(self, k: int = 10) -> List[str]:
        """Retorna las K clases con peor F1."""
        ...
```

### Formato de Salida por Clase

```python
{
    "yo": {
        "support": 12,
        "precision": 0.31,
        "recall": 0.25,
        "f1": 0.28,
        "accuracy": 0.25,
        "top5_acc": 0.75
    },
    ...
}
```

### Análisis de Cobertura

```python
{
    "total_classes": 505,
    "total_instances": 2308,
    "distribution": {
        "min": 1,
        "max": 45,
        "mean": 4.57,
        "median": 3,
        "std": 5.2
    },
    "low_support_classes": ["raro", "extraño", ...],  # < 5 samples
    "high_support_classes": ["yo", "hola", ...],      # > 20 samples
}
```

---

## 📝 Especificaciones de Implementación

### 1. Per-Class Metrics (sklearn)

```python
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, 
    average=None,  # Per-class, not macro
    zero_division=0,
    labels=range(num_classes)
)
```

### 2. Top-K Accuracy por Clase

```python
def compute_topk_per_class(logits, labels, k, num_classes):
    """Compute Top-K accuracy for each class."""
    topk_acc = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            topk_acc[c] = 0.0
            continue
        class_logits = logits[mask]
        class_labels = labels[mask]
        _, topk_preds = class_logits.topk(k, dim=1)
        correct = topk_preds.eq(class_labels.unsqueeze(1)).any(dim=1)
        topk_acc[c] = correct.float().mean().item()
    return topk_acc
```

### 3. Logging Estructurado

```
=== Per-Class Metrics (Top 5 Best) ===
Class "hola" | Support: 23 | P: 0.85 | R: 0.78 | F1: 0.81 | Top5: 0.95
Class "gracias" | Support: 18 | P: 0.72 | R: 0.67 | F1: 0.69 | Top5: 0.89
...

=== Per-Class Metrics (Top 5 Worst) ===
Class "raro" | Support: 2 | P: 0.00 | R: 0.00 | F1: 0.00 | Top5: 0.50
...

=== Summary ===
Classes with F1 > 0.5: 45/505 (8.9%)
Classes with F1 = 0: 320/505 (63.4%)
Mean F1 (non-zero support): 0.12
```

---

## 🧪 Tests Requeridos

| Test | Descripción |
|------|-------------|
| `test_per_class_metrics_shape` | Métricas para cada clase |
| `test_confusion_matrix_shape` | Matriz [C x C] |
| `test_coverage_analysis` | Estadísticas correctas |
| `test_worst_classes_ranking` | Ordenamiento por F1 |
| `test_topk_per_class` | Top-K por clase correcto |

---

## ✅ Criterios de Aceptación

- [ ] MetricsTracker extendido con `compute_per_class()`
- [ ] Matriz de confusión exportable
- [ ] Script de análisis de cobertura
- [ ] Logging estructurado en trainer
- [ ] Tests unitarios pasan
- [ ] No rompe compatibilidad con trainer actual

---

## 📚 Referencias

- sklearn.metrics.classification_report
- sklearn.metrics.confusion_matrix
- Trainer actual: `training/trainer.py`
- MetricsTracker: `training/metrics.py`
