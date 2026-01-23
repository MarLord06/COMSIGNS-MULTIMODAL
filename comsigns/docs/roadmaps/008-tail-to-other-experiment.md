# Roadmap 008: TAIL → OTHER Experiment

## Contexto

### Estado Actual
- **Dataset**: 505 clases (glosas), ~2278 muestras totales
- **Distribución por bucket** (basado en training support):
  - HEAD (≥10 samples): 3 clases, 39 muestras
  - MID (3-9 samples): 47 clases, 222 muestras  
  - TAIL (1-2 samples): 455 clases, 260 muestras
- **Diagnóstico**: TAIL representa 90.1% del vocabulario pero Acc@1=0%, Acc@5=0%
- **Conclusión**: TAIL es ruido estadístico en esta fase

### Objetivo
Implementar un **experimento controlado** donde todas las clases TAIL se colapsan en una única clase "OTHER" para:
1. Limpiar la señal de entrenamiento
2. Medir el impacto real del long-tail
3. Evaluar si el modelo aprende semántica útil sin el ruido del TAIL

---

## Arquitectura de la Solución

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TAIL → OTHER Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  dict.json ──► BucketClassifier ──► ClassRemapper ──► Dataset      │
│                     │                     │               │         │
│                     ▼                     ▼               ▼         │
│              bucket_mapping.json   class_mapping.json  Collate     │
│              (class→bucket)        (old_id→new_id)       │         │
│                                                          ▼         │
│                                                       Trainer      │
│                                                          │         │
│                                                          ▼         │
│                                                    Evaluation      │
│                                                          │         │
│                                                          ▼         │
│                                                   Comparison       │
│                                              (Baseline vs OTHER)   │
└─────────────────────────────────────────────────────────────────────┘
```

### Nuevo Vocabulario
```
ANTES (505 clases):
  [0] -adulto
  [1] -ustedes
  ...
  [504] último

DESPUÉS (51 clases = HEAD + MID + OTHER):
  [0]  clase_HEAD_0
  [1]  clase_HEAD_1
  [2]  clase_HEAD_2
  [3]  clase_MID_0
  ...
  [49] clase_MID_46
  [50] OTHER          ← Todas las 455 clases TAIL
```

---

## Tareas de Implementación

### Fase 1: Módulo de Remapping (Core)

#### 1.1 ClassRemapper
**Archivo**: `training/remapping.py`

```python
@dataclass
class RemapConfig:
    """Configuration for class remapping."""
    strategy: Literal["tail_to_other", "tail_exclude"]
    head_threshold: int = 10
    mid_range: Tuple[int, int] = (3, 9)
    other_class_name: str = "OTHER"

class ClassRemapper:
    """Maps original class IDs to new collapsed IDs."""
    
    def __init__(self, config: RemapConfig):
        self.config = config
        self.old_to_new: Dict[int, int] = {}
        self.new_to_old: Dict[int, List[int]] = {}
        self.bucket_to_classes: Dict[Bucket, List[int]] = {}
        self.other_class_id: int = -1
    
    def fit(self, class_support: Dict[int, int]) -> "ClassRemapper":
        """Build mapping from class support counts."""
        ...
    
    def transform(self, old_class_id: int) -> int:
        """Map old class ID to new class ID."""
        ...
    
    def inverse_transform(self, new_class_id: int) -> List[int]:
        """Get original class IDs for a new class ID."""
        ...
    
    def save(self, path: Path) -> None:
        """Persist mapping to JSON."""
        ...
    
    @classmethod
    def load(cls, path: Path) -> "ClassRemapper":
        """Load mapping from JSON."""
        ...
```

**Propiedades esperadas**:
- `num_classes_original`: 505
- `num_classes_remapped`: 51 (HEAD + MID + OTHER)
- `other_class_id`: 50
- Mapping determinístico y reproducible

#### 1.2 Exportación de Configuración
**Archivo de salida**: `experiments/run_XXX/tail_to_other_config.json`

```json
{
  "strategy": "tail_to_other",
  "head_threshold": 10,
  "mid_range": [3, 9],
  "other_class_name": "OTHER",
  "num_classes_original": 505,
  "num_classes_remapped": 51,
  "other_class_id": 50,
  "classes_collapsed": 455,
  "samples_in_other": 260
}
```

**Archivo de salida**: `experiments/run_XXX/class_mapping.json`

```json
{
  "old_to_new": {
    "0": 50,
    "1": 50,
    "106": 0,
    "367": 1,
    "478": 2,
    ...
  },
  "new_to_old": {
    "0": [106],
    "1": [367],
    "2": [478],
    ...
    "50": [0, 1, 2, 3, 4, ...]
  },
  "new_class_names": {
    "0": "HABLAR",
    "1": "YO",
    "2": "TU",
    ...
    "50": "OTHER"
  }
}
```

---

### Fase 2: Dataset Wrapper

#### 2.1 RemappedDataset
**Archivo**: `training/remapping.py` (mismo archivo)

```python
class RemappedDataset(torch.utils.data.Dataset):
    """Wrapper that applies class remapping to an existing dataset."""
    
    def __init__(
        self,
        base_dataset: BaseDataset,
        remapper: ClassRemapper
    ):
        self.base_dataset = base_dataset
        self.remapper = remapper
    
    def __getitem__(self, idx: int) -> EncoderReadySample:
        sample = self.base_dataset[idx]
        # Remap gloss_id
        sample.gloss_id = self.remapper.transform(sample.gloss_id)
        return sample
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    @property
    def num_classes(self) -> int:
        return self.remapper.num_classes_remapped
```

**Compatibilidad**:
- ✅ Collate existente funciona sin cambios
- ✅ Trainer recibe `num_classes` correcto
- ✅ No modifica dataset original

---

### Fase 3: Script de Entrenamiento

#### 3.1 Modificar `scripts/train.py`
Agregar flags:

```python
parser.add_argument(
    "--tail-to-other",
    action="store_true",
    help="Collapse TAIL classes into a single OTHER class"
)
parser.add_argument(
    "--head-threshold",
    type=int,
    default=10,
    help="Minimum samples for HEAD bucket"
)
```

#### 3.2 Flujo de Entrenamiento

```python
if args.tail_to_other:
    # 1. Compute training support
    train_support = compute_support(train_dataset)
    
    # 2. Create remapper
    remapper = ClassRemapper(RemapConfig(
        strategy="tail_to_other",
        head_threshold=args.head_threshold
    ))
    remapper.fit(train_support)
    
    # 3. Wrap datasets
    train_dataset = RemappedDataset(train_dataset, remapper)
    val_dataset = RemappedDataset(val_dataset, remapper)
    
    # 4. Save config
    remapper.save(output_dir / "class_mapping.json")
    
    # 5. Update num_classes for model
    num_classes = remapper.num_classes_remapped
```

---

### Fase 4: Métricas Comparables

#### 4.1 Métricas por Bucket (Post-Colapso)
**Archivo**: `training/analysis/remapped_metrics.py`

```python
class RemappedMetricsTracker:
    """Track metrics with awareness of class remapping."""
    
    def compute_bucket_metrics(self) -> Dict[str, BucketMetrics]:
        """Compute metrics for HEAD, MID, and OTHER buckets."""
        return {
            "HEAD": self._compute_for_bucket(Bucket.HEAD),
            "MID": self._compute_for_bucket(Bucket.MID),
            "OTHER": self._compute_for_other()
        }
    
    def compute_other_diagnostics(self) -> Dict:
        """Compute OTHER-specific diagnostics."""
        return {
            "samples_mapped_to_other": ...,
            "pct_samples_in_other": ...,
            "other_precision": ...,
            "other_recall": ...,
            "confusion_to_other": ...,  # Non-OTHER predicted as OTHER
            "confusion_from_other": ...,  # OTHER predicted as non-OTHER
        }
    
    def compute_prediction_diagnostics(self) -> Dict:
        """Compute prediction distribution diagnostics.
        
        ⚠️ IMPORTANTE: Accuracy puede subir "falsamente" si el modelo
        predice mucho OTHER. Estas métricas ayudan a interpretar.
        """
        return {
            "pct_predictions_other": ...,     # % de predicciones que son OTHER
            "entropy_of_predictions": ...,    # Diversidad de predicciones
            "num_unique_predictions": ...,    # Cuántas clases distintas predice
        }
```

#### 4.2 Comparativa Baseline vs TAIL→OTHER
**Archivo de salida**: `experiments/run_XXX/comparison_report.json`

> ⚠️ **NOTA sobre F1**: Reportamos AMBOS F1 macro y weighted:
> - **F1 macro**: Equidad entre clases (cada clase pesa igual)
> - **F1 weighted**: Impacto real en datos (ponderado por soporte)
> 
> Con OTHER como clase grande, F1 macro puede ser engañoso.

```json
{
  "baseline": {
    "accuracy": 0.090,
    "top5_accuracy": 0.090,
    "f1_macro": 0.008,
    "f1_weighted": 0.031,
    "head_accuracy": 0.615,
    "mid_accuracy": 0.104,
    "tail_accuracy": 0.000
  },
  "tail_to_other": {
    "accuracy": "???",
    "top5_accuracy": "???",
    "f1_macro": "???",
    "f1_weighted": "???",
    "head_accuracy": "???",
    "mid_accuracy": "???",
    "other_accuracy": "???",
    "pct_predictions_other": "???",
    "entropy_of_predictions": "???"
  },
  "delta": {
    "accuracy": "???",
    "top5_accuracy": "???",
    "f1_macro": "???",
    "f1_weighted": "???",
    "head_accuracy": "???",
    "mid_accuracy": "???"
  },
  "interpretation_notes": {
    "warning_if_pct_other_high": "Si pct_predictions_other > 50%, accuracy puede estar inflada",
    "warning_head_only_3_classes": "HEAD tiene solo 3 clases - mejoras fuertes pueden no generalizar"
  }
}
```

---

### Fase 5: Tests

#### 5.1 Tests Unitarios
**Archivo**: `tests/unit/test_remapping.py`

| Test | Descripción |
|------|-------------|
| `test_remapper_fit_creates_mapping` | Fit crea mapping válido |
| `test_remapper_deterministic` | Mismo input → mismo output |
| `test_remapper_head_preserved` | HEAD classes mantienen identidad (diferente ID) |
| `test_remapper_mid_preserved` | MID classes mantienen identidad |
| `test_remapper_tail_collapsed` | Todas TAIL → OTHER |
| `test_remapper_transform` | Transform retorna new_class_id correcto |
| `test_remapper_inverse_transform` | Inverse retorna old_class_ids correctos |
| `test_remapper_save_load` | Persistencia JSON funciona |
| `test_remapped_dataset_num_classes` | num_classes es correcto |
| `test_remapped_dataset_getitem` | __getitem__ retorna gloss_id remapeado |
| `test_remapped_dataset_collate_compatible` | Collate funciona igual |
| `test_other_class_is_last` | OTHER siempre es último índice |
| `test_no_class_id_collision` | Sin colisiones en new_class_id |

---

## Archivos a Crear/Modificar

### Nuevos Archivos
| Archivo | Descripción |
|---------|-------------|
| `training/remapping.py` | ClassRemapper, RemapConfig, RemappedDataset |
| `training/analysis/remapped_metrics.py` | Métricas post-remapping |
| `tests/unit/test_remapping.py` | Tests unitarios |

### Archivos a Modificar
| Archivo | Cambio |
|---------|--------|
| `scripts/train.py` | Agregar `--tail-to-other` flag |
| `training/__init__.py` | Exportar nuevos módulos |

### Outputs del Experimento
| Archivo | Contenido |
|---------|-----------|
| `tail_to_other_config.json` | Configuración del experimento |
| `class_mapping.json` | Mapping old→new class IDs |
| `metrics_global.json` | Métricas globales |
| `metrics_by_bucket.json` | Métricas por bucket (HEAD/MID/OTHER) |
| `comparison_report.json` | Comparativa vs baseline |

---

## Restricciones (NO hacer)

- ❌ NO implementar few-shot
- ❌ NO cambiar arquitectura del modelo
- ❌ NO cambiar encoder ni segmentación temporal
- ❌ NO introducir nuevas loss functions
- ❌ NO modificar el split estratificado existente
- ❌ NO cambiar hiperparámetros (lr, batch_size, epochs)

---

## Criterios de Éxito

| Criterio | Métrica | Umbral |
|----------|---------|--------|
| HEAD mejora | Acc@1 HEAD | > baseline (61.5%) |
| MID mejora | Acc@1 MID | > baseline (10.4%) |
| Confusión reducida | Confusion hacia OTHER | Documentado |
| Reproducibilidad | Same seed → same results | ✅ |
| Pipeline estable | 242+ tests passing | ✅ |

---

## Ejecución del Experimento

```bash
# Baseline (ya ejecutado)
python3 scripts/train.py --stratified --epochs 5 --eval

# TAIL → OTHER
python3 scripts/train.py --stratified --epochs 5 --eval --tail-to-other

# Comparar resultados
python3 scripts/compare_experiments.py \
    experiments/run_baseline/ \
    experiments/run_tail_to_other/
```

---

## Preguntas a Responder

1. **¿Mejora significativamente HEAD y MID?**
   - Comparar Acc@1, Acc@5, F1 por bucket
   - ⚠️ HEAD tiene solo 3 clases (39 muestras) - mejoras fuertes pueden no indicar generalización real

2. **¿Disminuye la confusión global?**
   - Analizar matriz de confusión **reducida** (ver sección Matriz de Confusión)
   - Ver si predicciones se distribuyen mejor

3. **¿El modelo deja de colapsar predicciones?**
   - Verificar `pct_predictions_other` - si > 50%, accuracy puede estar inflada
   - Verificar `entropy_of_predictions` - mayor entropía = más diversidad
   - ¿Predice muchas clases o pocas?

4. **¿Las métricas son estables y reproducibles?**
   - Ejecutar con misma seed múltiples veces
   - Verificar consistencia

---

## Matriz de Confusión

> ⚠️ **NO visualizar matriz completa 51×51** - es ilegible y no aporta insights.

### Enfoque Correcto: Matriz Reducida 3×3

Agregar confusión a nivel de **bucket**:

```
             Predicted
             HEAD    MID    OTHER
Actual HEAD   [.]    [.]    [.]
Actual MID    [.]    [.]    [.]  
Actual OTHER  [.]    [.]    [.]
```

### Métricas de Confusión Agregadas

```python
def compute_bucket_confusion(self) -> Dict:
    """Confusión agregada por bucket."""
    return {
        "head_to_other": ...,   # HEAD real → predicho OTHER
        "mid_to_other": ...,    # MID real → predicho OTHER
        "other_to_head": ...,   # OTHER real → predicho HEAD
        "other_to_mid": ...,    # OTHER real → predicho MID
        "within_head": ...,     # HEAD real → predicho HEAD (correcto)
        "within_mid": ...,      # MID real → predicho MID (correcto)
        "within_other": ...,    # OTHER real → predicho OTHER (correcto)
    }
```

Esto va alineado con `RemappedMetricsTracker.compute_other_diagnostics()`.

---

## Advertencias de Interpretación

### ⚠️ HEAD con solo 3 clases
```
HEAD: 3 clases, 39 muestras
```
- Una mejora fuerte en HEAD puede **dominar métricas por bucket**
- Pero **no necesariamente indica generalización real**
- Interpretar con cautela en conclusiones

### ⚠️ Accuracy puede subir "falsamente"
Si el modelo aprende a predecir mucho OTHER:
- Accuracy global sube (porque OTHER es ~50% del val set)
- Pero no está aprendiendo semántica útil

**Solución**: Siempre revisar:
- `pct_predictions_other` < 60% para considerar válido
- `entropy_of_predictions` > umbral mínimo

### ⚠️ F1 macro vs F1 weighted
- **F1 macro**: Cada clase pesa igual → puede ser engañoso con OTHER grande
- **F1 weighted**: Ponderado por soporte → refleja impacto real

**Reportar ambos** y dejar claro cuál usar para cada interpretación.

---

## Timeline Estimado

| Fase | Duración | Entregable |
|------|----------|------------|
| 1. Módulo Remapping | 30 min | `training/remapping.py` |
| 2. Dataset Wrapper | 15 min | `RemappedDataset` |
| 3. Script Training | 15 min | `--tail-to-other` flag |
| 4. Métricas | 20 min | `remapped_metrics.py` |
| 5. Tests | 20 min | `test_remapping.py` |
| 6. Ejecución | 10 min | Resultados |
| **Total** | **~2 horas** | Experimento completo |

---

## Notas Técnicas

### Determinismo
Para reproducibilidad exacta:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### Compatibilidad Collate
El collate existente funciona porque:
- Solo cambia `gloss_id` (entero)
- Shape de keypoints no cambia
- Padding igual

### Métricas Justas
Para comparación justa:
- Usar mismos epochs (5)
- Misma learning rate
- Mismo batch size
- Mismo split estratificado
