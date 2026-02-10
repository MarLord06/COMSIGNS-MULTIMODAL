# ROADMAP: Clean, Rebalance & Retrain Pipeline

## Objetivo
Eliminar clases `UNKNOWN_*` garantizando resolución semántica completa:
```
new_class_id → class_mapping.json (new → old) → dict.json (old → gloss) → respuesta estable
```

---

## ✅ Fase 1: Semantic Closure (COMPLETADO)

### Módulo: `training/semantic_closure.py`

| Componente | Estado | Descripción |
|------------|--------|-------------|
| `load_dict_mapping()` | ✅ | Carga dict.json → mapping old_id → gloss |
| `build_semantic_whitelist()` | ✅ | Whitelist estricta de clases válidas |
| `SemanticClosureReport` | ✅ | Reporte de clases válidas vs inválidas |
| `DictIdDataset` | ✅ | Dataset usando old_class_id de dict.json |
| `FilteredGlossDataset` | ✅ | Filtra solo glosses semánticamente válidos |
| `LowSupportFilter` | ✅ | Elimina clases con < N samples |

### CLI Arguments Agregados:
- `--semantic-closure` - Habilitar filtrado semántico
- `--class-mapping PATH` - Ruta a class_mapping.json
- `--use-dict-ids` - Usar IDs directos de dict.json
- `--min-class-samples N` - Mínimo de samples por clase
- `--remove-low-support` - Eliminar clases bajo el mínimo

---

## ✅ Fase 2: Class Consolidation (COMPLETADO)

### Estrategia TAIL → OTHER

| Componente | Estado | Descripción |
|------------|--------|-------------|
| `ClassRemapper` | ✅ (existente) | Remapea TAIL → OTHER |
| `RemappedDataset` | ✅ (existente) | Wrapper on-the-fly |
| Integración con semantic closure | ✅ | Filtrado antes de remapping |

### Buckets:
- **HEAD**: ≥10 samples (configurable via `--head-threshold`)
- **MID**: 3-9 samples
- **TAIL**: <3 samples → consolidar en OTHER

---

## ✅ Fase 3: Dataset Rebalancing (COMPLETADO)

### Módulo: `training/rebalance.py`

| Componente | Estado | Descripción |
|------------|--------|-------------|
| `RebalanceConfig` | ✅ | Configuración de rebalanceo |
| `RebalancedDataset` | ✅ | Dataset con down/up-sampling |
| Down-sample OTHER | ✅ | Limitar OTHER a N × median |
| Up-sample valid classes | ✅ | Aumentar clases minoritarias |

### Módulo: `training/augmentation.py`

| Componente | Estado | Descripción |
|------------|--------|-------------|
| `AugmentConfig` | ✅ | Configuración de augmentación |
| `KeypointAugmenter` | ✅ | Aplicar augmentaciones |
| `_add_noise()` | ✅ | Ruido gaussiano en keypoints |
| `_time_shift()` | ✅ | Jitter temporal (desplazar frames) |
| `_mirror()` | ✅ | Espejado horizontal |

### CLI Arguments Agregados:
- `--rebalance` - Habilitar rebalanceo
- `--other-max-multiplier N` - OTHER máximo = N × median
- `--other-max-cap N` - Cap absoluto para OTHER
- `--augment` - Habilitar augmentación
- `--augment-noise-std F` - Desviación std del ruido
- `--augment-time-shift N` - Frames máx de shift temporal
- `--augment-mirror-prob F` - Probabilidad de espejado

---

## ✅ Fase 4: Training Configuration (COMPLETADO)

### Class-Weighted CrossEntropyLoss

| Componente | Estado | Descripción |
|------------|--------|-------------|
| Compute class weights | ✅ | Pesos inversamente proporcionales al soporte |
| OTHER penalty | ✅ | Multiplicador extra para penalizar OTHER |
| Pasar loss_fn a Trainer | ✅ | Integrado en train.py |

### CLI Arguments Agregados:
- `--class-weighting` - Usar pérdida ponderada
- `--other-penalty F` - Multiplicador de penalización OTHER

---

## ✅ Fase 5: Artifact Export (COMPLETADO)

### Archivos Generados

| Archivo | Estado | Descripción |
|---------|--------|-------------|
| `best.pt` | ✅ | Mejor checkpoint del modelo |
| `class_mapping.json` | ✅ | Mapeo new_id → old_id |
| `class_index.json` | ✅ | Mapeo new_id → gloss |
| `training_state.json` | ✅ | Estado completo del entrenamiento |
| `training_report.md` | ✅ | Reporte markdown legible |
| `learned_words_report.json` | ✅ | Análisis de palabras aprendidas |

---

## 🔲 Fase 6: Evaluation (PENDIENTE)

### Métricas Requeridas

| Métrica | Estado | Descripción |
|---------|--------|-------------|
| Top-1 accuracy (overall) | 🔲 | Accuracy general |
| Top-1 accuracy (excl. OTHER) | 🔲 | Accuracy sin contar OTHER |
| Top-3 accuracy | 🔲 | Entre las 3 predicciones |
| Confusion matrix MID vs MID | 🔲 | Análisis de confusiones |
| Test palabras problemáticas | 🔲 | "a", "faltar", etc. |

---

## 🔲 Fase 7: Integration Tests (PENDIENTE)

### Tests a Crear

| Test | Estado | Ubicación |
|------|--------|-----------|
| `test_semantic_closure.py` | 🔲 | `tests/unit/` |
| `test_augmentation.py` | 🔲 | `tests/unit/` |
| `test_rebalance.py` | 🔲 | `tests/unit/` |
| Integration test full pipeline | 🔲 | `tests/integration/` |

---

## Uso del Pipeline Completo

```bash
cd comsigns

# Entrenamiento con todas las fases habilitadas
python -m scripts.train \
    --stratified data/splits/aec_stratified.json \
    --semantic-closure \
    --class-mapping experiments/latest/class_mapping.json \
    --min-class-samples 3 \
    --remove-low-support \
    --tail-to-other \
    --head-threshold 10 \
    --rebalance \
    --other-max-multiplier 2.0 \
    --augment \
    --augment-noise-std 0.01 \
    --augment-time-shift 2 \
    --augment-mirror-prob 0.3 \
    --class-weighting \
    --other-penalty 1.5 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --eval
```

---

## Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATASET                               │
│                    (AECDataset, 505 glosses)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 1: SEMANTIC CLOSURE                           │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   dict.json      │ +  │ class_mapping    │ → WHITELIST       │
│  │  (old_id→gloss)  │    │   (new→old)      │                   │
│  └──────────────────┘    └──────────────────┘                   │
│                                │                                 │
│                    FilteredGlossDataset                         │
│                    LowSupportFilter                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: CLASS CONSOLIDATION                        │
│                                                                  │
│   HEAD (≥10)  │  MID (3-9)  │  TAIL (<3) → OTHER                │
│                                                                  │
│                      ClassRemapper                               │
│                      RemappedDataset                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3: REBALANCING                                │
│                                                                  │
│  ┌────────────────┐         ┌────────────────┐                  │
│  │ DOWN-SAMPLE    │         │ UP-SAMPLE      │                  │
│  │ OTHER class    │         │ Valid classes  │                  │
│  │ (max 2×median) │         │ (augmentation) │                  │
│  └────────────────┘         └────────────────┘                  │
│                                                                  │
│              RebalancedDataset + KeypointAugmenter              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 4: TRAINING                                   │
│                                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │ Class-Weighted CrossEntropyLoss          │                   │
│  │ weight[i] = N / (n_classes × support[i]) │                   │
│  │ weight[OTHER] *= other_penalty           │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│                        Trainer                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 5: ARTIFACTS                                  │
│                                                                  │
│  ├── best.pt                 (modelo)                           │
│  ├── class_mapping.json      (new→old mapping)                  │
│  ├── class_index.json        (new→gloss mapping)                │
│  ├── training_state.json     (configuración completa)           │
│  └── training_report.md      (reporte legible)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Garantías de Compatibilidad

1. **Backend**: `class_mapping.json` mantiene formato existente
2. **Frontend**: No requiere cambios
3. **Inference**: `class_index.json` permite resolución directa new_id → gloss
4. **Zero UNKNOWN_***: Todas las clases tienen gloss resuelto

---

## Próximos Pasos

1. [ ] Ejecutar tests unitarios
2. [ ] Entrenar modelo con pipeline completo
3. [ ] Evaluar métricas de Phase 6
4. [ ] Integrar modelo entrenado con backend
5. [ ] Verificar que no hay UNKNOWN_* en producción
