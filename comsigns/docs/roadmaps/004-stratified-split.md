# 🗺️ Roadmap: Split Estratificado por Glosa (Dataset AEC)

> **Fecha**: 20 de enero de 2026  
> **Estado**: ✅ Completado  
> **Módulo**: `comsigns.core.data.datasets.aec` + `scripts`

---

## 📋 Contexto del Problema

El dataset AEC (Lengua de Señas Peruana) presenta limitaciones estructurales:

| Campo | Valor | ¿Útil para split? |
|-------|-------|-------------------|
| `signer_id` | Siempre `-1` | ❌ No válido |
| `source_video_name` | Solo 2 valores | ❌ No estadísticamente viable |
| `gloss` | 505 valores únicos | ✅ Útil para estratificación |

**Problema**: El split actual usa `random_split` que puede desbalancear las clases.

**Solución**: Implementar split estratificado por glosa para:
- Mantener distribución de clases consistente entre train/val
- Asegurar reproducibilidad
- Separar lógica de split del Dataset (Clean Architecture)

### ⚠️ Justificación Técnica Importante

> "El dataset AEC contiene únicamente dos videos fuente y no incluye información válida 
> de signer. Por ello, se utiliza un split estratificado por glosa con fines de validación 
> técnica del pipeline, **no como evaluación de generalización real**."

---

## 🎯 Objetivos

| ID | Objetivo | Prioridad |
|----|----------|-----------|
| O1 | Crear script de generación de split estratificado | Alta |
| O2 | Generar archivo `data/splits/aec_stratified.json` | Alta |
| O3 | Modificar AECDataset para leer split externo | Alta |
| O4 | Tests de validación de split | Alta |
| O5 | Mantener backward compatibility (sin split_file) | Media |

### No-Objetivos (explícitamente excluidos)

- ❌ Cross-validation
- ❌ Split por video/signer
- ❌ Sampling avanzado o class weighting
- ❌ Test set (solo train/val)
- ❌ Generar split dentro del Dataset

---

## 🏗️ Diseño Propuesto

### Arquitectura de Archivos

```
comsigns/
├── scripts/
│   └── generate_aec_split.py       # NUEVO: genera split estratificado
├── core/data/datasets/aec/
│   └── aec_dataset.py              # MODIFICAR: soporte split externo
└── tests/unit/
    └── test_stratified_split.py    # NUEVO: tests de split

data/
└── splits/
    └── aec_stratified.json         # NUEVO: archivo de split generado
```

### Formato del Archivo de Split

```json
{
  "metadata": {
    "created_at": "2026-01-20T10:00:00",
    "seed": 42,
    "train_ratio": 0.8,
    "total_instances": 2308,
    "total_glosses": 505,
    "strategy": "stratified_by_gloss"
  },
  "train": ["yo_1", "yo_14", "comer_12", ...],
  "val": ["yo_50", "comer_3", ...]
}
```

### Flujo de Datos

```
┌────────────────┐                      ┌────────────────────────┐
│   dict.json    │ ─── parse ────────▶  │ instances_by_gloss     │
│  (original)    │                      │ {gloss: [unique_names]}│
└────────────────┘                      └────────────────────────┘
                                                  │
                                     stratified_split(80/20)
                                                  │
                                                  ▼
                                        ┌────────────────────────┐
                                        │  aec_stratified.json   │
                                        │  train: [...], val:[..]│
                                        └────────────────────────┘
                                                  │
                                                  ▼
┌────────────────┐     filter_by_split   ┌────────────────────────┐
│  AECDataset    │ ◀──────────────────── │ train_set / val_set    │
│ (with split)   │                       │                        │
└────────────────┘                       └────────────────────────┘
```

---

## 📝 Especificaciones de Implementación

### 1. Script `generate_aec_split.py`

```python
def generate_stratified_split(
    dict_path: Path,
    output_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Dict[str, List[str]]
```

**Pseudo-lógica**:
```python
random.seed(seed)
instances_by_gloss = group_instances_by_gloss(dict_data)

train_names, val_names = [], []

for gloss, instances in instances_by_gloss.items():
    unique_names = [inst['unique_name'] for inst in instances]
    random.shuffle(unique_names)
    
    # Glosas con 1 instancia van a train
    if len(unique_names) == 1:
        train_names.extend(unique_names)
    else:
        split_idx = max(1, int(train_ratio * len(unique_names)))
        train_names.extend(unique_names[:split_idx])
        val_names.extend(unique_names[split_idx:])
```

### 2. Modificación de `AECDataset.__init__`

```python
def __init__(
    self,
    dataset_root: Path,
    dict_path: Optional[Path] = None,
    split_file: Optional[Path] = None,     # NUEVO
    split: Literal["train", "val"] = None,  # NUEVO
    ...
):
    ...
    # Después de _flatten_instances()
    if split_file is not None:
        self._apply_split(split_file, split or "train")
```

**Nuevo método**:
```python
def _apply_split(self, split_file: Path, split: str) -> None:
    """Filter instances based on external split file."""
    with open(split_file) as f:
        split_data = json.load(f)
    
    valid_names = set(split_data[split])
    original_count = len(self._flat_instances)
    
    self._flat_instances = [
        entry for entry in self._flat_instances
        if entry['instance'].get('unique_name') in valid_names
    ]
    
    logger.info(f"Applied {split} split: {original_count} → {len(self)} samples")
```

---

## 🧪 Tests Requeridos

| Test | Descripción |
|------|-------------|
| `test_split_has_no_overlap` | `len(train ∩ val) == 0` |
| `test_split_preserves_all_instances` | `len(train) + len(val) == total` |
| `test_dataset_respects_split` | Dataset filtra correctamente |
| `test_stratification_preserves_distribution` | Cada glosa aparece en train |
| `test_single_instance_glosses_in_train` | Glosas con 1 muestra van a train |
| `test_reproducibility_with_seed` | Mismo seed → mismo split |

---

## 📊 Resultado Esperado

**Uso**:
```python
train_ds = AECDataset(
    root_dir,
    split_file=Path("data/splits/aec_stratified.json"),
    split="train"
)
val_ds = AECDataset(
    root_dir,
    split_file=Path("data/splits/aec_stratified.json"),
    split="val"
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
# Train: ~1822, Val: ~456
```

**Output del script**:
```
Generando split estratificado...
  Total glosas: 505
  Total instancias: 2278
  Glosas con 1 instancia: 45 (todas a train)
  Train: 1823 (80.0%)
  Val: 455 (20.0%)
Guardado en: data/splits/aec_stratified.json
```

---

## ✅ Criterios de Aceptación

- [ ] Script de split ejecutable
- [ ] Archivo `aec_stratified.json` generado
- [ ] AECDataset soporta `split_file` y `split` params
- [ ] Tests pasan
- [ ] Trainer funciona sin cambios
- [ ] Backward compatible (sin split_file = comportamiento actual)

---

## 📚 Referencias

- AECDataset actual: `core/data/datasets/aec/aec_dataset.py`
- dict.json schema: Instancias con `unique_name`, `gloss`, `keypoints_path`
- Clean Architecture: Split es responsabilidad externa al Dataset
