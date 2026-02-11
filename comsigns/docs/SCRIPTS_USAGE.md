# 📜 COMSIGNS — Referencia de Scripts

> Lista de scripts en [`scripts/`](../scripts/) con sus flags CLI y ejemplos de uso.

---

## 📖 Navegación

| Documento | Descripción |
|-----------|-------------|
| [🏗️ Arquitectura General](ARCHITECTURE.md) | Pipeline y resultados |
| [🧠 Arquitectura del Modelo](../../docs/MODEL_ARCHITECTURE.md) | Encoder y clasificador |
| [📘 Documentación Técnica](../../docs/MODEL_TECHNICAL.md) | I/O y módulos |
| [🏋️ Entrenamiento](../../docs/TRAINING.md) | Trainer y métricas |
| [🌐 Inferencia Web](WEB_INFERENCE.md) | API REST + Frontend |

---

## `scripts/infer.py` — Inferencia sobre sample preprocesado

Ejecuta inferencia sobre un archivo `.pkl`, `.pt` o `.json` preprocesado.

| Flag | Tipo | Descripción | Default |
|------|------|-------------|---------|
| `--checkpoint, -c` | Path | **Requerido.** Path al checkpoint (`.pt`) | — |
| `--class-mapping, -m` | Path | Path a `class_mapping.json` | — |
| `--input, -i` | Path | **Requerido.** Input file (`.pkl`, `.pt`, `.json`) | — |
| `--topk, -k` | int | Número de top predicciones | 5 |
| `--device, -d` | str | Device (`cpu`, `cuda`, `mps`) | auto |
| `--json` | flag | Output como JSON | — |
| `--verbose, -v` | flag | Salida detallada | — |

```bash
python scripts/infer.py \
  --checkpoint experiments/micro_v1/checkpoints/best.pt \
  --input samples/example.pkl \
  --topk 5
```

---

## `scripts/infer_video.py` — Inferencia desde video

Procesa un video completo: extrae keypoints, codifica y clasifica.

| Flag | Tipo | Descripción | Default |
|------|------|-------------|---------|
| `--video, -v` | Path | **Requerido.** Path al video | — |
| `--model` | Path | Path al checkpoint | `experiments/micro_v1/best.pt` |
| `--mapping` | Path | Class mapping JSON | `experiments/micro_v1/class_mapping.json` |
| `--device` | str | Device (`cpu`, `cuda`) | auto |
| `--topk` | int | Top predicciones | 5 |

```bash
python scripts/infer_video.py \
  --video data/raw/.../comer_1001.mp4 \
  --topk 3
```

> [!TIP]
> Ver [Guía de Usuario](../../docs/USER_GUIDE.md#ejecutar-inferencia-de-video) para ejemplos completos.

---

## `scripts/test_micro_model.py` — Validación rápida

| Flag | Tipo | Descripción | Default |
|------|------|-------------|---------|
| `--samples` | int | Número de samples a testear | 10 |
| `--model` | Path | Checkpoint | `experiments/micro_v1/best.pt` |
| `--mapping` | Path | Class mapping JSON | `experiments/micro_v1/class_mapping.json` |
| `--device` | str | Device | `cpu` |

---

## Scripts de Entrenamiento

`scripts/train_v1.py` · `scripts/train.py` · `scripts/train_micro.py`

Flags comunes:

| Flag | Tipo | Descripción |
|------|------|-------------|
| `--epochs` | int | Número de épocas |
| `--batch-size` | int | Tamaño de batch |
| `--lr` | float | Learning rate |
| `--device` | str | `auto`\|`cuda`\|`mps`\|`cpu` |
| `--seed` | int | Semilla de reproducibilidad |
| `--min-support` | int | Soporte mínimo por clase |
| `--augment` / `--no-augment` | flag | Data augmentation |
| `--class-weighting` / `--no-class-weighting` | flag | Pesos por clase |
| `--dropout` | float | Dropout rate |
| `--weight-decay` | float | Weight decay |
| `--label-smoothing` | float | Label smoothing |
| `--lr-scheduler` | str | `none`\|`plateau`\|`cosine` |
| `--output-dir` | Path | Directorio de salida |
| `--eval` | flag | Ejecutar evaluación final |

> [!TIP]
> Ver [TRAINING.md](../../docs/TRAINING.md) para documentación completa del módulo de entrenamiento.

---

## `scripts/extract_samples.py` — Exportar samples para inferencia

Extrae samples `.pkl` listos para inferencia directa.

Flags típicos: `--dataset-root`, `--split-file`, `--out-dir`, `--limit`, `--verbose`.

---

## Scripts de Análisis de Datos

`scripts/analyze_dataset.py` · `scripts/extract_*`

Flags típicos: `--dataset-root`, `--split-file`, `--out-dir`, `--limit`, `--verbose`.

---

## `scripts/test_e2e_inference.py` — Test end-to-end

Usa los servicios del backend para una prueba de humo completa. Sin flags CLI; ejecutar directamente.

---

## Ejecutar Scripts de Forma Confiable

- **Working directory recomendado:** raíz del repositorio
- Si hay errores de import:

```bash
PYTHONPATH=. python scripts/<script>.py ...
```

---

## 📚 Documentos Relacionados

- [🏗️ Arquitectura](ARCHITECTURE.md) — Pipeline y dataset
- [🧠 Modelo](../../docs/MODEL_ARCHITECTURE.md) — Arquitectura del encoder
- [🏋️ Entrenamiento](../../docs/TRAINING.md) — Trainer, métricas
- [🌐 Inferencia Web](WEB_INFERENCE.md) — API REST