# 👤 ComSigns — Guía de Usuario

> Guía práctica para preparar el entorno, ejecutar inferencia de video y solucionar problemas comunes.

---

## 📖 Navegación

| Documento | Descripción |
|-----------|-------------|
| [🤟 README Principal](../README.md) | Índice general del proyecto |
| [🧠 Arquitectura del Modelo](MODEL_ARCHITECTURE.md) | Encoder, ramas, fusión, clasificador |
| [📘 Documentación Técnica](MODEL_TECHNICAL.md) | I/O del modelo, inferencia detallada |
| [🏋️ Entrenamiento](TRAINING.md) | Trainer, métricas, checkpointing |
| [🚀 Inicio Rápido](../comsigns/QUICKSTART.md) | Setup mínimo |
| [🔧 Setup MediaPipe](../comsigns/MODELS_SETUP.md) | Descarga de modelos |
| [📜 Referencia de Scripts](../comsigns/docs/SCRIPTS_USAGE.md) | CLI flags |

---

## Estructura Importante del Repositorio

| Carpeta / Archivo | Descripción |
|-------------------|-------------|
| [`comsigns/`](../comsigns/) | Código principal de la librería y servicios |
| [`comsigns/scripts/infer_video.py`](../comsigns/scripts/infer_video.py) | Script de inferencia por video |
| [`comsigns/experiments/`](../comsigns/experiments/) | Checkpoints y mappings |
| [`comsigns/services/`](../comsigns/services/) | Módulos de encoder, preprocesado |
| [`comsigns/training/`](../comsigns/training/) | Entrenamiento — ver [TRAINING.md](TRAINING.md) |
| [`data/`](../data/) | Datasets y videos de ejemplo |

---

## Requisitos y Entorno

1. **Python 3.9+** (desarrollado con 3.9 en macOS M1)
2. Crear y activar un virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install -r comsigns/requirements.txt
```

> [!NOTE]
> MediaPipe en macOS puede emitir advertencias sobre LibreSSL/OpenSSL; son informativas en muchos casos.

---

## Configuración Mínima

- Si falta `comsigns/config.yaml`, copia la configuración ejemplo:

```bash
cp infra/config.yaml comsigns/config.yaml
```

- Verifica rutas a modelos en [`comsigns/experiments/`](../comsigns/experiments/) y permisos de lectura.
- Para MediaPipe, consulta [MODELS_SETUP.md](../comsigns/MODELS_SETUP.md).

---

## Ejecutar Inferencia de Video

### Comando General

```bash
python3 comsigns/scripts/infer_video.py \
  --video <RUTA_VIDEO> \
  --model comsigns/experiments/micro_v1_retrained/best.pt \
  --mapping comsigns/experiments/micro_v1_retrained/class_mapping.json \
  --device cpu
```

### Ejemplo con Video de Datos

```bash
python3 comsigns/scripts/infer_video.py \
  --video data/raw/lsp_aec/Videos/SEGMENTED_SIGN/ira_alegria/tú_302.mp4 \
  --model comsigns/experiments/micro_v1_retrained/best.pt \
  --mapping comsigns/experiments/micro_v1_retrained/class_mapping.json
```

La salida mostrará shapes de los tensores extraídos y el top-K de predicciones.

> [!TIP]
> Para ver todos los flags disponibles, consulta la [Referencia de Scripts](../comsigns/docs/SCRIPTS_USAGE.md).

---

## Notas sobre Dimensiones y Preprocesado

El encoder multimodal espera estos formatos:

| Modalidad | Shape | Detalle |
|-----------|-------|---------|
| `hand` | `(seq_len, 126)` | 2 manos × 21 keypoints × 3 valores (x,y,z) |
| `body` | `(seq_len, 99)` | 33 keypoints × 3 valores |
| `face` | `(seq_len, 1404)` | 468 keypoints × 3 valores |

El script [`infer_video.py`](../comsigns/scripts/infer_video.py) fue adaptado para producir vectores con 3 valores por keypoint.

> [!IMPORTANT]
> Para diagramas detallados de la arquitectura y todas las shapes, ver [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

---

## Entrenamiento (Resumen Rápido)

Scripts de entrenamiento están bajo [`comsigns/scripts/`](../comsigns/scripts/) y [`comsigns/training/`](../comsigns/training/).

```bash
python3 comsigns/scripts/train.py --config path/to/config.yaml
```

Revisa [`comsigns/experiments/`](../comsigns/experiments/) para ver checkpoints y `class_mapping.json`.

> [!TIP]
> Para documentación completa del entrenamiento, ver [TRAINING.md](TRAINING.md).

---

## Cómo Añadir un Nuevo Video para Pruebas

1. Coloca el video en `data/raw/...` o en una carpeta local
2. Ejecuta el script de inferencia apuntando a esa ruta

---

## Problemas Comunes y Soluciones

### `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Causa:** Desajuste en dimensiones de keypoints.  
**Solución:** Usar la versión actualizada de [`infer_video.py`](../comsigns/scripts/infer_video.py) que usa 3 valores por keypoint.

### Advertencias de MediaPipe/TensorFlow

**Causa:** Advertencias sobre delegados o GL.  
**Solución:** Suelen ser informativas. Si hay errores, verificar versiones de `mediapipe` y `tensorflow`.

### `Archivo de configuración no encontrado`

**Solución:** Crear o copiar `comsigns/config.yaml`:
```bash
cp infra/config.yaml comsigns/config.yaml
```

### Problemas con GPU/CPU

**Solución:** Pasar `--device cpu` para forzar CPU. Para GPU, asegurar CUDA y versión de PyTorch adecuada.

---

## Verificaciones Rápidas

| Verificación | Qué buscar |
|--------------|------------|
| ¿Shapes compatibles? | `Hand shape: (N, 126)`, `Body shape: (N, 99)`, `Face shape: (N, 1404)` |
| ¿Checkpoint cargado? | El script imprimirá el nombre del archivo del modelo |

---

## 📚 Documentos Relacionados

- [🧠 Arquitectura del Modelo](MODEL_ARCHITECTURE.md) — Encoder, shapes, diagramas
- [📘 Documentación Técnica](MODEL_TECHNICAL.md) — Módulos internos, limitaciones
- [🏋️ Entrenamiento](TRAINING.md) — Trainer, checkpointing, métricas
- [🚀 Inicio Rápido](../comsigns/QUICKSTART.md) — Setup mínimo
- [🔧 Setup MediaPipe](../comsigns/MODELS_SETUP.md) — Modelos MediaPipe
- [🌐 Inferencia Web](../comsigns/docs/WEB_INFERENCE.md) — API REST + Frontend
