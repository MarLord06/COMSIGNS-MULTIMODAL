# 🤟 ComSigns — Reconocimiento de Lengua de Señas

> Sistema multimodal de reconocimiento de lengua de señas basado en keypoints, encoder LSTM y clasificación temporal.

---

## 📖 Documentación

### 🔬 Modelo y Arquitectura

| Documento | Descripción |
|-----------|-------------|
| [🧠 Arquitectura del Modelo](docs/MODEL_ARCHITECTURE.md) | Diagrama del encoder, ramas, fusión y clasificador |
| [📘 Documentación Técnica](docs/MODEL_TECHNICAL.md) | Formato de I/O, inferencia, módulos internos, limitaciones |
| [🏋️ Módulo de Entrenamiento](docs/TRAINING.md) | Trainer, métricas, checkpointing, augmentation |

### 🛠️ Uso y Setup

| Documento | Descripción |
|-----------|-------------|
| [👤 Guía de Usuario](docs/USER_GUIDE.md) | Instalación, inferencia de video, troubleshooting |
| [🚀 Inicio Rápido](comsigns/QUICKSTART.md) | Setup mínimo para empezar |
| [🔧 Setup de MediaPipe](comsigns/MODELS_SETUP.md) | Descarga y configuración de modelos MediaPipe |
| [📋 Guía Completa de Inicio](INICIO.md) | Guía detallada con todos los pasos |

### 🏗️ Arquitectura del Sistema

| Documento | Descripción |
|-----------|-------------|
| [🏗️ Arquitectura General](comsigns/docs/ARCHITECTURE.md) | Pipeline, dataset AEC, resultados Phase 1 |
| [⚙️ Servicios Técnicos](services/SERVICES_TECH_DOC.md) | Preprocessing, encoder, inference, API |
| [🌐 Inferencia Web](comsigns/docs/WEB_INFERENCE.md) | API REST + Frontend React |
| [📜 Referencia de Scripts](comsigns/docs/SCRIPTS_USAGE.md) | CLI flags y uso de scripts |

---

## 📂 Estructura del Proyecto

```
COMSIGNS-MULTIMODAL/
├── comsigns/                    # Código fuente principal
│   ├── core/data/               # Datasets y loaders
│   ├── services/                # Servicios del pipeline
│   │   ├── preprocessing/       # Extracción de keypoints (MediaPipe)
│   │   ├── encoder/             # Encoder multimodal (PyTorch)
│   │   ├── inference/           # Carga de checkpoints y predicción
│   │   └── api/                 # API FastAPI (REST + WebSocket)
│   ├── training/                # Clasificador, trainer, métricas
│   ├── scripts/                 # Scripts CLI (inferencia, entrenamiento)
│   └── experiments/             # Checkpoints y resultados
├── data/                        # Datasets (raw, splits)
├── models/                      # Modelos MediaPipe (.task)
├── docs/                        # Documentación técnica
│   ├── MODEL_ARCHITECTURE.md    # Arquitectura del modelo
│   ├── MODEL_TECHNICAL.md       # Documentación técnica detallada
│   ├── TRAINING.md              # Módulo de entrenamiento
│   └── USER_GUIDE.md            # Guía de usuario
└── services/                    # Documentación de servicios
```

---

## ⚡ Ejemplo Rápido — Inferencia de Video

```bash
python3 comsigns/scripts/infer_video.py \
  --video <RUTA_VIDEO> \
  --model comsigns/experiments/micro_v1_retrained/best.pt \
  --mapping comsigns/experiments/micro_v1_retrained/class_mapping.json \
  --device cpu
```

> [!TIP]
> Consulta la [Guía de Usuario](docs/USER_GUIDE.md) para más detalles y la [Referencia de Scripts](comsigns/docs/SCRIPTS_USAGE.md) para ver todos los flags disponibles.

---

## 📚 READMEs por Módulo

| Servicio | README |
|----------|--------|
| API | [services/api/README.md](comsigns/services/api/README.md) |
| Encoder | [services/encoder/README.md](comsigns/services/encoder/README.md) |
| Ingestion | [services/ingestion/README.md](comsigns/services/ingestion/README.md) |
| Preprocessing | [services/preprocessing/README.md](comsigns/services/preprocessing/README.md) |

---

## 📝 Notas

- Consulta [MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md) para entender la arquitectura interna del encoder multimodal.
- Consulta [USER_GUIDE.md](docs/USER_GUIDE.md) para pasos de instalación, preprocesado y troubleshooting.
- Este README es un índice; para cambios en la documentación edita los archivos en [`docs/`](docs/) y [`comsigns/docs/`](comsigns/docs/).
