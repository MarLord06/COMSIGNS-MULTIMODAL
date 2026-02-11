# 🤟 COMSIGNS — Sistema de Interpretación de Lengua de Señas

> Sistema experto para interpretación de Lengua de Señas Ecuatoriana (LSEC) mediante procesamiento multimodal de video.

---

## 📖 Navegación

| Documento | Descripción |
|-----------|-------------|
| [🤟 README Principal](../README.md) | Índice general del proyecto |
| [🧠 Arquitectura del Modelo](../docs/MODEL_ARCHITECTURE.md) | Encoder, ramas, fusión, clasificador |
| [📘 Documentación Técnica](../docs/MODEL_TECHNICAL.md) | I/O, módulos internos, limitaciones |
| [🏋️ Entrenamiento](../docs/TRAINING.md) | Trainer, métricas, checkpointing |
| [👤 Guía de Usuario](../docs/USER_GUIDE.md) | Instalación, inferencia, troubleshooting |
| [🚀 Inicio Rápido](QUICKSTART.md) | Setup mínimo para empezar |
| [🔧 Setup MediaPipe](MODELS_SETUP.md) | Descarga de modelos |

### Documentación Técnica

| Documento | Descripción |
|-----------|-------------|
| [🏗️ Arquitectura General](docs/ARCHITECTURE.md) | Pipeline, dataset AEC, resultados Phase 1 |
| [📜 Referencia de Scripts](docs/SCRIPTS_USAGE.md) | CLI flags y uso |
| [🌐 Inferencia Web](docs/WEB_INFERENCE.md) | API REST + Frontend |
| [⚙️ Servicios](../services/SERVICES_TECH_DOC.md) | Docs técnicos de services/ |

---

## Arquitectura

El sistema está compuesto por los siguientes servicios:

| Servicio | Descripción | README |
|----------|-------------|--------|
| **ingestion/** | Captura y procesamiento de video | [README](services/ingestion/README.md) |
| **preprocessing/** | Extracción de keypoints con MediaPipe | [README](services/preprocessing/README.md) |
| **encoder/** | Encoder multimodal (manos, cuerpo, rostro) | [README](services/encoder/README.md) |
| **glosador/** | Conversión de embeddings a glosas | — |
| **translator/** | Traducción de glosas a español | — |
| **api/** | API FastAPI para inferencia | [README](services/api/README.md) |

> [!TIP]
> Para diagramas detallados del encoder, ver [MODEL_ARCHITECTURE.md](../docs/MODEL_ARCHITECTURE.md).

---

## Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar en modo desarrollo (recomendado)
pip install -e .
```

---

## Uso

### Desarrollo Local

```bash
# Opción 1: Usar el script run_api.py (recomendado)
python3 run_api.py

# Opción 2: Usar uvicorn directamente
python3 -m uvicorn comsigns.services.api.main:app --reload

# Opción 3: Usar Makefile
make run-api

# Ejecutar servicios individuales
python -m comsigns.services.ingestion.capture
python -m comsigns.services.preprocessing.process_clip
```

### Docker

```bash
docker-compose up -d
```

---

## Estructura del Proyecto

```
comsigns/
├── core/data/          # Datasets y loaders
│   └── datasets/aec/   # Dataset AEC
├── services/           # Servicios del pipeline
│   ├── api/            # FastAPI
│   ├── encoder/        # MultimodalEncoder
│   ├── ingestion/      # Captura de video
│   ├── preprocessing/  # MediaPipe keypoints
│   └── inference/      # Predictor y loader
├── training/           # Trainer, Classifier, Metrics
├── scripts/            # Scripts CLI
├── experiments/        # Modelos entrenados
├── docs/               # Documentación interna
└── tests/              # Tests unitarios
```

---

## Configuración

Editar `config.yaml` para ajustar parámetros. Ver [QUICKSTART.md](QUICKSTART.md) para detalles.

## Tests

```bash
pytest tests/
```

---

## 📚 Documentos Relacionados

- [🧠 Arquitectura del Modelo](../docs/MODEL_ARCHITECTURE.md) — Diagramas detallados
- [🏗️ Arquitectura General](docs/ARCHITECTURE.md) — Pipeline completo + resultados
- [📜 Scripts](docs/SCRIPTS_USAGE.md) — Todos los scripts CLI
- [🌐 Inferencia Web](docs/WEB_INFERENCE.md) — API + Frontend

## Licencia

MIT
