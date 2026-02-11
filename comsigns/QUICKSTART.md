# 🚀 Guía de Inicio Rápido — COMSIGNS

> Setup mínimo para empezar a usar el sistema.

---

## 📖 Navegación

| Documento | Descripción |
|-----------|-------------|
| [📋 Guía Completa](../INICIO.md) | Guía detallada con todos los pasos |
| [🧠 Arquitectura del Modelo](../docs/MODEL_ARCHITECTURE.md) | Encoder multimodal y clasificador |
| [🔧 Setup MediaPipe](MODELS_SETUP.md) | Descarga de modelos MediaPipe |
| [📜 Referencia de Scripts](docs/SCRIPTS_USAGE.md) | CLI flags y uso |
| [🏗️ Arquitectura General](docs/ARCHITECTURE.md) | Pipeline + dataset + resultados |

---

## Instalación

```bash
# 1. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar estructura de directorios
make setup
```

---

## Uso Básico

### 1. Procesar un video

```python
from comsigns.services.preprocessing import process_video_clip

feature_clip = process_video_clip("video.mp4", fps=30.0)
print(f"Frames procesados: {len(feature_clip.frames)}")
```

> [!TIP]
> Ver [preprocessing/README.md](services/preprocessing/README.md) para más opciones.

### 2. Usar el encoder

```python
from comsigns.services.encoder import create_encoder, feature_clip_to_tensors
import torch

encoder = create_encoder()
tensors = feature_clip_to_tensors(feature_clip)

with torch.no_grad():
    embeddings = encoder(
        tensors['hand'].unsqueeze(0),
        tensors['body'].unsqueeze(0),
        tensors['face'].unsqueeze(0)
    )
```

> [!TIP]
> Ver [encoder/README.md](services/encoder/README.md) para configuración y [MODEL_ARCHITECTURE.md](../docs/MODEL_ARCHITECTURE.md) para entender la arquitectura.

### 3. Instalar el paquete

```bash
# Desde el directorio comsigns
pip install -e .
```

### 4. Ejecutar la API

```bash
# Opción 1: Usar el script run_api.py (recomendado)
python3 run_api.py

# Opción 2: Con Makefile
make run-api

# Opción 3: Con uvicorn directamente
python3 -m uvicorn comsigns.services.api.main:app --reload
```

> [!TIP]
> Ver [api/README.md](services/api/README.md) para endpoints disponibles.

### 5. Ejecutar la UI Web

```bash
cd web
npm install
npm run dev
```

Luego abrir `http://localhost:3000`

> [!NOTE]
> Ver [WEB_INFERENCE.md](docs/WEB_INFERENCE.md) para detalles de la inferencia web.

---

## Tests

```bash
# Ejecutar todos los tests
make test

# Con cobertura
make test-cov
```

---

## Docker

```bash
# Construir imágenes
make docker-build

# Iniciar servicios
make docker-up

# Detener servicios
make docker-down
```

---

## Estructura del Proyecto

```
comsigns/
├── services/           # Módulos principales
│   ├── ingestion/      # Captura de video
│   ├── preprocessing/  # Extracción de keypoints
│   ├── encoder/        # Encoder multimodal
│   └── api/            # API FastAPI
├── training/           # Entrenamiento y clasificador
├── scripts/            # Scripts CLI
├── tests/              # Tests unitarios
└── experiments/        # Checkpoints y resultados
```

---

## Próximos Pasos

1. **Glosador** — Implementar módulo embeddings → glosas
2. **Translator** — Implementar traducción glosas → español
3. **Feature Store** — Sistema de almacenamiento de features
4. **Entrenamiento** — Scripts para entrenar modelos · ver [TRAINING.md](../docs/TRAINING.md)

---

## 📚 Documentación por Servicio

| Servicio | README |
|----------|--------|
| Ingestion | [services/ingestion/README.md](services/ingestion/README.md) |
| Preprocessing | [services/preprocessing/README.md](services/preprocessing/README.md) |
| Encoder | [services/encoder/README.md](services/encoder/README.md) |
| API | [services/api/README.md](services/api/README.md) |
