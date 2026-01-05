# Guía de Inicio Rápido - COMSIGNS

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

## Uso Básico

### 1. Procesar un video

```python
from comsigns.services.preprocessing import process_video_clip

feature_clip = process_video_clip("video.mp4", fps=30.0)
print(f"Frames procesados: {len(feature_clip.frames)}")
```

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

### 3. Instalar el paquete (importante)

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

### 5. Ejecutar la UI Web

```bash
cd web
npm install
npm run dev
```

Luego abrir `http://localhost:3000`

## Tests

```bash
# Ejecutar todos los tests
make test

# Con cobertura
make test-cov
```

## Docker

```bash
# Construir imágenes
make docker-build

# Iniciar servicios
make docker-up

# Detener servicios
make docker-down
```

## Estructura del Proyecto

```
comsigns/
├── services/          # Módulos principales
│   ├── ingestion/     # Captura de video
│   ├── preprocessing/  # Extracción de keypoints
│   ├── encoder/        # Encoder multimodal
│   └── api/            # API FastAPI
├── web/                # Interfaz React
├── tests/              # Tests unitarios
└── examples/           # Ejemplos de uso
```

## Próximos Pasos

1. **Glosador**: Implementar módulo para convertir embeddings → glosas
2. **Translator**: Implementar traducción de glosas → español
3. **Feature Store**: Sistema de almacenamiento de features
4. **Entrenamiento**: Scripts para entrenar modelos

## Documentación

Ver README.md en cada módulo para más detalles:
- `services/ingestion/README.md`
- `services/preprocessing/README.md`
- `services/encoder/README.md`
- `services/api/README.md`

