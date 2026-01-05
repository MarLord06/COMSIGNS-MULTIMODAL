# COMSIGNS - Sistema de Interpretación de Lengua de Señas

Sistema experto para interpretación de Lengua de Señas Ecuatoriana (LSEC) mediante procesamiento multimodal de video.

## Arquitectura

El sistema está compuesto por los siguientes servicios:

- **ingestion/**: Captura y procesamiento de video
- **preprocessing/**: Extracción de keypoints con MediaPipe
- **feature-store/**: Almacenamiento de features para entrenamiento
- **encoder/**: Encoder multimodal (manos, cuerpo, rostro)
- **glosador/**: Conversión de embeddings a glosas
- **translator/**: Traducción de glosas a español
- **api/**: API FastAPI para inferencia
- **web/**: Interfaz React para pruebas y anotación

## Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Instalación del paquete (recomendado)

```bash
# Instalar en modo desarrollo
pip install -e .
```

### Desarrollo local

```bash
# Opción 1: Usar el script run_api.py (recomendado)
python3 run_api.py

# Opción 2: Usar uvicorn directamente (requiere instalación)
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

## Estructura del Proyecto

```
comsigns/
├── infra/              # Configuración de infraestructura
├── data/               # Datos (raw, processed, features)
├── services/           # Servicios del sistema
├── web/                # Interfaz web React
├── experiments/        # Experimentos y prototipos
└── notebooks/          # Jupyter notebooks
```

## Configuración

Editar `config/config.yaml` para ajustar parámetros del sistema.

## Tests

```bash
pytest tests/
```

## Licencia

MIT

