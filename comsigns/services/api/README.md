# API FastAPI

API REST para inferencia de video en tiempo real.

## Endpoints

### `GET /`

Información general de la API.

### `GET /health`

Health check.

### `POST /infer/video`

Procesa un video y retorna embeddings.

**Parámetros:**
- `file`: Archivo de video (multipart/form-data)
- `fps`: FPS objetivo (opcional)
- `normalize`: Normalizar keypoints (opcional, default: true)

**Respuesta:**
```json
{
  "clip_id": "uuid",
  "status": "success",
  "fps": 30.0,
  "num_frames": 100,
  "embedding_shape": [1, 100, 512],
  "embeddings": [...],
  "metadata": {...}
}
```

## Uso

### Ejecutar servidor

```bash
uvicorn comsigns.services.api.main:app --reload
```

### Con Docker

```bash
docker-compose up api
```

## Configuración

Editar `config.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  max_upload_size: 100  # MB
  timeout: 300  # segundos
```

