# 🌐 API FastAPI

> API REST para inferencia de video en tiempo real.

---

## 📖 Ver También

| Documento | Descripción |
|-----------|-------------|
| [🌐 Inferencia Web](../../docs/WEB_INFERENCE.md) | Guía completa API + Frontend |
| [⚙️ Servicios](../../../services/SERVICES_TECH_DOC.md) | Docs técnicos detallados |
| [🧠 Arquitectura del Modelo](../../../docs/MODEL_ARCHITECTURE.md) | Encoder y clasificador |
| [📜 Scripts](../../docs/SCRIPTS_USAGE.md) | CLI de inferencia alternativa |

---

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

---

## Uso

### Ejecutar servidor

```bash
uvicorn comsigns.services.api.main:app --reload
```

### Con Docker

```bash
docker-compose up api
```

---

## Configuración

Editar `config.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  max_upload_size: 100  # MB
  timeout: 300  # segundos
```

---

## 📚 Docs Relacionados

- [🌐 Inferencia Web](../../docs/WEB_INFERENCE.md) — Guía completa
- [⚙️ Servicios](../../../services/SERVICES_TECH_DOC.md) — Docs técnicos
