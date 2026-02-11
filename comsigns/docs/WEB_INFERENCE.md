# 🌐 ComSigns — Inferencia Web

> Guía rápida para ejecutar inferencia de lengua de señas vía API REST + Frontend React.

---

## 📖 Navegación

| Documento | Descripción |
|-----------|-------------|
| [🏗️ Arquitectura General](ARCHITECTURE.md) | Pipeline completo |
| [🧠 Arquitectura del Modelo](../../docs/MODEL_ARCHITECTURE.md) | Encoder y clasificador |
| [📜 Referencia de Scripts](SCRIPTS_USAGE.md) | CLI flags |
| [📋 Guía Completa](../../INICIO.md) | Todos los pasos de setup |
| [⚙️ Servicios](../../services/SERVICES_TECH_DOC.md) | Documentación técnica de API |

---

## Arquitectura

```
┌────────────────┐      HTTP/JSON      ┌─────────────────────┐
│   Frontend     │ ◄─────────────────► │   Backend API       │
│   (React)      │                     │   (FastAPI)         │
│   :5173        │                     │   :8000             │
└────────────────┘                     └─────────────────────┘
                                              │
                                              ▼
                                       ┌─────────────────┐
                                       │ InferenceService │
                                       │  + Model (best.pt)
                                       │  + SemanticResolver
                                       └─────────────────┘
```

---

## Requisitos

- Python 3.9+
- Node.js 18+
- Sample `.pkl` files — extraídos con [`scripts/extract_samples.py`](../scripts/extract_samples.py)

---

## 1. Iniciar Backend

```bash
cd comsigns

# Instalar dependencias si es necesario
pip install fastapi uvicorn python-multipart

# Iniciar API
python -m uvicorn backend.api.app:app --reload --port 8000
```

La API estará disponible en `http://localhost:8000`

### Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Info de la API |
| GET | `/health` | Health check |
| GET | `/info` | Info del modelo |
| POST | `/infer` | Inferencia de sample |
| POST | `/infer/batch` | Inferencia batch |

> [!TIP]
> Ver [api/README.md](../services/api/README.md) para documentación completa de la API.

### Probar con curl

```bash
# Health check
curl http://localhost:8000/health

# Info del modelo
curl http://localhost:8000/info

# Inferencia
curl -X POST http://localhost:8000/infer \
  -F "file=@samples/sample_000.pkl" \
  -F "topk=5"
```

---

## 2. Iniciar Frontend

```bash
cd comsigns/web

# Instalar dependencias
npm install

# Iniciar dev server
npm run dev
```

El frontend estará disponible en `http://localhost:5173`

---

## 3. Probar Inferencia Web

1. Abre `http://localhost:5173`
2. Selecciona el modo **"Inferir Sample"** (📦)
3. Arrastra o selecciona un archivo `.pkl`
4. Selecciona el número de top-k predicciones
5. Haz clic en **"Inferir Seña"**

---

## 4. Obtener Samples de Prueba

Si no tienes samples `.pkl`, extráelos del dataset:

```bash
cd comsigns

# Extraer 10 samples aleatorios
python scripts/extract_samples.py \
  --data_dir ../data/raw/lsp_aec \
  --output_dir samples \
  --num_samples 10
```

> [!TIP]
> Ver [SCRIPTS_USAGE.md](SCRIPTS_USAGE.md) para más opciones del script.

---

## Respuesta de la API

```json
{
  "top1": {
    "gloss": "HOLA",
    "confidence": 0.85,
    "bucket": "HEAD",
    "is_other": false,
    "new_class_id": 28,
    "old_class_id": 319
  },
  "topk": [
    { "rank": 1, "gloss": "HOLA", "confidence": 0.85, "bucket": "HEAD" },
    { "rank": 2, "gloss": "TU", "confidence": 0.10, "bucket": "MID" }
  ],
  "meta": {
    "model": "best.pt",
    "num_classes": 142,
    "device": "cpu"
  }
}
```

---

## Configuración

Variables de entorno para personalizar paths:

```bash
export COMSIGNS_CHECKPOINT=experiments/run_XXX/checkpoints/best.pt
export COMSIGNS_CLASS_MAPPING=experiments/run_XXX/class_mapping.json
export COMSIGNS_DICT=../data/raw/lsp_aec/dict.json
export COMSIGNS_DEVICE=cpu  # o "cuda"
```

---

## Troubleshooting

### Error: "CORS blocked"

- Asegúrate de que el backend está corriendo en `localhost:8000`
- El frontend debe estar en `localhost:5173`

### Error: "Model not found"

- Verifica que existe el checkpoint en `experiments/.../checkpoints/best.pt`
- O configura `COMSIGNS_CHECKPOINT` con la ruta correcta

### Error: "Invalid .pkl file"

- El archivo debe contener: `hand`, `body`, `face` (tensores o arrays)
- Usa [`scripts/extract_samples.py`](../scripts/extract_samples.py) para generar samples válidos

> [!NOTE]
> Para más troubleshooting, ver [Guía de Usuario](../../docs/USER_GUIDE.md#problemas-comunes-y-soluciones).

---

## 📚 Documentos Relacionados

- [🏗️ Arquitectura](ARCHITECTURE.md) — Pipeline y dataset
- [🧠 Modelo](../../docs/MODEL_ARCHITECTURE.md) — Encoder multimodal
- [📜 Scripts](SCRIPTS_USAGE.md) — CLI flags
- [⚙️ Servicios](../../services/SERVICES_TECH_DOC.md) — Docs técnicos de API
