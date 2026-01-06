# 🚀 Guía Completa de Inicio - COMSIGNS

## 📋 Resumen del Sistema

COMSIGNS es un sistema completo de traducción de lenguaje de señas en tiempo real que incluye:

- ✅ **Captura en tiempo real** vía webcam con WebSocket
- ✅ **Procesamiento de video** con MediaPipe (keypoints)
- ✅ **Encoder multimodal** (manos, cuerpo, rostro)
- ✅ **Glosador** (embeddings → glosas)
- ✅ **Traductor** (glosas → español)
- ✅ **Frontend React** con modo cámara y subida de video

---

## 📁 Estructura del Proyecto

```
COMSIGNS-MULTIMODAL/
├── comsigns/
│   ├── services/
│   │   ├── api/              # FastAPI con WebSocket
│   │   ├── ingestion/        # Captura de video
│   │   ├── preprocessing/    # MediaPipe keypoints
│   │   ├── encoder/          # Modelo multimodal
│   │   ├── glosador/         # Embeddings → Glosas
│   │   └── translator/       # Glosas → Español
│   ├── web/                  # Frontend React
│   ├── config.yaml           # Configuración
│   ├── run_api.py            # Script para iniciar API
│   └── requirements.txt      # Dependencias Python
├── models/
│   └── mediapipe/            # Modelos MediaPipe
│       ├── face_landmarker.task
│       ├── hand_landmarker.task
│       └── pose_landmarker_lite.task
└── README.md
```

---

## ⚙️ Instalación y Configuración

### 1. Requisitos Previos

- **Python 3.10+**
- **Node.js 16+** y npm
- **CUDA** (opcional, para GPU)

### 2. Instalar Dependencias Python

```bash
cd /home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Verificar Instalación

```bash
python check_setup.py
```

Esto verificará:
- ✓ Dependencias instaladas
- ✓ Modelos MediaPipe descargados
- ✓ Configuración correcta

### 4. Instalar Dependencias Frontend

```bash
cd web
npm install
```

---

## 🔧 Configuración

### Archivo `config.yaml`

Ubicación: `/home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns/config.yaml`

```yaml
# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  reload: true

# MediaPipe Models
mediapipe:
  models_dir: "../models/mediapipe"
  hand_model: "hand_landmarker.task"
  pose_model: "pose_landmarker_lite.task"
  face_model: "face_landmarker.task"

# Modelos de IA (cuando los tengas entrenados)
models:
  encoder: null  # "path/to/encoder.pth"
  glosador: null  # "path/to/glosador.pth"
  translator: null  # "path/to/translator.pth"

# Procesamiento
preprocessing:
  default_fps: 30
  normalize_keypoints: true
```

### Configurar Rutas de Modelos (Cuando los tengas)

Cuando entrenes tus modelos, actualiza `config.yaml`:

```yaml
models:
  encoder: "models/encoder_trained.pth"
  glosador: "models/glosador_ctc.pth"
  translator: "models/translator_seq2seq.pth"
```

---

## 🚀 Iniciar el Sistema

### Opción 1: Inicio Rápido (Todo en uno)

```bash
# Terminal 1: Backend
cd /home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns
python run_api.py

# Terminal 2: Frontend
cd /home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns/web
npm run dev
```

### Opción 2: Inicio Manual

#### Backend

```bash
cd /home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns

# Activar entorno virtual
source venv/bin/activate

# Iniciar API
python run_api.py
```

**Verificar que funciona:**
- Abrir: `http://localhost:8000`
- Deberías ver: `{"message": "COMSIGNS API", "version": "0.1.0", ...}`

#### Frontend

```bash
cd /home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns/web

# Iniciar servidor de desarrollo
npm run dev
```

**Verificar que funciona:**
- Abrir: `http://localhost:5173`
- Deberías ver la interfaz de COMSIGNS

---

## 🎥 Usar el Sistema

### Modo 1: Cámara en Tiempo Real

1. **Abrir** `http://localhost:5173`
2. **Hacer clic** en "🎥 Cámara en Vivo"
3. **Hacer clic** en "🎥 Iniciar Cámara"
4. **Permitir** acceso a la cámara cuando el navegador lo solicite
5. **Ver resultados** en tiempo real:
   - Glosa detectada
   - Confianza
   - Traducción
   - Texto acumulado

### Modo 2: Subir Video

1. **Hacer clic** en "📤 Subir Video"
2. **Arrastrar** o seleccionar un archivo de video
3. **Hacer clic** en "Procesar Video"
4. **Ver resultados** completos del procesamiento

---

## 🔍 Estado Actual del Sistema

### ✅ Completamente Funcional

- [x] Backend API con FastAPI
- [x] Endpoint WebSocket `/ws/infer`
- [x] Procesamiento de frames con MediaPipe
- [x] Encoder multimodal (placeholder funcional)
- [x] Frontend React con modo cámara
- [x] Comunicación bidireccional en tiempo real
- [x] Acumulación de texto con contexto

### ⚠️ Usando Modelos Placeholder

Actualmente, el sistema usa **modelos placeholder** que funcionan pero retornan datos de ejemplo:

- **Encoder**: Modelo simple funcional (puede ser reemplazado)
- **Glosador**: Retorna glosas de ejemplo (HOLA, GRACIAS, etc.)
- **Traductor**: Usa diccionario simple para traducción

### 🎯 Para Usar Modelos Reales

Cuando tengas tus modelos entrenados:

1. **Entrenar modelos** (ver `GUIA_MODELOS.md`)
2. **Guardar checkpoints** en formato `.pth`
3. **Actualizar** `config.yaml` con rutas
4. **Reemplazar clases** en:
   - `services/glosador/__init__.py`
   - `services/translator/__init__.py`
5. **Reiniciar** el servidor

---

## 📊 Flujo de Datos Completo

```
Usuario → Webcam
    ↓
Frame (base64) → WebSocket
    ↓
Backend: decode_base64_frame()
    ↓
MediaPipe: extract_keypoints()
    ↓
Encoder: keypoints → embeddings (T × 512)
    ↓
Glosador: embeddings → glosa + confianza
    ↓
Traductor: glosa → texto español
    ↓
TextAccumulator: acumular con contexto
    ↓
WebSocket → Frontend
    ↓
RealtimeResult: mostrar traducción
```

---

## 🧪 Probar el Sistema

### Test 1: Health Check

```bash
curl http://localhost:8000/health
# Respuesta: {"status": "healthy"}
```

### Test 2: Endpoints Disponibles

```bash
curl http://localhost:8000/
# Muestra todos los endpoints disponibles
```

### Test 3: WebSocket (con wscat)

```bash
# Instalar wscat
npm install -g wscat

# Conectar al WebSocket
wscat -c ws://localhost:8000/ws/infer

# Deberías recibir:
# {"type": "status", "status": "connected", "session_id": "..."}
```

### Test 4: Cámara en Navegador

1. Abrir `http://localhost:5173`
2. Clic en "Cámara en Vivo"
3. Iniciar cámara
4. Verificar:
   - ✓ Preview de cámara visible
   - ✓ Estado "Conectado" (verde)
   - ✓ Contador de frames aumenta
   - ✓ Resultados aparecen abajo

---

## 🐛 Solución de Problemas

### Problema: "ModuleNotFoundError: No module named 'comsigns'"

**Solución:**
```bash
cd /home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python comsigns/run_api.py
```

### Problema: "MediaPipe models not found"

**Solución:**
```bash
cd /home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns
python scripts/download_mediapipe_models.py
```

### Problema: "Port 8000 already in use"

**Solución:**
```bash
# Encontrar proceso usando el puerto
lsof -i :8000

# Matar proceso
kill -9 <PID>

# O cambiar puerto en config.yaml
```

### Problema: "WebSocket connection failed"

**Solución:**
1. Verificar que el backend esté corriendo
2. Verificar URL en `CameraCapture.jsx` (línea 7)
3. Revisar consola del navegador para errores
4. Verificar firewall no bloquee WebSocket

### Problema: "Cámara no se activa"

**Solución:**
1. Verificar permisos del navegador
2. Usar HTTPS en producción (o localhost en desarrollo)
3. Cerrar otras aplicaciones usando la cámara
4. Probar con otro navegador

---

## 📝 Comandos Útiles

### Backend

```bash
# Iniciar API
python run_api.py

# Iniciar con logs detallados
python run_api.py --log-level debug

# Ejecutar tests
pytest tests/

# Verificar setup
python check_setup.py
```

### Frontend

```bash
# Desarrollo
npm run dev

# Build para producción
npm run build

# Preview de producción
npm run preview

# Limpiar node_modules
rm -rf node_modules && npm install
```

### Docker (si lo usas)

```bash
# Build
docker-compose build

# Iniciar
docker-compose up

# Detener
docker-compose down
```

---

## 🔐 Configuración para Producción

### 1. Variables de Entorno

```bash
export COMSIGNS_ENV=production
export COMSIGNS_API_HOST=0.0.0.0
export COMSIGNS_API_PORT=8000
```

### 2. HTTPS/WSS

Para producción, usa:
- **HTTPS** para el frontend
- **WSS** (WebSocket Secure) para WebSocket

Actualizar en `CameraCapture.jsx`:
```javascript
const WS_URL = 'wss://tu-dominio.com/ws/infer'
```

### 3. CORS

En `main.py`, actualizar:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tu-dominio.com"],  # Especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📚 Documentación Adicional

- **Guía de Modelos**: `GUIA_MODELOS.md` - Cómo reemplazar modelos placeholder
- **Walkthrough**: Ver artifacts - Detalles de implementación
- **README Frontend**: `web/README.md` - Documentación del frontend
- **API Docs**: `http://localhost:8000/docs` - Documentación interactiva

---

## 🎯 Próximos Pasos

### Para Desarrollo

1. **Entrenar modelos reales**:
   - Glosador con CTC o Transformer
   - Traductor con Seq2Seq o mT5
   
2. **Optimizar rendimiento**:
   - Usar ONNX Runtime
   - Implementar caché de predicciones
   - Reducir latencia a <50ms

3. **Agregar features**:
   - Grabación de sesiones
   - Exportar traducciones
   - Múltiples idiomas

### Para Producción

1. **Deploy backend** (ej: AWS, GCP, Azure)
2. **Deploy frontend** (ej: Vercel, Netlify)
3. **Configurar SSL/TLS**
4. **Implementar autenticación**
5. **Agregar monitoreo** (logs, métricas)

---

## ✅ Checklist de Inicio

- [ ] Instalar dependencias Python
- [ ] Instalar dependencias Node.js
- [ ] Verificar modelos MediaPipe
- [ ] Configurar `config.yaml`
- [ ] Iniciar backend (`python run_api.py`)
- [ ] Iniciar frontend (`npm run dev`)
- [ ] Probar health check
- [ ] Probar modo cámara
- [ ] Probar modo subida de video
- [ ] Revisar logs para errores

---

## 🆘 Soporte

Si encuentras problemas:

1. **Revisar logs** del backend y frontend
2. **Verificar** `check_setup.py`
3. **Consultar** documentación en `GUIA_MODELOS.md`
4. **Revisar** issues conocidos en este documento

---

**¡El sistema está listo para usar! 🎉**

Para empezar ahora mismo:
```bash
# Terminal 1
cd /home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns && python run_api.py

# Terminal 2
cd /home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns/web && npm run dev
```

Luego abre `http://localhost:5173` y haz clic en "🎥 Cámara en Vivo"
