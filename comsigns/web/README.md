# COMSIGNS Web - Frontend en Tiempo Real

Interfaz web para el sistema COMSIGNS de interpretación de Lengua de Señas con soporte para **cámara en tiempo real** y subida de archivos de video.

## 🚀 Características

### Modo Cámara en Vivo (Nuevo)
- ✅ Captura en tiempo real desde webcam
- ✅ Comunicación bidireccional vía WebSocket
- ✅ Procesamiento de frames a 10 FPS
- ✅ Predicciones instantáneas
- ✅ Reconexión automática
- ✅ Indicadores de estado de conexión

### Modo Subida de Video (Original)
- ✅ Subida de archivos de video (MP4, AVI, MOV, MKV)
- ✅ Procesamiento completo del video
- ✅ Resultados detallados con embeddings

## 📋 Requisitos

- Node.js 16+
- npm o yarn
- Navegador moderno con soporte para:
  - WebSocket
  - MediaStream API (getUserMedia)
  - Canvas API

## 🛠️ Instalación

```bash
# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm run dev

# Construir para producción
npm run build
```

## 🎥 Uso

### Iniciar el Frontend

```bash
npm run dev
```

La aplicación estará disponible en `http://localhost:5173`

### Iniciar el Backend

Asegúrate de que el backend esté corriendo en `http://localhost:8000`:

```bash
cd ../../
python comsigns/run_api.py
```

## 🔧 Configuración

### URL del WebSocket

Por defecto, el frontend se conecta a `ws://localhost:8000/ws/infer`. Para cambiar esto, edita `src/components/CameraCapture.jsx`:

```javascript
const WS_URL = 'ws://tu-servidor:puerto/ws/infer'
```

### Frecuencia de Captura

Para ajustar la frecuencia de captura de frames (por defecto 10 FPS), edita `src/components/CameraCapture.jsx`:

```javascript
const CAPTURE_FPS = 10 // Cambiar según necesidad
```

### Resolución de Cámara

Para cambiar la resolución de captura, edita:

```javascript
const FRAME_WIDTH = 640
const FRAME_HEIGHT = 480
```

## 📱 Permisos de Cámara

Al usar el modo de cámara en vivo, el navegador solicitará permisos para acceder a la webcam. Asegúrate de:

1. Permitir el acceso cuando se solicite
2. Usar HTTPS en producción (requerido por navegadores modernos)
3. Verificar que ninguna otra aplicación esté usando la cámara

## 🔄 Protocolo WebSocket

### Mensajes del Cliente al Servidor

**Enviar Frame:**
```json
{
  "type": "frame",
  "data": {
    "frame": "base64_encoded_image",
    "timestamp": 1234567890123,
    "sequence": 42
  }
}
```

**Control:**
```json
{
  "type": "control",
  "action": "reset" | "stop"
}
```

### Mensajes del Servidor al Cliente

**Predicción:**
```json
{
  "type": "prediction",
  "data": {
    "sequence": 42,
    "gloss": "HOLA",
    "confidence": 0.85,
    "text": "Hola",
    "accumulated_text": "Hola, ¿cómo estás?",
    "processing_time_ms": 45.2,
    "frames_in_buffer": 30
  }
}
```

**Estado:**
```json
{
  "type": "status",
  "status": "connected" | "reset" | "stopped",
  "session_id": "uuid",
  "message": "Mensaje descriptivo"
}
```

**Error:**
```json
{
  "type": "error",
  "error": "Descripción del error",
  "code": "ERROR_CODE"
}
```

## 🐛 Solución de Problemas

### La cámara no se activa

1. Verifica los permisos del navegador
2. Asegúrate de que ninguna otra aplicación esté usando la cámara
3. Revisa la consola del navegador para errores
4. Intenta con otro navegador

### WebSocket no se conecta

1. Verifica que el backend esté corriendo
2. Comprueba la URL del WebSocket
3. Revisa la consola del navegador
4. Verifica que no haya firewall bloqueando la conexión

### Latencia alta

1. Reduce `CAPTURE_FPS` (ej: de 10 a 5 FPS)
2. Reduce la resolución de captura
3. Verifica la carga del servidor backend
4. Comprueba la velocidad de tu conexión

### Reconexión constante

1. Verifica la estabilidad del backend
2. Revisa los logs del servidor
3. Comprueba la conexión de red
4. Aumenta el timeout de reconexión

## 📊 Componentes

### `CameraCapture.jsx`
Componente principal para captura de cámara en tiempo real con:
- Acceso a webcam
- Cliente WebSocket
- Captura de frames
- Reconexión automática
- Indicadores de estado

### `RealtimeResult.jsx`
Visualización de resultados en tiempo real:
- Glosa actual
- Barra de confianza
- Texto traducido
- Texto acumulado
- Estadísticas de procesamiento

### `VideoUploader.jsx`
Componente original para subida de archivos de video.

### `InferenceResult.jsx`
Visualización de resultados de procesamiento de video completo.

## 🎨 Personalización

Los estilos están en archivos CSS individuales para cada componente. Puedes personalizar:

- Colores en `App.css`
- Diseño de cámara en `CameraCapture.css`
- Visualización de resultados en `RealtimeResult.css`

## 📝 Notas de Desarrollo

- El componente usa `useRef` para manejar referencias a video, canvas y WebSocket
- La reconexión usa backoff exponencial (3s, 6s, 12s, máx 30s)
- Los frames se capturan usando Canvas API
- La conversión a base64 usa `toDataURL` con calidad JPEG 0.8

## 🔐 Seguridad

- Los frames no se almacenan en el backend (solo procesamiento en memoria)
- Usa HTTPS en producción
- El WebSocket debe usar WSS (WebSocket Secure) en producción
- Implementa autenticación si es necesario

## 📄 Licencia

[Especificar licencia del proyecto]

## 👥 Contribuciones

[Instrucciones para contribuir al proyecto]
