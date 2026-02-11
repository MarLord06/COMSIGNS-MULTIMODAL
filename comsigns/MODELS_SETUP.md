# 🔧 Configuración de Modelos de MediaPipe

> Descarga y configuración de los modelos MediaPipe Tasks necesarios para la extracción de keypoints.

---

## 📖 Navegación

| Documento | Descripción |
|-----------|-------------|
| [🚀 Inicio Rápido](QUICKSTART.md) | Setup mínimo |
| [👤 Guía de Usuario](../docs/USER_GUIDE.md) | Instalación completa y troubleshooting |
| [📋 Guía Completa](../INICIO.md) | Todos los pasos detallados |
| [🧠 Arquitectura del Modelo](../docs/MODEL_ARCHITECTURE.md) | Cómo se usan los keypoints en el encoder |

---

MediaPipe Tasks requiere modelos específicos (archivos `.task`) para funcionar. Estos modelos se descargan automáticamente la primera vez, pero si hay problemas de conexión, puedes descargarlos manualmente.

## Opción 1: Descarga Automática (Recomendado) ✅

Los modelos se descargarán automáticamente la primera vez que uses el sistema. Si falla, verifica tu conexión a internet.

## Opción 2: Descarga Manual con Script

```bash
python3 scripts/download_mediapipe_models.py
```

Este script descargará los modelos desde múltiples URLs y los guardará en `models/mediapipe/`.

## Opción 3: Descarga Manual desde el Navegador

1. Visita: https://developers.google.com/mediapipe/solutions/vision
2. Descarga los siguientes modelos:

| Modelo | Uso | Keypoints |
|--------|-----|-----------|
| `hand_landmarker.task` | Detección de manos | 21 keypoints × 2 manos |
| `pose_landmarker.task` | Pose corporal | 33 keypoints |
| `face_landmarker.task` | Landmarks faciales | 468 keypoints |

3. Colócalos en: `models/mediapipe/`

## Opción 4: Rutas Personalizadas

Si tienes los modelos en otra ubicación:

```python
from comsigns.services.preprocessing import KeypointExtractor

extractor = KeypointExtractor(model_paths={
    'hand': '/ruta/a/hand_landmarker.task',
    'pose': '/ruta/a/pose_landmarker.task',
    'face': '/ruta/a/face_landmarker.task'
})
```

> [!TIP]
> Ver [preprocessing/README.md](services/preprocessing/README.md) para más opciones del extractor.

---

## Verificar Instalación

```bash
ls -lh models/mediapipe/
```

Deberías ver los 3 archivos `.task` listados.

---

## 📚 Documentos Relacionados

- [🧠 Arquitectura del Modelo](../docs/MODEL_ARCHITECTURE.md) — Cómo se procesan los keypoints
- [📘 Documentación Técnica](../docs/MODEL_TECHNICAL.md) — Formato de entradas esperado
- [🚀 Inicio Rápido](QUICKSTART.md) — Setup mínimo
- [📋 Guía Completa](../INICIO.md) — Todos los pasos
