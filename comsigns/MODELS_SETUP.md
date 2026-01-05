# Configuración de Modelos de MediaPipe

MediaPipe Tasks requiere modelos específicos (archivos `.task`) para funcionar. Estos modelos se descargan automáticamente la primera vez, pero si hay problemas de conexión, puedes descargarlos manualmente.

## Opción 1: Descarga Automática (Recomendado)

Los modelos se descargarán automáticamente la primera vez que uses el sistema. Si falla, verifica tu conexión a internet.

## Opción 2: Descarga Manual con Script

Ejecuta el script de descarga:

```bash
python3 scripts/download_mediapipe_models.py
```

Este script intentará descargar los modelos desde múltiples URLs y los guardará en `models/mediapipe/`.

## Opción 3: Descarga Manual desde el Navegador

1. Visita: https://developers.google.com/mediapipe/solutions/vision
2. Descarga los siguientes modelos:
   - `hand_landmarker.task`
   - `pose_landmarker.task`
   - `face_landmarker.task`
3. Colócalos en: `comsigns/models/mediapipe/`

## Opción 4: Especificar Rutas Personalizadas

Si tienes los modelos en otra ubicación, puedes especificarlos en el código:

```python
from comsigns.services.preprocessing import KeypointExtractor

extractor = KeypointExtractor(model_paths={
    'hand': '/ruta/a/hand_landmarker.task',
    'pose': '/ruta/a/pose_landmarker.task',
    'face': '/ruta/a/face_landmarker.task'
})
```

## Verificar Instalación

Para verificar que los modelos están correctamente instalados:

```bash
ls -lh models/mediapipe/
```

Deberías ver los 3 archivos `.task` listados.

