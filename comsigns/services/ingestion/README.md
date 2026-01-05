# Módulo de Ingestion

Módulo para captura y procesamiento de video desde webcam o archivo.

## Uso

### Captura desde webcam

```python
from comsigns.services.ingestion import capture_from_webcam

video_path, manifest = capture_from_webcam(
    camera_id=0,
    duration=10.0  # segundos
)
```

### Procesar archivo de video

```python
from comsigns.services.ingestion import capture_from_file

video_path, manifest = capture_from_file("video.mp4")
```

### Validar video

```python
from comsigns.services.ingestion.utils import validate_video

is_valid, error = validate_video("video.mp4")
```

## Funcionalidades

- Captura desde webcam con OpenCV
- Procesamiento de archivos de video
- Validación de formatos y contenido
- Generación de manifests JSON
- Extracción de frames

