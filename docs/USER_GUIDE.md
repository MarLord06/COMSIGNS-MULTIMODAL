# ComSigns — Guía de Usuario

**Resumen rápido**
- ComSigns es un sistema para reconocimiento de signos a partir de video. Extrae keypoints con MediaPipe, los codifica con un encoder multimodal y clasifica con un clasificador entrenado.

**Estructura importante del repositorio**
- `comsigns/`: código principal de la librería y servicios.
- `comsigns/scripts/infer_video.py`: script de inferencia por video (ya ajustado a las dimensiones del modelo).
- `comsigns/experiments/`: carpeta con checkpoints y mappings (ej. `micro_v1_retrained/best.pt`).
- `services/`, `training/`: módulos para encoder, preprocesado y entrenamiento.
- `data/`: datasets y videos de ejemplo.

**Requisitos y entorno**
1. Tener Python 3.9+ (se desarrolló con 3.9 en macOS M1).
2. Crear y activar un virtualenv (recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install -r comsigns/requirements.txt
```

Nota: MediaPipe en macOS puede emitir advertencias sobre LibreSSL/OpenSSL; son informativas en muchos casos.

**Configuración mínima**
- Si falta `comsigns/config.yaml`, copia la configuración ejemplo desde `infra/config.yaml`:

```bash
cp infra/config.yaml comsigns/config.yaml
```

- Verifica rutas a modelos en `comsigns/experiments/...` y que tengas permisos de lectura.

**Ejecutar inferencia de video**
1. Comando general (reemplaza `<RUTA_VIDEO>`):

```bash
python3 comsigns/scripts/infer_video.py \
  --video <RUTA_VIDEO> \
  --model comsigns/experiments/micro_v1_retrained/best.pt \
  --mapping comsigns/experiments/micro_v1_retrained/class_mapping.json \
  --device cpu
```

2. Ejemplo con un video de datos de ejemplo:

```bash
python3 comsigns/scripts/infer_video.py --video data/raw/lsp_aec/Videos/SEGMENTED_SIGN/ira_alegria/tú_302.mp4 --model comsigns/experiments/micro_v1_retrained/best.pt --mapping comsigns/experiments/micro_v1_retrained/class_mapping.json
```

3. Salida esperada: el script imprimirá shapes de los tensors extraídos y el top-K de predicciones.

**Notas sobre dimensiones y preprocesado**
- El encoder multimodal espera formatos:
  - `hand`: (seq_len, 126) → 2 manos × 21 keypoints × 3 valores (x,y,z)
  - `body`: (seq_len, 99) → 33 keypoints × 3 valores
  - `face`: (seq_len, 1404) → 468 keypoints × 3 valores
- El script `comsigns/scripts/infer_video.py` fue adaptado para producir vectores con 3 valores por keypoint y evitar errores de multiplicación de matrices.

**Entrenamiento (resumen rápido)**
- Scripts de entrenamiento están bajo `comsigns/scripts/` y `comsigns/training/`.
- Para entrenar, normalmente se llama al script principal de entrenamiento (ver `scripts/train.py` o `training/trainer.py` según versión). Ejemplo genérico:

```bash
python3 comsigns/scripts/train.py --config path/to/config.yaml
```

(Revisa `comsigns/experiments/*` para ver cómo se guardan checkpoints y `class_mapping.json`.)

**Cómo añadir un nuevo video al dataset para pruebas rápidas**
1. Coloca el video en `data/raw/...` o en una carpeta local.
2. Ejecuta el script de inferencia apuntando a esa ruta.

**Problemas comunes y soluciones**
- RuntimeError mat1 and mat2 shapes cannot be multiplied: indica desajuste en dimensiones de keypoints. Solución: usar la versión actualizada de `infer_video.py` que usa 3 valores por keypoint.
- Advertencias de MediaPipe/TensorFlow sobre delegados o GL: suelen ser informativas; si hay errores, verifica versiones de `mediapipe` y `tensorflow`.
- `Archivo de configuración no encontrado`: crea o copia `comsigns/config.yaml` desde `infra/config.yaml`.
- Problemas con GPU/CPU: pase `--device cpu` para forzar CPU; para GPU, asegúrate de tener CUDA y la versión de PyTorch adecuada.

**Verificaciones rápidas**
- ¿El script imprime shapes compatibles? Debe mostrar, por ejemplo: `Hand shape: (N, 126)`, `Body shape: (N, 99)`, `Face shape: (N, 1404)`.
- ¿Se cargó el checkpoint? El script imprimirá el nombre del archivo del modelo.

**Contacto y siguientes pasos**
- Si quieres que automatice pruebas sobre un video concreto, indícame la ruta y lo ejecuto aquí.
- Puedo añadir secciones adicionales: despliegue en Docker, tests automáticos, o un script `make infer`.

---
Guía generada automáticamente. Puedo ajustar el nivel de detalle o traducir secciones específicas si lo deseas.
