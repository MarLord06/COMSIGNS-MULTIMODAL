ComSigns — Documentación y referencia

Resumen
- Esta carpeta contiene el código y los modelos para reconocimiento de lengua de señas.

Documentación principal
- Documentación técnica del modelo: [docs/MODEL_TECHNICAL.md](docs/MODEL_TECHNICAL.md)
- Guía de usuario (cómo preparar entorno y ejecutar scripts): [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

Ubicaciones útiles
- Código del modelo y servicios: [comsigns/](comsigns/)
- Scripts de utilidad (inferencia / entrenamiento): [comsigns/scripts/](comsigns/scripts/)
- Experimentos y checkpoints: [comsigns/experiments/](comsigns/experiments/)
- Datos de ejemplo (videos): [data/raw/](data/raw/)

Ejemplo rápido — inferencia de un video

Reemplaza `<RUTA_VIDEO>` por la ruta del video a evaluar.

```bash
python3 comsigns/scripts/infer_video.py \
  --video <RUTA_VIDEO> \
  --model comsigns/experiments/micro_v1_retrained/best.pt \
  --mapping comsigns/experiments/micro_v1_retrained/class_mapping.json \
  --device cpu
```

Notas
- Consulta [docs/MODEL_TECHNICAL.md](docs/MODEL_TECHNICAL.md) para detalles internos del encoder, formatos de entrada (shapes) y artefactos.
- Consulta [docs/USER_GUIDE.md](docs/USER_GUIDE.md) para pasos de instalación, preprocesado y troubleshooting.

Contribuciones
- Este README es un índice; para cambios en la documentación edita los archivos en `docs/`.
