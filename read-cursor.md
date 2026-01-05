# === CONTEXTO DEL PROYECTO COMSIGNS ===

Comsigns es un sistema experto de interpretación de Lengua de Señas (primero enfocado en LSEC - Lengua de Señas Ecuatoriana). 
El sistema debe:
- Capturar video (webcam o archivo).
- Extraer keypoints del cuerpo, manos y rostro.
- Procesar los keypoints en un encoder multimodal.
- Convertir la secuencia embebida en glosas (glosses).
- Traducir esas glosas a español natural.
- Servir la inferencia vía API para Web/Mobile.
- Permitir la recolección y anotación de datasets de señas.

La arquitectura está compuesta por los siguientes servicios:

1. **ingestion/**  
   - Captura video en tiempo real (WebRTC o webcam).  
   - Procesa clips vía FFmpeg.  
   - Genera un archivo MP4 y un manifest JSON.  

2. **preprocessing/**  
   - Extrae fotogramas.  
   - Usa MediaPipe (Hands, Pose, Face Mesh) para obtener keypoints.  
   - Guarda features en JSON o Parquet.  
   - Normaliza y calibra keypoints.  

3. **feature-store/**  
   - Almacena features T x K para entrenamiento.  
   - Maneja versiones, cleaning y sincronización.  

4. **encoder/**  
   - Implementación del Encoder Multimodal (PyTorch).  
   - Tres ramas: manos, cuerpo, rostro.  
   - Combina todo en un embedding temporal T x 512.  
   - Exporta modelo a ONNX para inferencia.  

5. **glosador/**  
   - Modelo SLTUNET-like (Transformer + CTC + Seq2Seq).  
   - Convierte embeddings → glosas.  

6. **translator/**  
   - Transformer o mT5-small para convertir glosas → español.  

7. **api/** (FastAPI)  
   - Exponer endpoint `/infer/video`.  
   - Pipeline completo: video → keypoints → gloss → texto.  
   - Manejo de sesiones, colas y errores.  

8. **web/**  
   - Interfaz React para:  
     - Subir video  
     - Ver inferencia  
     - Herramienta de anotación básica (glosas + texto)  

---

# === OBJETIVO GENERAL PARA CURSOR ===
Ayudar a construir todo el código del sistema Comsigns siguiendo esta arquitectura.  
Cursor debe generar código limpio, modular, reproducible y compatible con contenedores (Docker).  
Todas las herramientas deben integrarse por pasos, desde prototipos hasta producción.

---

# === ESPECIFICACIONES TÉCNICAS ===

## Lenguajes y frameworks
- **Python 3.10+**
- **FastAPI**
- **PyTorch**
- **Transformers (HuggingFace)**
- **MediaPipe**
- **OpenCV**
- **React + Vite** (para web)
- **Docker + docker-compose**
- **ONNX + ONNX Runtime** para inferencia optimizada

## Pipelines
1. Video → frames (FFmpeg)
2. Frames → keypoints (MediaPipe)
3. Keypoints → features (JSON/Parquet)
4. Features → embeddings (Encoder)
5. Embeddings → glosas (Glosador)
6. Glosas → texto español (Translator)

## Formato de features
```json
{
  "clip_id": "uuid",
  "fps": 30,
  "frames": [
    {
      "t": 0.033,
      "hand_keypoints": [[x,y,c], ...],   // 21
      "body_keypoints": [[x,y,c], ...],   // 33
      "face_keypoints": [[x,y,c], ...]    // 468
    }
  ],
  "meta": {
    "camera_id": "cam01",
    "user_id": "anon"
  }
}
