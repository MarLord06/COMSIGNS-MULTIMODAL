# Módulo de Encoder Multimodal

Encoder PyTorch para procesar keypoints de manos, cuerpo y rostro.

## Arquitectura

El encoder consta de tres ramas independientes:

1. **HandBranch**: Procesa keypoints de manos (21 puntos × 2 manos)
2. **BodyBranch**: Procesa keypoints del cuerpo (33 puntos)
3. **FaceBranch**: Procesa keypoints del rostro (468 puntos)

Cada rama utiliza LSTM para procesamiento temporal, y los embeddings se fusionan en un embedding final de dimensión configurable (por defecto 512).

## Uso

### Crear encoder

```python
from comsigns.services.encoder import create_encoder

encoder = create_encoder()
```

### Procesar features

```python
from comsigns.services.encoder import feature_clip_to_tensors

# Convertir FeatureClip a tensores
tensors = feature_clip_to_tensors(feature_clip)

# Codificar
with torch.no_grad():
    embeddings = encoder(
        tensors['hand'].unsqueeze(0),
        tensors['body'].unsqueeze(0),
        tensors['face'].unsqueeze(0)
    )
```

## Configuración

Editar `config.yaml`:

```yaml
encoder:
  hidden_dim: 256
  output_dim: 512
  num_layers: 2
  dropout: 0.1
```

