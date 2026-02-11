# 🧠 Módulo de Encoder Multimodal

> Encoder PyTorch para procesar keypoints de manos, cuerpo y rostro.

---

## 📖 Ver También

| Documento | Descripción |
|-----------|-------------|
| [🧠 Arquitectura del Modelo](../../../docs/MODEL_ARCHITECTURE.md) | Diagramas detallados del encoder |
| [🏗️ Arquitectura General](../../docs/ARCHITECTURE.md) | Pipeline completo |
| [⚙️ Servicios](../../../services/SERVICES_TECH_DOC.md) | Docs técnicos detallados |
| [📘 Documentación Técnica](../../../docs/MODEL_TECHNICAL.md) | I/O y módulos internos |

---

## Arquitectura

El encoder consta de tres ramas independientes:

| Rama | Input | Keypoints | Output |
|------|-------|-----------|--------|
| **HandBranch** | `(B, T, 126)` | 21 puntos × 2 manos × 3 | `(B, T, 256)` |
| **BodyBranch** | `(B, T, 99)` | 33 puntos × 3 | `(B, T, 256)` |
| **FaceBranch** | `(B, T, 1404)` | 468 puntos × 3 | `(B, T, 256)` |

Cada rama utiliza LSTM para procesamiento temporal. Los embeddings se fusionan en un embedding final de **512 dimensiones**.

> [!TIP]
> Para diagramas mermaid detallados, ver [MODEL_ARCHITECTURE.md](../../../docs/MODEL_ARCHITECTURE.md).

---

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

---

## Configuración

Editar `config.yaml`:

```yaml
encoder:
  hidden_dim: 256
  output_dim: 512
  num_layers: 2
  dropout: 0.1
```

---

## 📚 Docs Relacionados

- [🧠 Arquitectura del Modelo](../../../docs/MODEL_ARCHITECTURE.md) — Diagramas detallados
- [🏗️ Arquitectura General](../../docs/ARCHITECTURE.md) — Pipeline + resultados
- [📘 Técnico](../../../docs/MODEL_TECHNICAL.md) — I/O y módulos
