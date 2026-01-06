# Guía: Reemplazar Modelos Placeholder con Modelos Reales

## 📋 Resumen

El sistema COMSIGNS ahora tiene una infraestructura completa para traducción de lenguaje de señas en tiempo real. Los módulos **glosador** y **traductor** están implementados con modelos placeholder que puedes reemplazar fácilmente con tus modelos entrenados.

---

## 🎯 Arquitectura Actual

```
Frame → Encoder → Glosador → Traductor → Texto Español
         (512D)    (Glosa)    (Traducción)
```

### Pipeline Completo:
1. **Frame** (base64) → decodificado a imagen
2. **MediaPipe** → extrae keypoints (manos, cuerpo, rostro)
3. **Encoder** → convierte keypoints a embeddings (T × 512)
4. **Glosador** → convierte embeddings a glosas (ej: "HOLA")
5. **Traductor** → convierte glosas a texto español (ej: "Hola")
6. **TextAccumulator** → acumula traducciones en contexto

---

## 🔧 Módulo 1: Glosador

### Ubicación
[`comsigns/services/glosador/__init__.py`](file:///home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns/services/glosador/__init__.py)

### Interfaz Actual

```python
class GlosadorPlaceholder(nn.Module):
    def __init__(self, embedding_dim=512, vocab_size=1000):
        # Tu modelo aquí
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, seq_len, embedding_dim)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        
    def decode_sequence(self, embeddings: torch.Tensor) -> Tuple[str, float]:
        """
        Args:
            embeddings: (batch_size, seq_len, embedding_dim)
        Returns:
            (glosa, confianza)
        """
```

### Cómo Reemplazar

#### Opción 1: Modelo CTC (Recomendado para secuencias)

```python
import torch
import torch.nn as nn

class GlosadorCTC(nn.Module):
    def __init__(self, embedding_dim=512, vocab_size=1000, hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Capa de salida para CTC
        self.fc = nn.Linear(hidden_dim * 2, vocab_size + 1)  # +1 para blank token
        
        # Vocabulario (cargar desde archivo)
        self.idx_to_gloss = self.load_vocabulary()
    
    def forward(self, embeddings):
        lstm_out, _ = self.lstm(embeddings)
        logits = self.fc(lstm_out)
        return logits
    
    def decode_sequence(self, embeddings):
        with torch.no_grad():
            logits = self.forward(embeddings)
            # CTC greedy decoding
            probs = torch.softmax(logits, dim=-1)
            # ... implementar CTC decode
            return gloss, confidence
```

#### Opción 2: Modelo Transformer (SLTUNET-like)

```python
class GlosadorTransformer(nn.Module):
    def __init__(self, embedding_dim=512, vocab_size=1000):
        super().__init__()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=2048
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Clasificador
        self.classifier = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, embeddings):
        # embeddings: (batch, seq_len, embedding_dim)
        # Transformer espera (seq_len, batch, embedding_dim)
        x = embeddings.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        logits = self.classifier(x)
        return logits
```

### Cargar Tu Modelo Entrenado

```python
def create_glosador(model_path=None, device="cpu"):
    # Reemplazar GlosadorPlaceholder con tu clase
    model = GlosadorCTC(embedding_dim=512, vocab_size=1000)
    
    if model_path:
        # Cargar pesos entrenados
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Cargar vocabulario
        model.idx_to_gloss = checkpoint['vocabulary']
    
    model.to(device)
    model.eval()
    return model
```

### Vocabulario

Crear archivo `vocabulary.json`:
```json
{
  "0": "HOLA",
  "1": "GRACIAS",
  "2": "POR_FAVOR",
  ...
}
```

---

## 🔧 Módulo 2: Traductor

### Ubicación
[`comsigns/services/translator/__init__.py`](file:///home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns/services/translator/__init__.py)

### Interfaz Actual

```python
class TranslatorPlaceholder(nn.Module):
    def translate_single(self, gloss: str) -> str:
        """Traduce una glosa a español"""
        
    def translate_sequence(self, glosses: List[str]) -> str:
        """Traduce secuencia de glosas"""
        
    def translate_with_context(self, gloss: str, previous_glosses: List[str]) -> str:
        """Traduce con contexto"""
```

### Cómo Reemplazar

#### Opción 1: Modelo Seq2Seq con Atención

```python
class TranslatorSeq2Seq(nn.Module):
    def __init__(self, vocab_size_gloss=1000, vocab_size_spanish=5000):
        super().__init__()
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size_gloss, 256)
        self.encoder = nn.LSTM(256, 512, batch_first=True)
        
        # Decoder con atención
        self.decoder_embedding = nn.Embedding(vocab_size_spanish, 256)
        self.decoder = nn.LSTM(256 + 512, 512, batch_first=True)
        self.attention = nn.MultiheadAttention(512, num_heads=8)
        self.fc_out = nn.Linear(512, vocab_size_spanish)
        
        # Vocabularios
        self.gloss_to_idx = {}
        self.idx_to_spanish = {}
    
    def translate_sequence(self, glosses: List[str]) -> str:
        # Convertir glosas a índices
        indices = [self.gloss_to_idx.get(g, 0) for g in glosses]
        
        # Encode
        # Decode con atención
        # Retornar texto español
        
        return translated_text
```

#### Opción 2: Modelo Transformer (mT5 o similar)

```python
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

class TranslatorMT5:
    def __init__(self, model_name="google/mt5-small"):
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
    
    def translate_sequence(self, glosses: List[str]) -> str:
        # Unir glosas con espacios
        input_text = " ".join(glosses)
        
        # Tokenizar
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Generar traducción
        outputs = self.model.generate(**inputs)
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translation
```

### Fine-tuning con tus datos

```python
# Preparar dataset
train_data = [
    {"glosses": ["HOLA", "COMO_ESTAS"], "spanish": "Hola, ¿cómo estás?"},
    {"glosses": ["GRACIAS", "POR_FAVOR"], "spanish": "Gracias, por favor"},
    # ...
]

# Fine-tune mT5
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./translator_model",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

---

## 🔄 Integración en la API

### Paso 1: Actualizar `create_glosador`

En [`glosador/__init__.py`](file:///home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns/services/glosador/__init__.py):

```python
def create_glosador(model_path=None, device="cpu"):
    # Reemplazar esta línea:
    # model = GlosadorPlaceholder()
    
    # Con tu modelo real:
    model = GlosadorCTC(embedding_dim=512, vocab_size=1000)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model
```

### Paso 2: Actualizar `create_translator`

En [`translator/__init__.py`](file:///home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns/services/translator/__init__.py):

```python
def create_translator(model_path=None, device="cpu"):
    # Reemplazar esta línea:
    # model = TranslatorPlaceholder()
    
    # Con tu modelo real:
    model = TranslatorSeq2Seq()
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model
```

### Paso 3: Configurar rutas de modelos

En [`config.py`](file:///home/srchaoz/ChaozDev/COMSIGNS-MULTIMODAL/comsigns/services/config.py):

```python
{
    "models": {
        "encoder": "path/to/encoder.pth",
        "glosador": "path/to/glosador.pth",
        "translator": "path/to/translator.pth"
    }
}
```

---

## 📊 Formato de Datos de Entrenamiento

### Para el Glosador

```json
{
  "video_id": "001",
  "embeddings": [[0.1, 0.2, ...], ...],  // T × 512
  "glosses": ["HOLA", "COMO_ESTAS"],
  "timestamps": [0.0, 0.5, 1.0, ...]
}
```

### Para el Traductor

```json
{
  "glosses": ["HOLA", "COMO_ESTAS"],
  "spanish": "Hola, ¿cómo estás?",
  "context": "greeting"
}
```

---

## 🧪 Probar Tus Modelos

### Test del Glosador

```python
# En glosador/__init__.py, al final:
if __name__ == "__main__":
    glosador = create_glosador("path/to/model.pth")
    
    # Simular embeddings
    embeddings = torch.randn(1, 30, 512)
    
    gloss, confidence = glosador.decode_sequence(embeddings)
    print(f"Glosa: {gloss}, Confianza: {confidence:.2f}")
```

### Test del Traductor

```python
# En translator/__init__.py, al final:
if __name__ == "__main__":
    translator = create_translator("path/to/model.pth")
    
    glosses = ["HOLA", "COMO_ESTAS"]
    translation = translator.translate_sequence(glosses)
    print(f"Traducción: {translation}")
```

---

## ✅ Checklist de Implementación

### Glosador
- [ ] Entrenar modelo de glosado (CTC/Transformer)
- [ ] Crear vocabulario de glosas
- [ ] Guardar checkpoint del modelo
- [ ] Reemplazar `GlosadorPlaceholder` con tu clase
- [ ] Actualizar `create_glosador()` para cargar tu modelo
- [ ] Probar con embeddings reales

### Traductor
- [ ] Entrenar modelo de traducción (Seq2Seq/Transformer)
- [ ] Crear vocabulario español
- [ ] Guardar checkpoint del modelo
- [ ] Reemplazar `TranslatorPlaceholder` con tu clase
- [ ] Actualizar `create_translator()` para cargar tu modelo
- [ ] Probar con glosas reales

### Integración
- [ ] Configurar rutas de modelos en `config.py`
- [ ] Verificar que los modelos se cargan correctamente
- [ ] Probar pipeline completo: frame → glosa → texto
- [ ] Verificar acumulación de texto
- [ ] Optimizar rendimiento (latencia < 100ms)

---

## 🚀 Ejemplo Completo de Uso

```python
# Después de reemplazar los modelos:

# 1. Frame llega al WebSocket
frame_base64 = "..."

# 2. Procesar con MediaPipe
keypoints = extract_keypoints(frame)

# 3. Encoder
embeddings = encoder(keypoints)  # → (1, T, 512)

# 4. Glosador
gloss, confidence = glosador.decode_sequence(embeddings)
# → ("HOLA", 0.95)

# 5. Traductor
translation = translator.translate_with_context(gloss, previous_glosses)
# → "Hola"

# 6. Acumular
accumulator.add(gloss, translation)
# → "Hola, ¿cómo estás?"
```

---

## 📝 Notas Importantes

1. **Mantén la interfaz**: No cambies los nombres de métodos (`decode_sequence`, `translate_with_context`, etc.) para que la API siga funcionando.

2. **Lazy loading**: Los modelos se cargan solo cuando se necesitan (primera llamada).

3. **Device**: Por defecto usa CPU. Para GPU, modifica `create_glosador(device="cuda")`.

4. **Vocabulario**: Asegúrate de que el vocabulario del glosador coincida con el del traductor.

5. **Performance**: Si la latencia es alta, considera:
   - Usar ONNX Runtime
   - Cuantización de modelos
   - Reducir tamaño de modelos
   - Procesamiento en GPU

---

## 🆘 Soporte

Si tienes dudas sobre cómo integrar tus modelos:

1. Revisa los ejemplos en los archivos `__init__.py` de cada módulo
2. Ejecuta los tests con `python -m comsigns.services.glosador`
3. Verifica logs del servidor para errores de carga

**¡El sistema está listo para recibir tus modelos entrenados!** 🎉
