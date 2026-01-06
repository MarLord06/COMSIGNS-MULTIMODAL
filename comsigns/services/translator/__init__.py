"""
Módulo Traductor para COMSIGNS
Convierte secuencias de glosas en texto en español natural

Este módulo proporciona una interfaz para el modelo de traducción.
Actualmente usa un placeholder que puede ser reemplazado con el modelo real.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TranslatorPlaceholder(nn.Module):
    """
    Placeholder para el modelo de traducción glosas → español.
    
    Este modelo será reemplazado por el modelo real (ej: Transformer o mT5-small).
    La interfaz se mantiene para facilitar la integración.
    """
    
    def __init__(self):
        """Inicializa el traductor placeholder"""
        super().__init__()
        
        # Diccionario de traducción simple (reemplazar con modelo real)
        self.gloss_to_spanish = {
            "HOLA": "Hola",
            "GRACIAS": "Gracias",
            "POR_FAVOR": "Por favor",
            "ADIOS": "Adiós",
            "SI": "Sí",
            "NO": "No",
            "AYUDA": "Ayuda",
            "BIEN": "Bien",
            "MAL": "Mal",
            "COMO_ESTAS": "¿Cómo estás?",
            "BUENOS_DIAS": "Buenos días",
            "BUENAS_TARDES": "Buenas tardes",
            "BUENAS_NOCHES": "Buenas noches",
            "MUCHO_GUSTO": "Mucho gusto",
            "DE_NADA": "De nada",
            "PERDON": "Perdón",
            "DISCULPA": "Disculpa",
            "ENTIENDO": "Entiendo",
            "NO_ENTIENDO": "No entiendo",
            "REPITE": "Repite por favor",
            # Agregar más traducciones según el vocabulario
        }
        
        # Reglas de combinación simple (reemplazar con modelo seq2seq real)
        self.phrase_patterns = {
            ("HOLA", "COMO_ESTAS"): "Hola, ¿cómo estás?",
            ("GRACIAS", "POR_FAVOR"): "Gracias, por favor",
            ("BUENOS_DIAS", "COMO_ESTAS"): "Buenos días, ¿cómo estás?",
            ("MUCHO_GUSTO", "COMO_ESTAS"): "Mucho gusto, ¿cómo estás?",
        }
        
        logger.info("TranslatorPlaceholder inicializado")
    
    def translate_single(self, gloss: str) -> str:
        """
        Traduce una sola glosa a español.
        
        Args:
            gloss: Glosa a traducir
        
        Returns:
            Texto en español
        """
        return self.gloss_to_spanish.get(gloss, gloss.lower().replace("_", " "))
    
    def translate_sequence(self, glosses: List[str]) -> str:
        """
        Traduce una secuencia de glosas a texto en español natural.
        
        Args:
            glosses: Lista de glosas
        
        Returns:
            Texto en español
        """
        if not glosses:
            return ""
        
        # Intentar encontrar patrones de frases
        for i in range(len(glosses) - 1):
            pattern = tuple(glosses[i:i+2])
            if pattern in self.phrase_patterns:
                return self.phrase_patterns[pattern]
        
        # Si no hay patrón, traducir cada glosa y unir
        translations = [self.translate_single(gloss) for gloss in glosses]
        
        # Unir con espacios y capitalizar la primera letra
        result = " ".join(translations)
        if result:
            result = result[0].upper() + result[1:]
        
        return result
    
    def translate_with_context(self, gloss: str, previous_glosses: List[str] = None) -> str:
        """
        Traduce una glosa considerando el contexto de glosas anteriores.
        
        Args:
            gloss: Glosa actual a traducir
            previous_glosses: Lista de glosas anteriores para contexto
        
        Returns:
            Texto en español
        """
        if previous_glosses is None:
            previous_glosses = []
        
        # Combinar con glosas anteriores para buscar patrones
        all_glosses = previous_glosses[-5:] + [gloss]  # Últimas 5 glosas + actual
        
        # Intentar traducir la secuencia completa
        full_translation = self.translate_sequence(all_glosses)
        
        # Si hay traducción previa, solo retornar la parte nueva
        if previous_glosses:
            prev_translation = self.translate_sequence(previous_glosses[-5:])
            if full_translation.startswith(prev_translation):
                new_part = full_translation[len(prev_translation):].strip()
                return new_part if new_part else self.translate_single(gloss)
        
        return self.translate_single(gloss)


class TextAccumulator:
    """
    Acumulador de texto para mantener el historial de traducción.
    Maneja la concatenación inteligente de traducciones.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Args:
            max_history: Número máximo de glosas a mantener en historial
        """
        self.glosses_history: List[str] = []
        self.text_history: List[str] = []
        self.max_history = max_history
    
    def add(self, gloss: str, translation: str):
        """
        Agrega una nueva glosa y su traducción al historial.
        
        Args:
            gloss: Glosa detectada
            translation: Traducción al español
        """
        self.glosses_history.append(gloss)
        self.text_history.append(translation)
        
        # Mantener solo las últimas max_history entradas
        if len(self.glosses_history) > self.max_history:
            self.glosses_history = self.glosses_history[-self.max_history:]
            self.text_history = self.text_history[-self.max_history:]
    
    def get_accumulated_text(self) -> str:
        """
        Obtiene el texto acumulado completo.
        
        Returns:
            Texto completo acumulado
        """
        if not self.text_history:
            return ""
        
        # Unir todas las traducciones
        # Filtrar traducciones vacías
        valid_translations = [t for t in self.text_history if t.strip()]
        
        if not valid_translations:
            return ""
        
        # Unir con espacios, evitando duplicados consecutivos
        result = []
        prev = None
        for text in valid_translations:
            if text != prev:
                result.append(text)
                prev = text
        
        return " ".join(result)
    
    def reset(self):
        """Reinicia el acumulador"""
        self.glosses_history.clear()
        self.text_history.clear()
    
    def get_recent_glosses(self, n: int = 5) -> List[str]:
        """
        Obtiene las últimas n glosas.
        
        Args:
            n: Número de glosas a obtener
        
        Returns:
            Lista de glosas recientes
        """
        return self.glosses_history[-n:] if self.glosses_history else []


def create_translator(model_path: Optional[str] = None, device: str = "cpu") -> TranslatorPlaceholder:
    """
    Crea y carga el modelo de traducción.
    
    Args:
        model_path: Ruta al modelo entrenado (opcional para placeholder)
        device: Dispositivo para el modelo ('cpu' o 'cuda')
    
    Returns:
        Modelo de traducción listo para inferencia
    """
    model = TranslatorPlaceholder()
    model.to(device)
    model.eval()
    
    if model_path:
        try:
            logger.info(f"Cargando modelo de traducción desde {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("Modelo de traducción cargado exitosamente")
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo: {e}. Usando placeholder.")
    else:
        logger.info("Usando TranslatorPlaceholder (sin modelo entrenado)")
    
    return model


# Ejemplo de uso
if __name__ == "__main__":
    # Crear traductor
    translator = create_translator()
    
    # Crear acumulador
    accumulator = TextAccumulator()
    
    # Simular secuencia de glosas
    glosses = ["HOLA", "COMO_ESTAS", "BIEN", "GRACIAS"]
    
    print("Traducción frame por frame:")
    for i, gloss in enumerate(glosses):
        # Traducir con contexto
        translation = translator.translate_with_context(
            gloss, 
            accumulator.get_recent_glosses()
        )
        
        # Agregar al acumulador
        accumulator.add(gloss, translation)
        
        print(f"Frame {i}: {gloss} → {translation}")
        print(f"  Acumulado: {accumulator.get_accumulated_text()}")
        print()
    
    # Traducir secuencia completa
    print("\nTraducción de secuencia completa:")
    full_translation = translator.translate_sequence(glosses)
    print(f"Glosas: {' '.join(glosses)}")
    print(f"Traducción: {full_translation}")
