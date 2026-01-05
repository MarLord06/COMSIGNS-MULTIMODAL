"""
Módulo de configuración para cargar settings desde config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)


class Config:
    """Carga y gestiona la configuración del sistema desde config.yaml"""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Carga la configuración desde config.yaml

        Args:
            config_path: Ruta al archivo de configuración. Si es None, busca en la raíz del proyecto.

        Returns:
            Diccionario con la configuración cargada
        """
        if self._config:
            return self._config

        if config_path is None:
            # Buscar config.yaml en el directorio comsigns (donde está este archivo)
            # Primero intenta en el directorio del módulo comsigns
            module_dir = Path(__file__).parent.parent  # comsigns/
            config_path = module_dir / "config.yaml"
            
            # Si no existe, intenta en el directorio padre (por si se ejecuta desde otro lugar)
            if not config_path.exists():
                project_root = Path(__file__).parent.parent.parent
                alt_config_path = project_root / "config.yaml"
                if alt_config_path.exists():
                    config_path = alt_config_path

        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Archivo de configuración no encontrado: {config_path}")
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Configuración cargada desde: {config_path}")
            return self._config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de punto

        Args:
            key: Clave de configuración (ej: 'ingestion.video_dir')
            default: Valor por defecto si no se encuentra la clave

        Returns:
            Valor de configuración o default
        """
        if not self._config:
            self.load()

        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def reload(self) -> Dict[str, Any]:
        """Recarga la configuración desde el archivo"""
        self._config = {}
        return self.load()


# Instancia global de configuración
config = Config()

