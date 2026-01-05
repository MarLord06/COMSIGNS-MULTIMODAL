#!/usr/bin/env python3
"""
Script para ejecutar la API con el PYTHONPATH configurado correctamente
"""

import sys
from pathlib import Path

# Agregar el directorio padre al PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn
from comsigns.services.config import config

if __name__ == "__main__":
    cfg = config.load()
    api_config = cfg.get('api', {})
    
    uvicorn.run(
        "comsigns.services.api.main:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=True
    )

