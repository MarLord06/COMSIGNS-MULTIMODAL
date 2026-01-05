#!/usr/bin/env python3
"""
Script para verificar que el setup esté correcto
"""

import sys
from pathlib import Path

def check_imports():
    """Verifica que las importaciones funcionen"""
    print("Verificando importaciones...")
    
    try:
        from comsigns.services.config import config
        print("✓ Config importado correctamente")
    except ImportError as e:
        print(f"✗ Error importando config: {e}")
        return False
    
    try:
        from comsigns.services.schemas import FeatureClip
        print("✓ Schemas importado correctamente")
    except ImportError as e:
        print(f"✗ Error importando schemas: {e}")
        return False
    
    try:
        from comsigns.services.api.main import app
        print("✓ API importada correctamente")
    except ImportError as e:
        print(f"✗ Error importando API: {e}")
        return False
    
    return True

def check_structure():
    """Verifica la estructura de directorios"""
    print("\nVerificando estructura...")
    
    required_dirs = [
        "services",
        "services/ingestion",
        "services/preprocessing",
        "services/encoder",
        "services/api",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/ existe")
        else:
            print(f"✗ {dir_path}/ no existe")
            all_ok = False
    
    return all_ok

def main():
    """Función principal"""
    print("=== Verificación de Setup COMSIGNS ===\n")
    
    # Agregar al path si es necesario
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Agregado al PYTHONPATH: {project_root}\n")
    
    structure_ok = check_structure()
    imports_ok = check_imports()
    
    print("\n=== Resultado ===")
    if structure_ok and imports_ok:
        print("✓ Todo está correcto!")
        print("\nPara ejecutar la API:")
        print("  python3 run_api.py")
        print("\nO instalar el paquete:")
        print("  pip install -e .")
        return 0
    else:
        print("✗ Hay problemas con el setup")
        if not structure_ok:
            print("  - Verifica la estructura de directorios")
        if not imports_ok:
            print("  - Instala el paquete: pip install -e .")
            print("  - O ejecuta desde el directorio padre")
        return 1

if __name__ == "__main__":
    sys.exit(main())

