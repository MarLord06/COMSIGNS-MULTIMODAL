#!/bin/bash
# Script de inicio rápido para COMSIGNS
# Inicia backend y frontend automáticamente

set -e

echo "🚀 Iniciando COMSIGNS..."
echo ""

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directorio del proyecto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMSIGNS_DIR="$PROJECT_ROOT/comsigns"
WEB_DIR="$COMSIGNS_DIR/web"

# Función para verificar si un comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verificar Python
if ! command_exists python3; then
    echo "❌ Python 3 no está instalado"
    exit 1
fi

# Verificar Node.js
if ! command_exists node; then
    echo "❌ Node.js no está instalado"
    exit 1
fi

echo -e "${BLUE}📦 Verificando dependencias...${NC}"

# Verificar dependencias Python
cd "$COMSIGNS_DIR"
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Entorno virtual no encontrado. Creando...${NC}"
    python3 -m venv venv
fi

# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias Python si es necesario
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}📥 Instalando dependencias Python...${NC}"
    pip install -r requirements.txt
fi

# Verificar dependencias Node.js
cd "$WEB_DIR"
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📥 Instalando dependencias Node.js...${NC}"
    npm install
fi

echo -e "${GREEN}✓ Dependencias verificadas${NC}"
echo ""

# Función para limpiar procesos al salir
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Deteniendo servicios...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Servicios detenidos${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Iniciar backend
echo -e "${BLUE}🔧 Iniciando backend...${NC}"
cd "$COMSIGNS_DIR"
source venv/bin/activate
python run_api.py > logs/backend.log 2>&1 &
BACKEND_PID=$!

# Esperar a que el backend esté listo
echo -e "${YELLOW}⏳ Esperando a que el backend esté listo...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend listo en http://localhost:8000${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${YELLOW}⚠️  Backend tardó demasiado en iniciar. Revisa logs/backend.log${NC}"
    fi
done

# Iniciar frontend
echo -e "${BLUE}🎨 Iniciando frontend...${NC}"
cd "$WEB_DIR"
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!

# Esperar a que el frontend esté listo
echo -e "${YELLOW}⏳ Esperando a que el frontend esté listo...${NC}"
sleep 3

echo ""
echo -e "${GREEN}✅ COMSIGNS está corriendo!${NC}"
echo ""
echo -e "${BLUE}📍 URLs:${NC}"
echo -e "   Backend:  ${GREEN}http://localhost:8000${NC}"
echo -e "   Frontend: ${GREEN}http://localhost:5173${NC}"
echo -e "   API Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}💡 Consejos:${NC}"
echo "   - Abre http://localhost:5173 en tu navegador"
echo "   - Haz clic en '🎥 Cámara en Vivo' para usar la cámara"
echo "   - Presiona Ctrl+C para detener los servicios"
echo ""
echo -e "${BLUE}📋 Logs:${NC}"
echo "   - Backend:  tail -f $COMSIGNS_DIR/logs/backend.log"
echo "   - Frontend: tail -f $COMSIGNS_DIR/logs/frontend.log"
echo ""

# Mantener el script corriendo
wait
