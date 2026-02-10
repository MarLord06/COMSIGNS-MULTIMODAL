.PHONY: install test lint format clean run-api run-web docker-build docker-up docker-down

# Instalación
install:
	pip install -r requirements.txt
	pip install -e .

# Tests
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=comsigns --cov-report=html

# Linting
lint:
	flake8 comsigns tests
	mypy comsigns

# Formateo
format:
	black comsigns tests

# Limpieza
clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

# Ejecutar API
run-api:
	cd .. && python3 -m comsigns.run_api

# Alternativa: ejecutar con uvicorn directamente (requiere instalación)
run-api-uvicorn:
	PYTHONPATH=.. python3 -m uvicorn comsigns.services.api.main:app --reload --host 0.0.0.0 --port 8000

# Ejecutar web (requiere npm install primero)
run-web:
	cd web && npm run dev

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Setup completo
setup: install
	mkdir -p data/{raw,processed,features} models logs
	touch data/raw/.gitkeep data/processed/.gitkeep data/features/.gitkeep
	touch models/.gitkeep logs/.gitkeep
	pip install -e .

# Verificar setup
check:
	python3 check_setup.py

