"""
Setup para el paquete COMSIGNS
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="comsigns",
    version="0.1.0",
    description="Sistema de Interpretación de Lengua de Señas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="COMSIGNS Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pyyaml>=6.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "torch>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ]
    },
)

