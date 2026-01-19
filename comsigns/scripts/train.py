#!/usr/bin/env python3
"""
Script de entrenamiento para ComSigns.

Ejemplo de uso:
    # Entrenamiento normal
    python scripts/train.py

    # Modo overfit (debug)
    python scripts/train.py --overfit

    # Personalizar parámetros
    python scripts/train.py --epochs 20 --lr 0.001 --batch-size 8
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from comsigns.core.data.datasets.aec import AECDataset
from comsigns.core.data.loaders import encoder_collate_fn
from comsigns.services.encoder import MultimodalEncoder
from comsigns.training import Trainer, TrainerConfig, SignLanguageClassifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Entrenar ComSigns')
    
    parser.add_argument(
        '--dataset-path',
        type=Path,
        default=Path('data/raw/lsp_aec'),
        help='Ruta al dataset AEC'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Número de epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Tamaño del batch (default: 16)'
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device para entrenar (default: auto)'
    )
    parser.add_argument(
        '--overfit',
        action='store_true',
        help='Modo overfit: entrena con un solo batch (para debug)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("ComSigns Training")
    logger.info("=" * 60)
    
    # =========================================================================
    # 1. Cargar Dataset
    # =========================================================================
    logger.info(f"Cargando dataset desde: {args.dataset_path}")
    
    dataset = AECDataset(args.dataset_path)
    num_classes = len(dataset.gloss_to_id)
    
    logger.info(f"  Muestras totales: {len(dataset)}")
    logger.info(f"  Clases (glosas): {num_classes}")
    
    # =========================================================================
    # 2. Crear DataLoader
    # =========================================================================
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=encoder_collate_fn,
        num_workers=0,  # 0 para evitar problemas en macOS
        pin_memory=True if args.device == 'cuda' else False
    )
    
    logger.info(f"  Batches por epoch: {len(train_loader)}")
    
    # =========================================================================
    # 3. Crear Modelo
    # =========================================================================
    logger.info("Creando modelo...")
    
    encoder = MultimodalEncoder(
        hand_input_dim=168,   # 2 manos × 21 keypoints × 4 valores
        body_input_dim=132,   # 33 keypoints × 4 valores
        face_input_dim=1872,  # 468 keypoints × 4 valores
        hidden_dim=256,
        output_dim=512,
        num_layers=2,
        dropout=0.1
    )
    
    model = SignLanguageClassifier(
        encoder=encoder,
        num_classes=num_classes,
        pooling="mean",
        dropout=0.1
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parámetros totales: {total_params:,}")
    logger.info(f"  Parámetros entrenables: {trainable_params:,}")
    
    # =========================================================================
    # 4. Configurar Trainer
    # =========================================================================
    config = TrainerConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        device=args.device,
        log_every_n_steps=10,
        overfit_single_batch=args.overfit,
        gradient_clip_val=1.0,
        seed=args.seed
    )
    
    logger.info(f"Configuración:")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Overfit mode: {config.overfit_single_batch}")
    
    # =========================================================================
    # 5. Entrenar
    # =========================================================================
    trainer = Trainer(model, config)
    
    # Validar que todo funciona antes de entrenar
    logger.info("\nValidando setup...")
    first_batch = next(iter(train_loader))
    validation = trainer.validate_training(first_batch)
    
    if not validation["has_gradients"]:
        logger.error("¡Error! No hay gradientes. Algo está mal.")
        return 1
    
    logger.info(f"  Loss inicial: {validation['loss']:.4f}")
    logger.info(f"  Gradientes OK: {validation['non_zero_params']}/{validation['total_params']}")
    
    # Entrenar
    logger.info("\n" + "=" * 60)
    logger.info("Iniciando entrenamiento...")
    logger.info("=" * 60 + "\n")
    
    history = trainer.fit(train_loader)
    
    # =========================================================================
    # 6. Resultados
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Entrenamiento completado!")
    logger.info("=" * 60)
    logger.info(f"  Loss inicial: {history['loss'][0]:.4f}")
    logger.info(f"  Loss final: {history['loss'][-1]:.4f}")
    
    if history['loss'][-1] < history['loss'][0]:
        logger.info("  ✅ El loss disminuyó - el modelo está aprendiendo")
    else:
        logger.warning("  ⚠️ El loss no disminuyó - revisar hiperparámetros")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
