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

# Agregar el directorio raíz al path (donde está el paquete comsigns)
SCRIPT_DIR = Path(__file__).parent.resolve()
COMSIGNS_ROOT = SCRIPT_DIR.parent  # /comsigns
PROJECT_ROOT = COMSIGNS_ROOT.parent  # /COMSIGNS-MULTIMODAL
sys.path.insert(0, str(COMSIGNS_ROOT))

import torch
from torch.utils.data import DataLoader

from core.data.datasets.aec import AECDataset
from core.data.loaders import encoder_collate_fn
from core.data.splits import create_train_val_split
from services.encoder import MultimodalEncoder
from training import Trainer, TrainerConfig, SignLanguageClassifier

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
        default=None,  # Se resuelve dinámicamente
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
        '--no-validate',
        action='store_true',
        help='Desactivar validación (por defecto está activada)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Fracción de datos para validación con random split (default: 0.2)'
    )
    parser.add_argument(
        '--stratified',
        action='store_true',
        help='Usar split estratificado pre-generado (data/splits/aec_stratified.json)'
    )
    parser.add_argument(
        '--split-file',
        type=Path,
        default=None,
        help='Ruta a archivo de split personalizado (implica --stratified)'
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
    
    # Resolver path del dataset
    if args.dataset_path is None:
        args.dataset_path = PROJECT_ROOT / 'data' / 'raw' / 'lsp_aec'
    
    # Resolver split file
    if args.split_file is not None:
        args.stratified = True  # --split-file implica --stratified
    
    if args.stratified and args.split_file is None:
        args.split_file = PROJECT_ROOT / 'data' / 'splits' / 'aec_stratified.json'
    
    logger.info("=" * 60)
    logger.info("ComSigns Training")
    logger.info("=" * 60)
    
    # =========================================================================
    # 1. Cargar Dataset
    # =========================================================================
    logger.info(f"Cargando dataset desde: {args.dataset_path}")
    
    validate = not args.no_validate and not args.overfit
    val_loader = None
    
    if args.stratified and validate:
        # Usar split estratificado pre-generado
        logger.info(f"Usando split estratificado: {args.split_file}")
        
        if not args.split_file.exists():
            logger.error(f"Split file no encontrado: {args.split_file}")
            logger.error("Ejecuta primero: python scripts/generate_aec_split.py")
            sys.exit(1)
        
        train_dataset = AECDataset(
            args.dataset_path,
            split_file=args.split_file,
            split="train"
        )
        val_dataset = AECDataset(
            args.dataset_path,
            split_file=args.split_file,
            split="val"
        )
        
        num_classes = len(train_dataset.gloss_to_id)
        
        logger.info(f"  Train: {len(train_dataset)} muestras")
        logger.info(f"  Val: {len(val_dataset)} muestras")
        logger.info(f"  Clases (glosas): {num_classes}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=encoder_collate_fn,
            num_workers=0,
            pin_memory=args.device == 'cuda'
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=encoder_collate_fn,
            num_workers=0,
            pin_memory=args.device == 'cuda'
        )
    else:
        # Cargar dataset completo
        dataset = AECDataset(args.dataset_path)
        num_classes = len(dataset.gloss_to_id)
        
        logger.info(f"  Muestras totales: {len(dataset)}")
        logger.info(f"  Clases (glosas): {num_classes}")
        
        # =========================================================================
        # 2. Crear Train/Val Split (random)
        # =========================================================================
        if validate:
            logger.info(f"Creando split train/val random ({1-args.val_ratio:.0%}/{args.val_ratio:.0%})...")
            train_set, val_set = create_train_val_split(
                dataset, 
                val_ratio=args.val_ratio,
                seed=args.seed
            )
            logger.info(f"  Train: {len(train_set)} muestras")
            logger.info(f"  Val: {len(val_set)} muestras")
            
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=encoder_collate_fn,
                num_workers=0,
                pin_memory=args.device == 'cuda'
            )
            
            val_loader = DataLoader(
                val_set,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=encoder_collate_fn,
                num_workers=0,
                pin_memory=args.device == 'cuda'
            )
        else:
            logger.info("Validación desactivada - usando todo el dataset para training")
            train_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=encoder_collate_fn,
                num_workers=0,
                pin_memory=args.device == 'cuda'
            )
    
    logger.info(f"  Batches por epoch (train): {len(train_loader)}")
    if val_loader:
        logger.info(f"  Batches por epoch (val): {len(val_loader)}")
    
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
        seed=args.seed,
        validate=validate,
        val_ratio=args.val_ratio
    )
    
    logger.info(f"Configuración:")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Overfit mode: {config.overfit_single_batch}")
    logger.info(f"  Validation: {config.validate}")
    
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
    
    history = trainer.fit(train_loader, val_loader=val_loader)
    
    # =========================================================================
    # 6. Resultados
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Entrenamiento completado!")
    logger.info("=" * 60)
    logger.info(f"  Train Loss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}")
    
    if history['val_loss']:
        logger.info(f"  Val Loss: {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}")
        
        # Diagnóstico
        train_improved = history['train_loss'][-1] < history['train_loss'][0]
        val_improved = history['val_loss'][-1] < history['val_loss'][0]
        
        if train_improved and val_improved:
            logger.info("  ✅ Modelo generaliza - train y val loss disminuyen")
        elif train_improved and not val_improved:
            logger.warning("  ⚠️ Posible overfitting - train ↓ pero val ↑")
        else:
            logger.warning("  ⚠️ Underfitting - loss no disminuye")
    else:
        if history['train_loss'][-1] < history['train_loss'][0]:
            logger.info("  ✅ El loss disminuyó - el modelo está aprendiendo")
        else:
            logger.warning("  ⚠️ El loss no disminuyó - revisar hiperparámetros")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
