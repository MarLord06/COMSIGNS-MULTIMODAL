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
from datetime import datetime
from pathlib import Path

# Agregar el directorio raíz al path (donde está el paquete comsigns)
SCRIPT_DIR = Path(__file__).parent.resolve()
COMSIGNS_ROOT = SCRIPT_DIR.parent  # /comsigns
PROJECT_ROOT = COMSIGNS_ROOT.parent  # /COMSIGNS-MULTIMODAL
sys.path.insert(0, str(COMSIGNS_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.data.datasets.aec import AECDataset
from core.data.loaders import encoder_collate_fn
from core.data.splits import create_train_val_split
from services.encoder import MultimodalEncoder
from training import Trainer, TrainerConfig, SignLanguageClassifier
from training import RemapConfig, ClassRemapper, RemappedDataset, compute_class_support
from training import (
    build_semantic_whitelist,
    DictIdDataset,
    FilteredGlossDataset,
    LowSupportFilter,
    AugmentConfig,
    KeypointAugmenter,
    RebalanceConfig,
    RebalancedDataset
)
from training import ExperimentMetricsTracker, create_experiment_tracker
from training import CheckpointManager, load_checkpoint_for_training
from training.analysis import (
    LearnedWordCriteria,
    LearnedWordsAnalyzer,
    ClassMetrics,
    analyze_learned_words
)

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
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Ejecutar evaluación final y guardar matriz de confusión'
    )
    parser.add_argument(
        '--eval-output',
        type=Path,
        default=None,
        help='Directorio para guardar artefactos de evaluación (default: experiments/run_XXX/)'
    )
    parser.add_argument(
        '--tail-to-other',
        action='store_true',
        help='Collapse TAIL classes into a single OTHER class'
    )
    parser.add_argument(
        '--head-threshold',
        type=int,
        default=10,
        help='Minimum samples for HEAD bucket (default: 10)'
    )
    parser.add_argument(
        '--semantic-closure',
        action='store_true',
        help='Enable semantic closure filtering using dict.json + class_mapping.json'
    )
    parser.add_argument(
        '--class-mapping',
        type=Path,
        default=None,
        help='Path to class_mapping.json for semantic closure (required if --semantic-closure)'
    )
    parser.add_argument(
        '--use-dict-ids',
        action='store_true',
        help='Use dict.json old_id as class IDs (recommended for semantic compatibility)'
    )
    parser.add_argument(
        '--min-class-samples',
        type=int,
        default=20,
        help='Minimum samples per class (classes below are removed)'
    )
    parser.add_argument(
        '--remove-low-support',
        action='store_true',
        help='Remove classes with < min-class-samples'
    )
    parser.add_argument(
        '--rebalance',
        action='store_true',
        help='Enable dataset rebalancing with augmentation'
    )
    parser.add_argument(
        '--other-max-multiplier',
        type=float,
        default=2.0,
        help='Max OTHER samples = multiplier * median(non-OTHER)'
    )
    parser.add_argument(
        '--other-max-cap',
        type=int,
        default=0,
        help='Absolute cap for OTHER samples (0 = no cap)'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable keypoint augmentation for upsampling'
    )
    parser.add_argument(
        '--augment-noise-std',
        type=float,
        default=0.01,
        help='Gaussian noise std for keypoints (default: 0.01)'
    )
    parser.add_argument(
        '--augment-time-shift',
        type=int,
        default=2,
        help='Max temporal shift in frames (default: 2)'
    )
    parser.add_argument(
        '--augment-mirror-prob',
        type=float,
        default=0.0,
        help='Probability of left/right mirroring (default: 0.0)'
    )
    parser.add_argument(
        '--class-weighting',
        action='store_true',
        help='Use class-weighted CrossEntropyLoss'
    )
    parser.add_argument(
        '--other-penalty',
        type=float,
        default=1.2,
        help='Multiplier for OTHER class weight (default: 1.2)'
    )
    parser.add_argument(
        '--resume',
        type=Path,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--keep-last-n',
        type=int,
        default=0,
        help='Number of recent checkpoints to keep (0 = keep all)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize variables that may be used later
    remapper = None
    train_dataset = None
    val_dataset = None
    dataset = None
    
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

        # Use dict.json IDs for semantic compatibility (optional)
        dict_path = args.dataset_path / "dict.json"
        if args.use_dict_ids:
            logger.info("Aplicando DictIdDataset (gloss_id = dict.json old_id)")
            train_dataset = DictIdDataset(train_dataset, dict_path)
            val_dataset = DictIdDataset(val_dataset, dict_path)

        # Semantic closure filtering (optional)
        if args.semantic_closure:
            if args.class_mapping is None:
                logger.error("--semantic-closure requiere --class-mapping")
                sys.exit(1)
            valid_old_ids, missing_gloss = build_semantic_whitelist(
                dict_path=dict_path,
                class_mapping_path=args.class_mapping
            )
            logger.info(
                f"Semantic closure: {len(valid_old_ids)} clases válidas, "
                f"missing_gloss={len(missing_gloss)}"
            )
            train_dataset = FilteredGlossDataset(
                train_dataset,
                valid_old_ids,
                remap_ids=not args.use_dict_ids
            )
            val_dataset = FilteredGlossDataset(
                val_dataset,
                valid_old_ids,
                remap_ids=not args.use_dict_ids
            )

        # Remove low-support classes (optional)
        if args.remove_low_support:
            allowed_ids, counts = LowSupportFilter.filter_by_min_support(
                train_dataset,
                min_support=args.min_class_samples
            )
            removed = sorted(set(counts.keys()) - set(allowed_ids))
            logger.info(
                f"Low-support filter: removed {len(removed)} classes (<{args.min_class_samples})"
            )
            train_dataset = FilteredGlossDataset(
                train_dataset,
                allowed_ids,
                remap_ids=not args.use_dict_ids
            )
            val_dataset = FilteredGlossDataset(
                val_dataset,
                allowed_ids,
                remap_ids=not args.use_dict_ids
            )
        
        num_classes = len(train_dataset.gloss_to_id)
        
        logger.info(f"  Train: {len(train_dataset)} muestras")
        logger.info(f"  Val: {len(val_dataset)} muestras")
        logger.info(f"  Clases (glosas): {num_classes}")
        
        # =====================================================================
        # TAIL → OTHER: Apply class remapping if enabled
        # =====================================================================
        remapper = None
        if args.tail_to_other:
            logger.info(f"\n  Applying TAIL → OTHER remapping (head_threshold={args.head_threshold})...")
            
            # Compute training support
            train_support = compute_class_support(train_dataset)
            
            # Create and fit remapper
            remap_config = RemapConfig(
                strategy="tail_to_other",
                head_threshold=args.head_threshold
            )
            remapper = ClassRemapper(remap_config)
            remapper.fit(train_support, dict(train_dataset.gloss_to_id))
            
            # Log remapping stats
            summary = remapper.get_config_summary()
            logger.info(f"  Original classes: {summary['num_classes_original']}")
            logger.info(f"  Remapped classes: {summary['num_classes_remapped']}")
            logger.info(f"  HEAD: {summary['head_classes']} classes")
            logger.info(f"  MID: {summary['mid_classes']} classes")
            logger.info(f"  TAIL → OTHER: {summary['tail_classes']} classes collapsed")
            logger.info(f"  Samples in OTHER: {summary['samples_in_other']}")
            
            # Wrap datasets
            train_dataset = RemappedDataset(train_dataset, remapper)
            val_dataset = RemappedDataset(val_dataset, remapper)
            num_classes = remapper.num_classes_remapped
            
            logger.info(f"  New num_classes: {num_classes}")

        # Rebalancing (optional)
        augmenter = None
        if args.rebalance and args.augment:
            augmenter = KeypointAugmenter(AugmentConfig(
                time_shift=args.augment_time_shift,
                noise_std=args.augment_noise_std,
                mirror_prob=args.augment_mirror_prob
            ))

        if args.rebalance:
            class_support = compute_class_support(train_dataset)
            other_class_id = remapper.other_class_id if remapper is not None else None
            rebalance_config = RebalanceConfig(
                other_max_multiplier=args.other_max_multiplier,
                other_max_cap=args.other_max_cap
            )
            train_dataset = RebalancedDataset(
                train_dataset,
                other_class_id=other_class_id,
                class_counts=class_support,
                augmenter=augmenter,
                config=rebalance_config
            )
        
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

        # Use dict.json IDs for semantic compatibility (optional)
        dict_path = args.dataset_path / "dict.json"
        if args.use_dict_ids:
            logger.info("Aplicando DictIdDataset (gloss_id = dict.json old_id)")
            dataset = DictIdDataset(dataset, dict_path)

        # Semantic closure filtering (optional)
        if args.semantic_closure:
            if args.class_mapping is None:
                logger.error("--semantic-closure requiere --class-mapping")
                sys.exit(1)
            valid_old_ids, missing_gloss = build_semantic_whitelist(
                dict_path=dict_path,
                class_mapping_path=args.class_mapping
            )
            logger.info(
                f"Semantic closure: {len(valid_old_ids)} clases válidas, "
                f"missing_gloss={len(missing_gloss)}"
            )
            dataset = FilteredGlossDataset(
                dataset,
                valid_old_ids,
                remap_ids=not args.use_dict_ids
            )

        # Remove low-support classes (optional)
        if args.remove_low_support:
            allowed_ids, counts = LowSupportFilter.filter_by_min_support(
                dataset,
                min_support=args.min_class_samples
            )
            removed = sorted(set(counts.keys()) - set(allowed_ids))
            logger.info(
                f"Low-support filter: removed {len(removed)} classes (<{args.min_class_samples})"
            )
            dataset = FilteredGlossDataset(
                dataset,
                allowed_ids,
                remap_ids=not args.use_dict_ids
            )
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

            # Apply TAIL → OTHER if enabled
            if args.tail_to_other:
                logger.info(f"\n  Applying TAIL → OTHER remapping (head_threshold={args.head_threshold})...")
                train_support = compute_class_support(train_set)
                remap_config = RemapConfig(
                    strategy="tail_to_other",
                    head_threshold=args.head_threshold
                )
                remapper = ClassRemapper(remap_config)
                remapper.fit(train_support, dict(train_set.gloss_to_id))

                summary = remapper.get_config_summary()
                logger.info(f"  Original classes: {summary['num_classes_original']}")
                logger.info(f"  Remapped classes: {summary['num_classes_remapped']}")
                logger.info(f"  HEAD: {summary['head_classes']} classes")
                logger.info(f"  MID: {summary['mid_classes']} classes")
                logger.info(f"  TAIL → OTHER: {summary['tail_classes']} classes collapsed")
                logger.info(f"  Samples in OTHER: {summary['samples_in_other']}")

                train_set = RemappedDataset(train_set, remapper)
                val_set = RemappedDataset(val_set, remapper)
                num_classes = remapper.num_classes_remapped

            # Rebalancing (optional)
            augmenter = None
            if args.rebalance and args.augment:
                augmenter = KeypointAugmenter(AugmentConfig(
                    time_shift=args.augment_time_shift,
                    noise_std=args.augment_noise_std,
                    mirror_prob=args.augment_mirror_prob
                ))

            if args.rebalance:
                class_support = compute_class_support(train_set)
                other_class_id = remapper.other_class_id if remapper is not None else None
                rebalance_config = RebalanceConfig(
                    other_max_multiplier=args.other_max_multiplier,
                    other_max_cap=args.other_max_cap
                )
                train_set = RebalancedDataset(
                    train_set,
                    other_class_id=other_class_id,
                    class_counts=class_support,
                    augmenter=augmenter,
                    config=rebalance_config
                )
            
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
            # Apply TAIL → OTHER if enabled
            if args.tail_to_other:
                logger.info(f"\n  Applying TAIL → OTHER remapping (head_threshold={args.head_threshold})...")
                train_support = compute_class_support(dataset)
                remap_config = RemapConfig(
                    strategy="tail_to_other",
                    head_threshold=args.head_threshold
                )
                remapper = ClassRemapper(remap_config)
                remapper.fit(train_support, dict(dataset.gloss_to_id))

                summary = remapper.get_config_summary()
                logger.info(f"  Original classes: {summary['num_classes_original']}")
                logger.info(f"  Remapped classes: {summary['num_classes_remapped']}")
                logger.info(f"  HEAD: {summary['head_classes']} classes")
                logger.info(f"  MID: {summary['mid_classes']} classes")
                logger.info(f"  TAIL → OTHER: {summary['tail_classes']} classes collapsed")
                logger.info(f"  Samples in OTHER: {summary['samples_in_other']}")

                dataset = RemappedDataset(dataset, remapper)
                num_classes = remapper.num_classes_remapped

            # Rebalancing (optional)
            augmenter = None
            if args.rebalance and args.augment:
                augmenter = KeypointAugmenter(AugmentConfig(
                    time_shift=args.augment_time_shift,
                    noise_std=args.augment_noise_std,
                    mirror_prob=args.augment_mirror_prob
                ))

            if args.rebalance:
                class_support = compute_class_support(dataset)
                other_class_id = remapper.other_class_id if remapper is not None else None
                rebalance_config = RebalanceConfig(
                    other_max_multiplier=args.other_max_multiplier,
                    other_max_cap=args.other_max_cap
                )
                dataset = RebalancedDataset(
                    dataset,
                    other_class_id=other_class_id,
                    class_counts=class_support,
                    augmenter=augmenter,
                    config=rebalance_config
                )

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
    # 4.1 Class-Weighted Loss (if enabled)
    # =========================================================================
    loss_fn = None
    
    if args.class_weighting:
        logger.info("\nCalculando pesos de clases para la pérdida...")
        
        # Get final training dataset (after all transforms)
        if args.stratified and train_dataset is not None:
            # Try to get the underlying dataset
            final_train_ds = train_dataset
        else:
            final_train_ds = dataset
        
        # Compute class support from final dataset
        final_support = compute_class_support(final_train_ds)
        
        # Compute inverse frequency weights
        total_samples = sum(final_support.values())
        class_weights = torch.zeros(num_classes, dtype=torch.float32)
        
        for class_id in range(num_classes):
            support = final_support.get(class_id, 0)
            if support > 0:
                # Inverse frequency weighting: w_i = N / (n_classes * n_i)
                class_weights[class_id] = total_samples / (num_classes * support)
            else:
                # Classes with no samples get weight 1.0 (shouldn't affect training)
                class_weights[class_id] = 1.0
        
        # Apply OTHER penalty if configured
        if args.other_penalty != 1.0 and remapper is not None:
            # Find the OTHER class ID
            other_class_id = remapper.other_class_id
            if other_class_id is not None:
                original_weight = class_weights[other_class_id].item()
                class_weights[other_class_id] *= args.other_penalty
                logger.info(f"  Penalización OTHER (class_id={other_class_id}): "
                           f"{original_weight:.4f} → {class_weights[other_class_id].item():.4f}")
        
        # Move weights to device (align with training device)
        device = config.get_torch_device()
        class_weights = class_weights.to(device)
        
        # Create weighted loss
        if device.type == "mps":
            class WeightedCrossEntropyLoss(nn.Module):
                def __init__(self, weight: torch.Tensor, reduction: str = "mean"):
                    super().__init__()
                    self.register_buffer("weight", weight)
                    self.reduction = reduction

                def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                    log_probs = torch.log_softmax(logits, dim=1)
                    nll = -log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
                    if self.weight is not None:
                        nll = nll * self.weight[targets]
                    if self.reduction == "sum":
                        return nll.sum()
                    return nll.mean()

            loss_fn = WeightedCrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        # Log weight statistics
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Weight range: [{class_weights.min().item():.4f}, {class_weights.max().item():.4f}]")
        logger.info(f"  Weight mean: {class_weights.mean().item():.4f}")
    
    # =========================================================================
    # 5. Entrenar con Checkpointing
    # =========================================================================
    trainer = Trainer(model, config, num_classes=num_classes, loss_fn=loss_fn)
    
    # Validar que todo funciona antes de entrenar
    logger.info("\nValidando setup...")
    first_batch = next(iter(train_loader))
    validation = trainer.validate_training(first_batch)
    
    if not validation["has_gradients"]:
        logger.error("¡Error! No hay gradientes. Algo está mal.")
        return 1
    
    logger.info(f"  Loss inicial: {validation['loss']:.4f}")
    logger.info(f"  Gradientes OK: {validation['non_zero_params']}/{validation['total_params']}")
    
    # =========================================================================
    # 5.1 Preparar directorio de salida (siempre para checkpoints)
    # =========================================================================
    if args.eval_output:
        output_dir = args.eval_output
    else:
        # Crear directorio con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / 'comsigns' / 'experiments' / f'run_{timestamp}'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener nombres de clases del dataset
    if args.tail_to_other and remapper is not None:
        # Use remapped class names
        class_names = [remapper.new_class_names.get(i, f"class_{i}") 
                      for i in range(remapper.num_classes_remapped)]
    elif args.stratified:
        class_names = list(train_dataset.gloss_to_id.keys())
    else:
        class_names = list(dataset.gloss_to_id.keys())
    
    # Save remapper config if TAIL → OTHER is enabled
    if args.tail_to_other and remapper is not None:
        remapper.save(output_dir / "class_mapping.json")
        logger.info(f"  Class mapping saved to: {output_dir / 'class_mapping.json'}")
    
    # =========================================================================
    # Export class_index.json (new_id → gloss mapping for semantic resolution)
    # =========================================================================
    import json
    class_index = {}
    for i, name in enumerate(class_names):
        class_index[str(i)] = name
    
    class_index_path = output_dir / "class_index.json"
    with open(class_index_path, 'w', encoding='utf-8') as f:
        json.dump(class_index, f, ensure_ascii=False, indent=2)
    logger.info(f"  Class index saved to: {class_index_path}")
    
    logger.info(f"  Output directory: {output_dir}")
    if args.eval:
        logger.info(f"  Evaluación final: habilitada")
    
    # For backward compatibility
    eval_output_dir = output_dir
    
    # =========================================================================
    # 5.2 Setup Checkpointing (SIEMPRE habilitado)
    # =========================================================================
    checkpoint_manager = CheckpointManager(
        output_dir=output_dir,
        keep_last_n=args.keep_last_n
    )
    logger.info(f"  Checkpointing enabled: {output_dir / 'checkpoints'}")
    
    start_epoch = 1
    
    # Handle resume
    if args.resume is not None:
        if not args.resume.exists():
            logger.error(f"Resume checkpoint not found: {args.resume}")
            return 1
        
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint_for_training(
            checkpoint_path=args.resume,
            model=model,
            optimizer=trainer.optimizer,
            device=config.get_torch_device()
        )
        logger.info(f"  Resuming from epoch {start_epoch}")
    
    # Build bucket mapping for learned words analysis (needed per-epoch)
    bucket_mapping = {}
    other_class_id = None
    
    if args.tail_to_other and remapper is not None:
        other_class_id = remapper.other_class_id
        for new_id in range(remapper.num_classes_remapped):
            if new_id == other_class_id:
                bucket_mapping[new_id] = "OTHER"
            else:
                try:
                    bucket = remapper.get_new_class_bucket(new_id)
                    bucket_mapping[new_id] = bucket.value
                except (ValueError, AttributeError):
                    bucket_mapping[new_id] = "UNKNOWN"
    else:
        # Baseline: compute bucket mapping from training support
        if args.stratified and train_dataset is not None:
            base_ds = train_dataset.base_dataset if hasattr(train_dataset, 'base_dataset') else train_dataset
            train_support = compute_class_support(base_ds)
        elif dataset is not None:
            train_support = compute_class_support(dataset)
        else:
            train_support = {}
        
        for class_id, support in train_support.items():
            if support >= args.head_threshold:
                bucket_mapping[class_id] = "HEAD"
            elif 3 <= support <= 9:
                bucket_mapping[class_id] = "MID"
            else:
                bucket_mapping[class_id] = "TAIL"
    
    # Create epoch end callback for checkpointing
    def epoch_end_callback(epoch: int, model: torch.nn.Module, optimizer, metrics: dict):
        """Callback to save checkpoints and track best model."""
        # Compute learned words for this epoch (only if we have validation data)
        learned_words_count = 0
        
        if val_loader is not None:
            import numpy as np
            
            # Quick validation pass to get predictions
            model.eval()
            device = config.get_torch_device()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    hand_kps = batch["hand"].to(device)
                    body_kps = batch["body"].to(device)
                    face_kps = batch["face"].to(device)
                    lengths = batch["lengths"].to(device)
                    labels = batch["labels"]
                    
                    logits = model(hand_kps, body_kps, face_kps, lengths)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_targets.extend(labels.numpy().tolist())
            
            model.train()
            
            # Compute per-class metrics
            preds_np = np.array(all_preds)
            targets_np = np.array(all_targets)
            
            metrics_by_class = {}
            for class_id in range(num_classes):
                support = (targets_np == class_id).sum()
                if support == 0:
                    metrics_by_class[class_id] = ClassMetrics(
                        class_id=class_id, support=0,
                        precision=0.0, recall=0.0, f1=0.0
                    )
                    continue
                
                tp = ((preds_np == class_id) & (targets_np == class_id)).sum()
                fp = ((preds_np == class_id) & (targets_np != class_id)).sum()
                fn = ((preds_np != class_id) & (targets_np == class_id)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                metrics_by_class[class_id] = ClassMetrics(
                    class_id=class_id,
                    support=int(support),
                    precision=float(precision),
                    recall=float(recall),
                    f1=float(f1)
                )
            
            # Run learned words analysis
            analyzer = LearnedWordsAnalyzer(
                metrics_by_class=metrics_by_class,
                bucket_mapping=bucket_mapping,
                criteria=LearnedWordCriteria()
            )
            learned_report = analyzer.analyze()
            learned_words_count = learned_report.learned_words_count
        
        # Build checkpoint metrics
        checkpoint_metrics = {
            "epoch": epoch,
            "train_loss": metrics.get("train_loss", float("inf")),
            "val_loss": metrics.get("val_loss", float("inf")),
            "f1_macro": metrics.get("f1_macro", 0.0),
            "learned_words_count": learned_words_count,
            "accuracy": metrics.get("accuracy"),
            "accuracy_top5": metrics.get("top5_acc")
        }
        
        # Save checkpoint
        extra_state = {
            "num_classes": num_classes,
            "tail_to_other": args.tail_to_other
        }
        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metrics=checkpoint_metrics,
            extra_state=extra_state
        )
        
        # Update best if needed
        is_new_best = checkpoint_manager.update_best_if_needed(
            model=model,
            metrics=checkpoint_metrics,
            optimizer=optimizer,
            extra_state=extra_state
        )
        
        if is_new_best:
            logger.info(f"  ★ New best model: learned_words={learned_words_count}, f1={checkpoint_metrics['f1_macro']:.4f}")
        else:
            # Log current metrics for visibility
            logger.info(f"  Checkpoint metrics: learned_words={learned_words_count}, f1={checkpoint_metrics['f1_macro']:.4f}")
    
    # Entrenar
    logger.info("\n" + "=" * 60)
    logger.info("Iniciando entrenamiento...")
    logger.info("=" * 60 + "\n")
    
    history = trainer.fit(
        train_loader, 
        val_loader=val_loader,
        run_final_eval=args.eval,
        eval_output_dir=eval_output_dir,
        class_names=class_names,
        epoch_end_callback=epoch_end_callback,
        start_epoch=start_epoch
    )
    
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
    
    # Mostrar artefactos de evaluación si se generaron
    if args.eval and 'eval_artifacts' in history:
        logger.info("\n" + "=" * 60)
        logger.info("Artefactos de evaluación guardados:")
        logger.info("=" * 60)
        for name, path in history['eval_artifacts'].items():
            logger.info(f"  {name}: {path}")
    
    # =========================================================================
    # 7. Experiment Metrics (TAIL → OTHER specific)
    # =========================================================================
    if args.eval and val_loader is not None and eval_output_dir is not None:
        logger.info("\n" + "=" * 60)
        logger.info("Calculando métricas del experimento...")
        logger.info("=" * 60)
        
        # Use bucket_mapping computed earlier in section 5.1
        # Determine experiment_id
        if args.tail_to_other:
            experiment_id = f"tail_to_other_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            experiment_id = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create tracker
        exp_tracker = ExperimentMetricsTracker(
            num_classes=num_classes,
            bucket_mapping=bucket_mapping,
            other_class_id=other_class_id,
            experiment_id=experiment_id
        )
        
        # Run inference on validation set
        model.eval()
        device = config.get_torch_device()
        model.to(device)
        
        with torch.no_grad():
            for batch in val_loader:
                hand_kps = batch["hand"].to(device)
                body_kps = batch["body"].to(device)
                face_kps = batch["face"].to(device)
                lengths = batch["lengths"].to(device)
                labels = batch["labels"]
                
                logits = model(hand_kps, body_kps, face_kps, lengths)
                exp_tracker.update(logits, labels)
        
        # Export experiment metrics
        exp_artifacts = exp_tracker.export_artifacts(eval_output_dir)
        
        # Print summary
        logger.info(exp_tracker.get_summary())
        
        logger.info("\nMétricas del experimento guardadas:")
        for name, path in exp_artifacts.items():
            logger.info(f"  {name}: {path}")
        
        # =====================================================================
        # 8. Learned Words Analysis
        # =====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("Analizando palabras aprendidas...")
        logger.info("=" * 60)
        
        # Get predictions and targets from tracker
        all_preds = exp_tracker.predictions
        all_targets = exp_tracker.targets
        
        # Compute per-class metrics
        import numpy as np
        preds_np = np.array(all_preds)
        targets_np = np.array(all_targets)
        
        metrics_by_class = {}
        for class_id in range(num_classes):
            support = (targets_np == class_id).sum()
            if support == 0:
                metrics_by_class[class_id] = ClassMetrics(
                    class_id=class_id, support=0,
                    precision=0.0, recall=0.0, f1=0.0
                )
                continue
            
            tp = ((preds_np == class_id) & (targets_np == class_id)).sum()
            fp = ((preds_np == class_id) & (targets_np != class_id)).sum()
            fn = ((preds_np != class_id) & (targets_np == class_id)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics_by_class[class_id] = ClassMetrics(
                class_id=class_id,
                support=int(support),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1)
            )
        
        # Run learned words analysis
        learned_criteria = LearnedWordCriteria(
            min_support=2,
            min_precision=0.5,
            min_recall=0.5,
            min_f1=0.5
        )
        
        analyzer = LearnedWordsAnalyzer(
            metrics_by_class=metrics_by_class,
            bucket_mapping=bucket_mapping,
            criteria=learned_criteria
        )
        learned_report = analyzer.analyze()
        
        # Save learned words report
        learned_report.save(eval_output_dir / "learned_words_report.json")
        
        # Print summary
        logger.info(learned_report.get_summary())
        
        logger.info(f"\n  Learned words report saved to: {eval_output_dir / 'learned_words_report.json'}")
    
    # =========================================================================
    # 9. Checkpoint Summary
    # =========================================================================
    if checkpoint_manager is not None and checkpoint_manager.has_best():
        logger.info("\n" + "=" * 60)
        logger.info("CHECKPOINT SUMMARY")
        logger.info("=" * 60)
        logger.info(checkpoint_manager.get_summary())
        
        # Save final training state
        training_state = {
            "final_epoch": args.epochs,
            "train_loss_final": history['train_loss'][-1] if history['train_loss'] else None,
            "val_loss_final": history['val_loss'][-1] if history['val_loss'] else None,
            "tail_to_other": args.tail_to_other,
            "head_threshold": args.head_threshold,
            "num_classes": num_classes,
            "seed": args.seed,
            "completed_at": datetime.now().isoformat(),
            # New semantic closure params
            "semantic_closure": args.semantic_closure,
            "use_dict_ids": args.use_dict_ids,
            "min_class_samples": args.min_class_samples,
            "remove_low_support": args.remove_low_support,
            # Rebalancing params
            "rebalance": args.rebalance,
            "other_max_multiplier": args.other_max_multiplier,
            "other_max_cap": args.other_max_cap if hasattr(args, 'other_max_cap') else None,
            # Augmentation params
            "augment": args.augment,
            "augment_noise_std": args.augment_noise_std,
            "augment_time_shift": args.augment_time_shift,
            "augment_mirror_prob": args.augment_mirror_prob,
            # Class weighting params
            "class_weighting": args.class_weighting,
            "other_penalty": args.other_penalty
        }
        checkpoint_manager.save_training_state(training_state)
        logger.info(f"  Training state saved to: {output_dir / 'training_state.json'}")
        
        # Generate training report markdown
        report_lines = [
            "# Training Report",
            "",
            f"**Completed at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Training Configuration",
            "",
            f"- **Epochs:** {args.epochs}",
            f"- **Batch Size:** {args.batch_size}",
            f"- **Learning Rate:** {args.lr}",
            f"- **Device:** {args.device}",
            f"- **Seed:** {args.seed}",
            "",
            "## Class Remapping",
            "",
            f"- **TAIL → OTHER:** {args.tail_to_other}",
            f"- **HEAD threshold:** {args.head_threshold}",
            f"- **Number of classes:** {num_classes}",
            "",
            "## Semantic Closure",
            "",
            f"- **Enabled:** {args.semantic_closure}",
            f"- **Use dict IDs:** {args.use_dict_ids}",
            f"- **Min class samples:** {args.min_class_samples}",
            f"- **Remove low support:** {args.remove_low_support}",
            "",
            "## Rebalancing",
            "",
            f"- **Enabled:** {args.rebalance}",
            f"- **OTHER max multiplier:** {args.other_max_multiplier}",
            f"- **OTHER max cap:** {getattr(args, 'other_max_cap', 'N/A')}",
            "",
            "## Augmentation",
            "",
            f"- **Enabled:** {args.augment}",
            f"- **Noise std:** {args.augment_noise_std}",
            f"- **Time shift:** {args.augment_time_shift}",
            f"- **Mirror prob:** {args.augment_mirror_prob}",
            "",
            "## Class Weighting",
            "",
            f"- **Enabled:** {args.class_weighting}",
            f"- **OTHER penalty:** {args.other_penalty}",
            "",
            "## Results",
            "",
            f"- **Final train loss:** {history['train_loss'][-1]:.4f}" if history['train_loss'] else "- **Final train loss:** N/A",
            f"- **Final val loss:** {history['val_loss'][-1]:.4f}" if history['val_loss'] else "- **Final val loss:** N/A",
            "",
            "## Artifacts",
            "",
            f"- `best.pt` - Best model checkpoint",
            f"- `class_mapping.json` - Class remapping configuration",
            f"- `class_index.json` - New class ID to gloss mapping",
            f"- `training_state.json` - Full training state",
            "",
        ]
        
        report_path = output_dir / "training_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        logger.info(f"  Training report saved to: {report_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
