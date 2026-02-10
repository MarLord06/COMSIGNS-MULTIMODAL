#!/usr/bin/env python3
"""
ComSigns v1 Training Pipeline

A complete pipeline for training a minimal but precise sign language model:
1. Semantic gloss cleaning (remove ???, deletreos, invalid prefixes)
2. Low-support filtering
3. TAIL → OTHER consolidation
4. Rebalancing with augmentation
5. Class-weighted training
6. Advanced metrics (learned words, rejection, composite score)
7. Checkpointing with best model selection

Usage:
    python -m scripts.train_v1 --config config.yaml
    
    # Or with CLI arguments
    python -m scripts.train_v1 \
        --split-file data/splits/aec_stratified.json \
        --min-support 3 \
        --head-threshold 10 \
        --epochs 50
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
COMSIGNS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = COMSIGNS_ROOT.parent
sys.path.insert(0, str(COMSIGNS_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.data.datasets.aec import AECDataset
from core.data.loaders import encoder_collate_fn
from services.encoder import MultimodalEncoder
from training import (
    Trainer, 
    TrainerConfig, 
    SignLanguageClassifier,
    RemapConfig,
    ClassRemapper,
    RemappedDataset,
    compute_class_support,
    CheckpointManager,
    AugmentConfig,
    KeypointAugmenter,
    RebalanceConfig,
    RebalancedDataset,
    GlossCleaner,
    CleanedGlossDataset,
    LearnedWordCriteria,
    AdvancedMetricsCalculator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ComSigns v1 Training Pipeline - Minimal but Precise Model"
    )
    
    # Data paths
    parser.add_argument(
        '--dataset-path',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'raw' / 'lsp_aec',
        help='Path to dataset'
    )
    parser.add_argument(
        '--split-file',
        type=Path,
        default=None,
        help='Path to stratified split file'
    )
    parser.add_argument(
        '--dict-path',
        type=Path,
        default=None,
        help='Path to dict.json (default: dataset-path/dict.json)'
    )
    
    # Training params
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    # Cleaning params
    parser.add_argument('--min-support', type=int, default=3, help='Minimum samples per class')
    parser.add_argument('--clean-glosses', action='store_true', default=True,
                       help='Remove invalid glosses (???, deletreos, etc.)')
    parser.add_argument('--no-clean-glosses', dest='clean_glosses', action='store_false')
    
    # TAIL → OTHER params
    parser.add_argument('--head-threshold', type=int, default=10,
                       help='Minimum support for HEAD bucket')
    parser.add_argument('--enable-other', action='store_true', default=True,
                       help='Enable TAIL → OTHER consolidation')
    parser.add_argument('--no-enable-other', dest='enable_other', action='store_false')
    
    # Rebalancing params
    parser.add_argument('--rebalance', action='store_true', default=True)
    parser.add_argument('--no-rebalance', dest='rebalance', action='store_false')
    parser.add_argument('--other-max-multiplier', type=float, default=2.0)
    
    # Augmentation params
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--augment-noise-std', type=float, default=0.01)
    parser.add_argument('--augment-time-shift', type=int, default=2)
    parser.add_argument('--augment-mirror-prob', type=float, default=0.3)
    
    # Loss params
    parser.add_argument('--class-weighting', action='store_true', default=True)
    parser.add_argument('--no-class-weighting', dest='class_weighting', action='store_false')
    parser.add_argument('--other-penalty', type=float, default=1.5)
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor (0=none, 0.1=recommended)')
    
    # Regularization params
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for model')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Stop if no improvement for N epochs (0=disable)')
    parser.add_argument('--lr-scheduler', choices=['none', 'plateau', 'cosine'], default='plateau',
                       help='Learning rate scheduler')
    parser.add_argument('--lr-patience', type=int, default=5,
                       help='LR scheduler patience (for plateau)')
    parser.add_argument('--lr-factor', type=float, default=0.5,
                       help='LR reduction factor')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                       help='Warmup epochs (gradual LR increase)')
    
    # Learned words criteria
    parser.add_argument('--learned-min-precision', type=float, default=0.6)
    parser.add_argument('--learned-min-recall', type=float, default=0.5)
    parser.add_argument('--learned-min-f1', type=float, default=0.5)
    
    # Output
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--eval', action='store_true', default=True)
    
    return parser.parse_args()


def setup_device(device: str) -> torch.device:
    """Setup and return the appropriate device."""
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device)


def create_weighted_loss(
    class_counts: Dict[int, int],
    num_classes: int,
    other_class_id: Optional[int],
    other_penalty: float,
    device: torch.device,
    label_smoothing: float = 0.0
) -> nn.Module:
    """Create class-weighted loss function with optional label smoothing."""
    total_samples = sum(class_counts.values())
    weights = torch.ones(num_classes, dtype=torch.float32)
    
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count > 0:
            weights[class_id] = total_samples / (num_classes * count)
    
    # Apply OTHER penalty
    if other_class_id is not None and other_penalty != 1.0:
        weights[other_class_id] *= other_penalty
        logger.info(f"  OTHER penalty applied: weight *= {other_penalty}")
    
    if label_smoothing > 0:
        logger.info(f"  Label smoothing: {label_smoothing}")
    
    weights = weights.to(device)
    
    # MPS-compatible weighted cross entropy with label smoothing
    if device.type == 'mps':
        class WeightedCEWithSmoothing(nn.Module):
            def __init__(self, weight, num_classes, smoothing):
                super().__init__()
                self.register_buffer('weight', weight)
                self.num_classes = num_classes
                self.smoothing = smoothing
            
            def forward(self, logits, targets):
                log_probs = torch.log_softmax(logits, dim=1)
                
                if self.smoothing > 0:
                    # Create smoothed labels
                    smooth_labels = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
                    smooth_labels.scatter_(1, targets.view(-1, 1), 1.0 - self.smoothing)
                    # Weighted cross entropy with smooth labels
                    nll = -(smooth_labels * log_probs).sum(dim=1)
                    nll = nll * self.weight[targets]
                else:
                    nll = -log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
                    nll = nll * self.weight[targets]
                return nll.mean()
        
        return WeightedCEWithSmoothing(weights, num_classes, label_smoothing)
    else:
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def main():
    args = parse_args()
    
    # Banner
    logger.info("=" * 70)
    logger.info("ComSigns v1 Training Pipeline")
    logger.info("Objetivo: Modelo mínimo pero preciso")
    logger.info("=" * 70)
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = setup_device(args.device)
    logger.info(f"Device: {device}")
    
    # Paths
    dict_path = args.dict_path or (args.dataset_path / 'dict.json')
    if not dict_path.exists():
        logger.error(f"dict.json not found: {dict_path}")
        return 1
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = COMSIGNS_ROOT / 'experiments' / f'v1_run_{timestamp}'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")
    
    # =========================================================================
    # PHASE 1: Load and Clean Dataset
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: SEMANTIC CLEANING")
    logger.info("=" * 70)
    
    # Load datasets
    if args.split_file:
        logger.info(f"Loading stratified split: {args.split_file}")
        train_dataset = AECDataset(
            dataset_root=args.dataset_path,
            split_file=args.split_file,
            split='train'
        )
        val_dataset = AECDataset(
            dataset_root=args.dataset_path,
            split_file=args.split_file,
            split='val'
        )
    else:
        logger.info(f"Loading full dataset: {args.dataset_path}")
        full_dataset = AECDataset(dataset_root=args.dataset_path)
        # Create random split
        from core.data.splits import create_train_val_split
        train_dataset, val_dataset = create_train_val_split(full_dataset, val_ratio=0.2)
    
    logger.info(f"  Raw train: {len(train_dataset)} samples")
    logger.info(f"  Raw val: {len(val_dataset)} samples")
    
    # Clean glosses
    if args.clean_glosses:
        logger.info("\nCleaning invalid glosses...")
        cleaner = GlossCleaner(dict_path=dict_path)
        
        train_dataset = CleanedGlossDataset(train_dataset, cleaner, remap_ids=True)
        val_dataset = CleanedGlossDataset(val_dataset, cleaner, remap_ids=False)
        
        # Save cleaning report
        cleaning_report = train_dataset.cleaning_report
        cleaning_report.save(output_dir / 'gloss_cleaning_report.json')
        logger.info(cleaning_report.get_summary())
    
    # Filter low-support classes
    if args.min_support > 1:
        logger.info(f"\nFiltering classes with support < {args.min_support}...")
        from training import LowSupportFilter
        
        allowed_ids, support_counts = LowSupportFilter.filter_by_min_support(
            train_dataset, 
            min_support=args.min_support
        )
        
        original_classes = len(train_dataset.gloss_to_id)
        
        from training import FilteredGlossDataset
        train_dataset = FilteredGlossDataset(train_dataset, allowed_ids, remap_ids=True)
        val_dataset = FilteredGlossDataset(val_dataset, allowed_ids, remap_ids=True)
        
        logger.info(f"  Classes: {original_classes} → {len(allowed_ids)}")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples: {len(val_dataset)}")
    
    num_classes = len(train_dataset.gloss_to_id)
    class_names = list(train_dataset.gloss_to_id.keys())
    logger.info(f"\nAfter cleaning: {num_classes} classes, {len(train_dataset)} train samples")
    
    # =========================================================================
    # PHASE 2: TAIL → OTHER Consolidation
    # =========================================================================
    remapper = None
    other_class_id = None
    bucket_mapping = {}
    
    if args.enable_other:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: TAIL → OTHER CONSOLIDATION")
        logger.info("=" * 70)
        
        # Compute support
        train_support = compute_class_support(train_dataset)
        
        # Configure remapper
        remap_config = RemapConfig(
            strategy="tail_to_other",
            head_threshold=args.head_threshold,
            mid_range=(3, args.head_threshold - 1),
            other_class_name="OTHER"
        )
        
        remapper = ClassRemapper(remap_config)
        remapper.fit(train_support, class_names=train_dataset.id_to_gloss)
        
        # Apply remapping
        train_dataset = RemappedDataset(train_dataset, remapper)
        val_dataset = RemappedDataset(val_dataset, remapper)
        
        num_classes = remapper.num_classes_remapped
        other_class_id = remapper.other_class_id
        class_names = [remapper.new_class_names.get(i, f"class_{i}") for i in range(num_classes)]
        
        # Build bucket mapping
        for new_id in range(num_classes):
            try:
                bucket = remapper.get_new_class_bucket(new_id)
                bucket_mapping[new_id] = bucket.value
            except:
                bucket_mapping[new_id] = "OTHER" if new_id == other_class_id else "UNKNOWN"
        
        # Save mappings
        remapper.save(output_dir / 'class_mapping.json')
        with open(output_dir / 'new_class_names.json', 'w') as f:
            json.dump(remapper.new_class_names, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  HEAD: {sum(1 for b in bucket_mapping.values() if b == 'HEAD')} classes")
        logger.info(f"  MID: {sum(1 for b in bucket_mapping.values() if b == 'MID')} classes")
        logger.info(f"  OTHER: 1 class (id={other_class_id})")
        logger.info(f"  Total: {num_classes} classes")
    else:
        # No remapping, build bucket mapping from support
        train_support = compute_class_support(train_dataset)
        for class_id, support in train_support.items():
            if support >= args.head_threshold:
                bucket_mapping[class_id] = "HEAD"
            elif support >= 3:
                bucket_mapping[class_id] = "MID"
            else:
                bucket_mapping[class_id] = "TAIL"
    
    # =========================================================================
    # PHASE 3: REBALANCING + AUGMENTATION
    # =========================================================================
    augmenter = None
    
    if args.rebalance or args.augment:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: REBALANCING & AUGMENTATION")
        logger.info("=" * 70)
        
        if args.augment:
            aug_config = AugmentConfig(
                time_shift=args.augment_time_shift,
                noise_std=args.augment_noise_std,
                mirror_prob=args.augment_mirror_prob
            )
            augmenter = KeypointAugmenter(aug_config)
            logger.info(f"  Augmentation: noise_std={aug_config.noise_std}, "
                       f"time_shift={aug_config.time_shift}, mirror_prob={aug_config.mirror_prob}")
        
        if args.rebalance:
            rebalance_config = RebalanceConfig(
                target_strategy="median",
                other_max_multiplier=args.other_max_multiplier
            )
            
            class_counts = compute_class_support(train_dataset)
            train_dataset = RebalancedDataset(
                train_dataset,
                other_class_id=other_class_id,
                class_counts=class_counts,
                augmenter=augmenter,
                config=rebalance_config
            )
            logger.info(f"  Rebalanced: {len(train_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=encoder_collate_fn,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=encoder_collate_fn,
        num_workers=0
    )
    
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    
    # =========================================================================
    # PHASE 4: MODEL + TRAINING CONFIG
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: MODEL SETUP")
    logger.info("=" * 70)
    
    # Create model
    encoder = MultimodalEncoder(
        hand_input_dim=168,
        body_input_dim=132,
        face_input_dim=1872,
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
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {total_params:,}")
    logger.info(f"  Classes: {num_classes}")
    
    # Create loss function
    loss_fn = None
    if args.class_weighting:
        logger.info("\n  Creating class-weighted loss...")
        final_counts = compute_class_support(train_dataset)
        loss_fn = create_weighted_loss(
            final_counts, 
            num_classes, 
            other_class_id, 
            args.other_penalty,
            device,
            label_smoothing=args.label_smoothing
        )
    
    # Trainer config with regularization
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  LR scheduler: {args.lr_scheduler}")
    logger.info(f"  Early stopping patience: {args.early_stopping_patience}")
    
    config = TrainerConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        device=str(device),
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        seed=args.seed,
        validate=True,
        weight_decay=args.weight_decay
    )
    
    trainer = Trainer(model, config, num_classes=num_classes, loss_fn=loss_fn)
    
    # Setup LR scheduler
    scheduler = None
    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer, mode='min', factor=args.lr_factor,
            patience=args.lr_patience
        )
        logger.info(f"  ReduceLROnPlateau: factor={args.lr_factor}, patience={args.lr_patience}")
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        logger.info(f"  CosineAnnealingLR: T_max={args.epochs}")
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(output_dir=output_dir, keep_last_n=3)
    
    # =========================================================================
    # PHASE 5: TRAINING LOOP
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: TRAINING")
    logger.info("=" * 70)
    
    # Learned words criteria
    learned_criteria = LearnedWordCriteria(
        min_support=args.min_support,
        min_precision=args.learned_min_precision,
        min_recall=args.learned_min_recall,
        min_f1=args.learned_min_f1
    )
    
    best_score = -float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Custom epoch callback for checkpointing with composite score
    def epoch_callback(epoch: int, model: nn.Module, optimizer, metrics: dict):
        nonlocal best_score, best_epoch, epochs_without_improvement
        
        val_loss = metrics.get("val_loss", 0.0)
        
        # Step LR scheduler
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"  LR: {current_lr:.2e}")
        
        # Compute advanced metrics on validation
        metrics_calc = AdvancedMetricsCalculator(
            num_classes=num_classes,
            class_names=class_names,
            bucket_mapping=bucket_mapping,
            other_class_id=other_class_id,
            learned_criteria=learned_criteria
        )
        
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                hand = batch["hand"].to(device)
                body = batch["body"].to(device)
                face = batch["face"].to(device)
                lengths = batch["lengths"].to(device)
                labels = batch["labels"].numpy()
                
                logits = model(hand, body, face, lengths)
                preds = logits.argmax(dim=1).cpu().numpy()
                logits_np = logits.cpu().numpy()
                
                metrics_calc.update(preds, labels, logits_np)
        
        learned_report = metrics_calc.compute_learned_words()
        composite = metrics_calc.compute_composite_score(max_possible_words=num_classes)
        
        logger.info(
            f"  → Learned: {learned_report.learned_count} words | "
            f"Composite: {composite.score:.4f}"
        )
        
        # Save best model using proper CheckpointManager API
        best_metrics = {
            "epoch": epoch,
            "val_loss": val_loss,
            "f1_macro": metrics.get("val_f1", 0.0),
            "learned_words_count": learned_report.learned_count,
            "composite_score": composite.score
        }
        
        if composite.score > best_score:
            best_score = composite.score
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_manager.save_best(
                model=model,
                metrics=best_metrics,
                optimizer=optimizer
            )
            logger.info(f"  → New best model saved!")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0:
                logger.info(f"  No improvement for {epochs_without_improvement}/{args.early_stopping_patience} epochs")
                if epochs_without_improvement >= args.early_stopping_patience:
                    logger.info(f"  ⚠️ Early stopping triggered!")
                    raise StopIteration("Early stopping")
    
    # Train using the Trainer's fit method
    try:
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            class_names=class_names,
            epoch_end_callback=epoch_callback,
            start_epoch=1
        )
    except StopIteration:
        logger.info("Training stopped early due to no improvement")
    
    # =========================================================================
    # PHASE 6: FINAL EVALUATION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: FINAL EVALUATION")
    logger.info("=" * 70)
    
    # Load best model
    best_checkpoint = checkpoint_manager.load_best()
    if best_checkpoint:
        model.load_state_dict(best_checkpoint["model_state"])
        logger.info(f"Loaded best model from epoch {best_epoch}")
    
    # Final metrics
    final_metrics = AdvancedMetricsCalculator(
        num_classes=num_classes,
        class_names=class_names,
        bucket_mapping=bucket_mapping,
        other_class_id=other_class_id,
        learned_criteria=learned_criteria
    )
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            hand = batch["hand"].to(device)
            body = batch["body"].to(device)
            face = batch["face"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].numpy()
            
            logits = model(hand, body, face, lengths)
            preds = logits.argmax(dim=1).cpu().numpy()
            logits_np = logits.cpu().numpy()
            
            final_metrics.update(preds, labels, logits_np)
    
    # Generate reports
    learned_report = final_metrics.compute_learned_words()
    rejection_metrics = final_metrics.compute_rejection_metrics()
    bucket_metrics = final_metrics.compute_bucket_metrics()
    composite = final_metrics.compute_composite_score(max_possible_words=num_classes)
    
    # Save reports
    learned_report.save(output_dir / 'learned_words_report.json')
    rejection_metrics.save(output_dir / 'rejection_metrics.json')
    
    with open(output_dir / 'bucket_metrics.json', 'w') as f:
        bucket_data = {}
        for bucket, bm in bucket_metrics.items():
            bucket_data[bucket] = {
                "num_classes": bm.num_classes,
                "total_support": bm.total_support,
                "accuracy_at_1": round(bm.accuracy_at_1, 4),
                "macro_f1": round(bm.macro_f1, 4),
                "coverage": round(bm.coverage, 4)
            }
        json.dump(bucket_data, f, indent=2)
    
    # Training summary
    summary = {
        "completed_at": datetime.now().isoformat(),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "min_support": args.min_support,
            "head_threshold": args.head_threshold,
            "clean_glosses": args.clean_glosses,
            "enable_other": args.enable_other,
            "rebalance": args.rebalance,
            "augment": args.augment,
            "class_weighting": args.class_weighting,
            "other_penalty": args.other_penalty
        },
        "dataset": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "num_classes": num_classes
        },
        "results": {
            "best_epoch": best_epoch,
            "best_composite_score": round(best_score, 4),
            "learned_words_count": learned_report.learned_count,
            "pct_vocabulary_learned": round(learned_report.pct_vocabulary_learned, 4),
            "pct_predictions_other": round(rejection_metrics.pct_predictions_other, 4),
            "false_accept_rate": round(rejection_metrics.false_accept_rate, 4),
            "false_reject_rate": round(rejection_metrics.false_reject_rate, 4)
        },
        "learned_criteria": learned_criteria.to_dict(),
        "composite_score": composite.to_dict()
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(learned_report.get_summary())
    logger.info(f"\nRejection Metrics:")
    logger.info(f"  Predictions → OTHER: {rejection_metrics.pct_predictions_other:.1%}")
    logger.info(f"  False Accept Rate: {rejection_metrics.false_accept_rate:.1%}")
    logger.info(f"  False Reject Rate: {rejection_metrics.false_reject_rate:.1%}")
    logger.info(f"\nComposite Score: {composite.score:.4f}")
    logger.info(f"\nArtifacts saved to: {output_dir}")
    logger.info("  - best.pt")
    logger.info("  - class_mapping.json")
    logger.info("  - new_class_names.json")
    logger.info("  - training_summary.json")
    logger.info("  - learned_words_report.json")
    logger.info("  - rejection_metrics.json")
    logger.info("  - bucket_metrics.json")
    logger.info("  - gloss_cleaning_report.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
