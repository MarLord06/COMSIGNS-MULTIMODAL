#!/usr/bin/env python3
"""
ComSigns Micro-Vocabulary Training (Phase 1)

Objetivo: Demostrar aprendizaje real con 7 palabras (sin OTHER)
Criterio de éxito: Al menos 2-3 palabras con F1 >= 0.5

Vocabulario V1:
    - comer (57 train, 15 val) - mano a boca
    - yo (31 train, 8 val) - señalar a uno mismo
    - tú (50 train, 13 val) - señalar al otro
    - sí (36 train, 9 val) - asentimiento
    - no (22 train, 6 val) - negación
    - dos (31 train, 8 val) - configuración de dedos
    - conflicto (X train, Y val) - etiqueta adicional para ambigüedad/conflicto

Uso:
        python -m scripts.train_micro --epochs 100 --output-dir experiments/micro_v1
"""
    
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data.datasets.aec import AECDataset
from core.data.loaders.collate import encoder_collate_fn
from services.encoder.model import MultimodalEncoder
from training.classifier import SignLanguageClassifier
from training import Trainer, TrainerConfig, CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Path to original dict.json for ID mapping
DICT_JSON_PATH = PROJECT_ROOT.parent / "data" / "raw" / "lsp_aec" / "dict.json"

# ============================================================================
# MICRO-VOCABULARY DEFINITION
# ============================================================================

MICRO_VOCAB = {
    "comer": 0,
    "yo": 1,
    "tú": 2,
    "sí": 3,
    "no": 4,
    "dos": 5,
    "conflicto": 6,
}
NUM_CLASSES = 7  # Ahora 7 palabras incluyendo 'conflicto'

CLASS_NAMES = ["comer", "yo", "tú", "sí", "no", "dos", "conflicto"]


# ============================================================================
# MICRO DATASET WRAPPER
# ============================================================================

class MicroVocabDataset(Dataset):
    """Wrapper that keeps only micro-vocabulary samples (no OTHER)."""
    
    def __init__(self, base_dataset: AECDataset, vocab: Dict[str, int]):
        self.base_dataset = base_dataset
        self.vocab = vocab
        
        # Build index map - ONLY keep samples in vocab
        self.index_map = []
        self.class_counts = {i: 0 for i in range(len(vocab))}
        
        for i in range(len(base_dataset)):
            sample = base_dataset[i]
            gloss = sample.gloss
            
            if gloss in vocab:
                new_label = vocab[gloss]
                self.index_map.append((i, new_label))
                self.class_counts[new_label] += 1
            # Skip samples not in vocab (no OTHER class)
        
        logger.info(f"MicroVocabDataset: {len(self.index_map)} samples (filtered from {len(base_dataset)})")
        for name, idx in vocab.items():
            logger.info(f"  {name}: {self.class_counts[idx]} samples")
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, idx: int):
        base_idx, new_label = self.index_map[idx]
        sample = self.base_dataset[base_idx]
        # Override gloss_id with new label
        sample.gloss_id = new_label
        return sample
    
    def get_class_counts(self) -> Dict[int, int]:
        return self.class_counts


# ============================================================================
# BALANCED SAMPLER
# ============================================================================

class BalancedBatchSampler:
    """Sample balanced batches for training."""
    
    def __init__(self, dataset: MicroVocabDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Group indices by class
        self.class_indices = {i: [] for i in range(NUM_CLASSES)}
        for idx, (_, label) in enumerate(dataset.index_map):
            self.class_indices[label].append(idx)
        
        # No OTHER class - all classes are real vocabulary
        for class_id, indices in self.class_indices.items():
            logger.info(f"  Class {CLASS_NAMES[class_id]}: {len(indices)} samples")
        
        # Calculate total samples per epoch (upsample to max)
        self.samples_per_class = max(len(v) for v in self.class_indices.values())
        self.total_samples = self.samples_per_class * NUM_CLASSES
        
        logger.info(f"BalancedBatchSampler: {self.samples_per_class} samples/class, {self.total_samples} total")
    
    def __iter__(self):
        # Upsample each class to samples_per_class
        all_indices = []
        for class_id, indices in self.class_indices.items():
            if len(indices) == 0:
                continue
            # Repeat indices to match samples_per_class
            repeated = (indices * (self.samples_per_class // len(indices) + 1))[:self.samples_per_class]
            all_indices.extend(repeated)
        
        # Shuffle
        np.random.shuffle(all_indices)
        
        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i + self.batch_size]
    
    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size


# ============================================================================
# METRICS
# ============================================================================

def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
    """Compute precision, recall, F1 per class."""
    metrics = {}
    
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(np.sum(y_true == c))
        
        metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }
    
    return metrics


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str) -> Dict:
    """Full evaluation with per-class metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            hand = batch["hand"].to(device)
            body = batch["body"].to(device)
            face = batch["face"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"]
            
            logits = model(hand, body, face, lengths)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    # Per-class metrics
    per_class = compute_per_class_metrics(y_true, y_pred, NUM_CLASSES)
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    
    # Macro F1 (all classes - no OTHER to exclude)
    f1_scores = [per_class[c]["f1"] for c in range(NUM_CLASSES)]
    macro_f1 = np.mean(f1_scores)
    
    # Count learned words (F1 >= 0.5)
    learned_words = [CLASS_NAMES[c] for c in range(NUM_CLASSES) if per_class[c]["f1"] >= 0.5]
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "learned_words": learned_words,
        "learned_count": len(learned_words),
        "confusion": {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist()
        }
    }


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train micro-vocabulary model")
    parser.add_argument("--split-file", type=str, 
                        default=str(PROJECT_ROOT.parent / "data" / "splits" / "aec_stratified.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="experiments/micro_v1",
                        help="Output directory relative to comsigns/")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Make output_dir relative to PROJECT_ROOT (comsigns/)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("=" * 70)
    logger.info("COMSIGNS MICRO-VOCABULARY TRAINING (PHASE 1)")
    logger.info("=" * 70)
    logger.info(f"Vocabulary: {list(MICRO_VOCAB.keys())}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)
    
    dataset_root = PROJECT_ROOT.parent / "data" / "raw" / "lsp_aec"
    
    train_base = AECDataset(
        dataset_root=dataset_root,
        split_file=args.split_file,
        split="train"
    )
    
    val_base = AECDataset(
        dataset_root=dataset_root,
        split_file=args.split_file,
        split="val"
    )
    
    # Wrap with micro-vocab (no OTHER class)
    train_dataset = MicroVocabDataset(train_base, MICRO_VOCAB)
    val_dataset = MicroVocabDataset(val_base, MICRO_VOCAB)
    
    # Create dataloaders
    train_sampler = BalancedBatchSampler(train_dataset, args.batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=encoder_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=encoder_collate_fn,
        num_workers=0
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # =========================================================================
    # MODEL
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MODEL SETUP")
    logger.info("=" * 70)
    
    encoder = MultimodalEncoder()
    model = SignLanguageClassifier(encoder=encoder, num_classes=NUM_CLASSES)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {num_params:,}")
    
    # Class weights (inverse frequency)
    train_counts = train_dataset.get_class_counts()
    total = sum(train_counts.values())
    weights = torch.tensor([total / (NUM_CLASSES * train_counts[i]) for i in range(NUM_CLASSES)], dtype=torch.float32)
    weights = weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING")
    logger.info("=" * 70)
    
    best_score = -float("inf")
    best_epoch = 0
    best_metrics = None
    patience = 10
    patience_counter = 0
    
    history = {"train_loss": [], "val_f1": [], "learned_count": []}
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            hand = batch["hand"].to(device)
            body = batch["body"].to(device)
            face = batch["face"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(hand, body, face, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        
        # Evaluate
        metrics = evaluate_model(model, val_loader, device)
        
        history["train_loss"].append(avg_loss)
        history["val_f1"].append(metrics["macro_f1"])
        history["learned_count"].append(metrics["learned_count"])
        
        # Composite score
        score = metrics["macro_f1"] + 0.1 * metrics["learned_count"]
        
        # Logging
        learned_str = ", ".join(metrics["learned_words"]) if metrics["learned_words"] else "none"
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"F1: {metrics['macro_f1']:.4f} | "
            f"Learned: {metrics['learned_count']} ({learned_str})"
        )
        
        # Best model
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            patience_counter = 0
            
            # Save
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": metrics,
                "class_names": CLASS_NAMES
            }, output_dir / "best.pt")
            
            logger.info(f"  → New best! Saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Macro F1: {best_metrics['macro_f1']:.4f}")
    logger.info(f"Learned words: {best_metrics['learned_count']}")
    
    logger.info("\nPer-class metrics:")
    for c in range(NUM_CLASSES):
        m = best_metrics["per_class"][c]
        status = "✓" if m["f1"] >= 0.5 else "✗"
        logger.info(
            f"  {CLASS_NAMES[c]:<10} | P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} | {status}"
        )
    
    # Save summary
    summary = {
        "completed_at": datetime.now().isoformat(),
        "vocabulary": list(MICRO_VOCAB.keys()),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay
        },
        "best_epoch": best_epoch,
        "results": {
            "macro_f1": best_metrics["macro_f1"],
            "accuracy": best_metrics["accuracy"],
            "learned_words": best_metrics["learned_words"],
            "learned_count": best_metrics["learned_count"]
        },
        "per_class": {CLASS_NAMES[c]: best_metrics["per_class"][c] for c in range(NUM_CLASSES)},
        "history": history
    }
    
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save class mapping for inference (consistent format with train_v1.py)
    # Load original dict.json to get original IDs
    gloss_to_original_id = {}
    if DICT_JSON_PATH.exists():
        with open(DICT_JSON_PATH, "r", encoding="utf-8") as f:
            dict_json = json.load(f)
        for str_id, entry in dict_json.items():
            gloss_name = entry.get("gloss", "")
            gloss_to_original_id[gloss_name] = int(str_id)
        logger.info(f"Loaded {len(gloss_to_original_id)} glosses from dict.json")
    else:
        logger.warning(f"dict.json not found at {DICT_JSON_PATH}")
    
    # Build old_to_new and new_to_old mappings using original IDs
    old_to_new = {}  # original_id -> new_class_id
    new_to_old = {}  # new_class_id -> [original_ids]
    new_class_names = {}  # new_class_id -> gloss_name
    
    for gloss_name, new_id in MICRO_VOCAB.items():
        original_id = gloss_to_original_id.get(gloss_name)
        if original_id is not None:
            old_to_new[str(original_id)] = new_id
            new_to_old[str(new_id)] = [original_id]
        else:
            logger.warning(f"Gloss '{gloss_name}' not found in dict.json")
            new_to_old[str(new_id)] = []
        new_class_names[str(new_id)] = gloss_name
    
    class_mapping = {
        "config": {
            "strategy": "micro_vocabulary",
            "vocabulary_size": len(MICRO_VOCAB),
            "description": "Phase 1 micro-vocabulary for demonstrating real learning"
        },
        "old_to_new": old_to_new,
        "new_to_old": new_to_old,
        "new_class_names": new_class_names,
        "statistics": {
            "num_classes_original": len(gloss_to_original_id) if gloss_to_original_id else 506,
            "num_classes_remapped": len(MICRO_VOCAB),
            "vocabulary": CLASS_NAMES,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset)
        }
    }
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nArtifacts saved to: {output_dir}")
    
    # Success criteria
    if best_metrics["learned_count"] >= 2:
        logger.info("\n✓ SUCCESS: Model learned at least 2 words with F1 >= 0.5")
        return 0
    else:
        logger.info("\n✗ NEEDS IMPROVEMENT: Less than 2 words learned")
        return 1


if __name__ == "__main__":
    sys.exit(main())
