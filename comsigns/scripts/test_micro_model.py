#!/usr/bin/env python3
"""
Quick test script for micro_v1 model.

Tests the trained model on samples from the validation set.

Usage:
    python scripts/test_micro_model.py
    python scripts/test_micro_model.py --samples 10
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data.datasets.aec import AECDataset
from services.encoder.model import MultimodalEncoder
from training.classifier import SignLanguageClassifier


def load_model(checkpoint_path: Path, class_mapping_path: Path, device: str = "cpu"):
    """Load the trained model and class mapping."""
    
    # Load class mapping
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
    
    num_classes = class_mapping["config"]["vocabulary_size"]
    new_class_names = class_mapping["new_class_names"]
    
    print(f"Loaded class mapping: {num_classes} classes")
    print(f"  Classes: {list(new_class_names.values())}")
    
    # Create model
    encoder = MultimodalEncoder()
    model = SignLanguageClassifier(encoder=encoder, num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    
    return model, new_class_names


def predict_sample(model, sample, class_names, device="cpu"):
    """Run prediction on a single sample."""
    
    # Prepare tensors
    hand = torch.tensor(sample.hand_keypoints, dtype=torch.float32).unsqueeze(0).to(device)
    body = torch.tensor(sample.body_keypoints, dtype=torch.float32).unsqueeze(0).to(device)
    face = torch.tensor(sample.face_keypoints, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(hand, body, face)
        probs = torch.softmax(logits, dim=-1)
        pred_class = probs.argmax(dim=-1).item()
        confidence = probs[0, pred_class].item()
    
    pred_name = class_names.get(str(pred_class), f"UNK_{pred_class}")
    
    return {
        "predicted_class": pred_class,
        "predicted_name": pred_name,
        "confidence": confidence,
        "true_gloss": sample.gloss,
        "correct": pred_name.lower() == sample.gloss.lower(),
        "all_probs": {class_names[str(i)]: probs[0, i].item() for i in range(len(class_names))}
    }


def main():
    parser = argparse.ArgumentParser(description="Test micro_v1 model")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "experiments" / "micro_v1" / "best.pt")
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT / "experiments" / "micro_v1" / "class_mapping.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    
    print("=" * 60)
    print("COMSIGNS MICRO_V1 MODEL TEST")
    print("=" * 60)
    
    # Load model
    model, class_names = load_model(args.model, args.mapping, device)
    vocab_words = set(name.lower() for name in class_names.values())
    
    # Load validation dataset
    dataset_root = PROJECT_ROOT.parent / "data" / "raw" / "lsp_aec"
    split_file = PROJECT_ROOT.parent / "data" / "splits" / "aec_stratified.json"
    
    print(f"\nLoading validation set from {dataset_root}...")
    
    val_dataset = AECDataset(
        dataset_root=dataset_root,
        split_file=split_file,
        split="val",
        skip_missing=True
    )
    
    # Filter to only micro-vocab samples
    micro_samples = []
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        if sample.gloss.lower() in vocab_words:
            micro_samples.append(sample)
    
    print(f"Found {len(micro_samples)} validation samples in micro-vocab")
    
    # Test samples
    print(f"\nTesting {min(args.samples, len(micro_samples))} samples:")
    print("-" * 60)
    
    correct = 0
    total = 0
    results_by_class = {name: {"correct": 0, "total": 0} for name in vocab_words}
    
    for i, sample in enumerate(micro_samples[:args.samples]):
        result = predict_sample(model, sample, class_names, device)
        
        status = "✓" if result["correct"] else "✗"
        print(f"{i+1:2d}. {status} True: {result['true_gloss']:8s} → Pred: {result['predicted_name']:8s} ({result['confidence']:.2%})")
        
        if result["correct"]:
            correct += 1
        total += 1
        
        true_class = result["true_gloss"].lower()
        if true_class in results_by_class:
            results_by_class[true_class]["total"] += 1
            if result["correct"]:
                results_by_class[true_class]["correct"] += 1
    
    # Summary
    print("-" * 60)
    print(f"\nAccuracy: {correct}/{total} ({correct/total:.1%})")
    
    print("\nPer-class accuracy:")
    for cls, stats in results_by_class.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"  {cls:8s}: {stats['correct']}/{stats['total']} ({acc:.1%})")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
