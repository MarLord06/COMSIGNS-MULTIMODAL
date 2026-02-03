#!/usr/bin/env python3
"""E2E test for semantic resolution."""

import torch
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 60)
    print("E2E INFERENCE TEST")
    print("=" * 60)
    
    from backend.services.inference_service import InferenceService
    
    # Create service
    service = InferenceService(
        checkpoint_path=Path("experiments/run_20260122_010532/checkpoints/best.pt"),
        class_mapping_path=Path("experiments/run_20260122_010532/class_mapping.json"),
        dict_path=Path("../data/raw/lsp_aec/dict.json"),
        device="cpu",
        lazy_load=False
    )
    
    print(f"Num classes: {service.num_classes}")
    
    # Create dummy features
    features = {
        "hand": torch.randn(1, 50, 168),
        "body": torch.randn(1, 50, 132),
        "face": torch.randn(1, 50, 1872),
    }
    
    # Run inference
    result = service.infer(features, topk=5)
    result_dict = result.to_dict()
    
    print()
    print("Top-1 prediction:")
    for k, v in result_dict["top1"].items():
        print(f"  {k}: {v}")
    
    print()
    print("Top-K predictions:")
    for pred in result_dict["topk"]:
        print(f"  Rank {pred['rank']}: {pred['gloss']} ({pred['confidence']:.4f}) - {pred['bucket']}")
    
    # Check for UNKNOWN
    unknowns = [p for p in result_dict["topk"] if "UNKNOWN" in p["gloss"]]
    if unknowns:
        print()
        print("!!! FOUND UNKNOWNS !!!")
        for u in unknowns:
            print(f"  {u}")
        return 1
    else:
        print()
        print("✓ No UNKNOWN values in predictions")
        return 0

if __name__ == "__main__":
    sys.exit(main())
