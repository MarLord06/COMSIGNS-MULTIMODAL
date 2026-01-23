#!/usr/bin/env python3
"""
Minimal inference script for ComSigns.

Loads a trained model and runs inference on preprocessed keypoints.

Usage:
    python scripts/infer.py \
      --checkpoint experiments/run_xxx/checkpoints/best.pt \
      --input path/to/sample.pkl \
      --topk 5

    # With explicit class mapping
    python scripts/infer.py \
      --checkpoint experiments/run_xxx/checkpoints/best.pt \
      --class-mapping experiments/run_xxx/class_mapping.json \
      --input path/to/sample.pkl

    # Run on GPU
    python scripts/infer.py \
      --checkpoint experiments/run_xxx/checkpoints/best.pt \
      --input path/to/sample.pkl \
      --device cuda

Output:
    ==================================================
    Inference Result
    ==================================================
    Top-1 Prediction:
      Class ID   : 26
      Class Name : YO
      Score      : 0.62

    Top-5 Predictions:
      [1] YO              (0.6234)
      [2] TU              (0.1423)
      [3] EL              (0.0912)
      [4] NOSOTROS        (0.0734)
      [5] OTHER           (0.0421)

    Prediction Summary:
      Is OTHER: False
    ==================================================
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import InferenceLoader, Predictor
from inference.types import PredictionResult


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on preprocessed keypoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python scripts/infer.py --checkpoint experiments/run_001/checkpoints/best.pt --input sample.pkl

  # With more top-k predictions
  python scripts/infer.py --checkpoint best.pt --input sample.pkl --topk 10

  # Output as JSON
  python scripts/infer.py --checkpoint best.pt --input sample.pkl --json
        """
    )
    
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    
    parser.add_argument(
        "--class-mapping", "-m",
        type=Path,
        default=None,
        help="Path to class_mapping.json (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input file (.pkl with keypoints or .pt tensor)"
    )
    
    parser.add_argument(
        "--topk", "-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: cpu)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON instead of formatted text"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including model info"
    )
    
    return parser.parse_args()


def load_input_file(input_path: Path) -> dict:
    """Load input keypoints from file.
    
    Supports:
    - .pkl files with dict {"hand": tensor, "body": tensor, "face": tensor}
    - .pt files with saved tensors
    - .json files with lists (converted to tensors)
    
    Args:
        input_path: Path to input file
    
    Returns:
        Dict with "hand", "body", "face" tensors and optional "lengths"
    """
    suffix = input_path.suffix.lower()
    
    if suffix == ".pkl":
        logger.info(f"Loading pickle file: {input_path}")
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(data, dict):
            # Expected format: {"hand": ..., "body": ..., "face": ...}
            features = {}
            
            for key in ["hand", "body", "face"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, torch.Tensor):
                        features[key] = val
                    else:
                        features[key] = torch.tensor(val, dtype=torch.float32)
                else:
                    raise KeyError(f"Missing required key '{key}' in pickle file")
            
            if "lengths" in data:
                features["lengths"] = torch.tensor(data["lengths"])
            
            return features
        else:
            raise ValueError(
                f"Unexpected pickle format. Expected dict with 'hand', 'body', 'face' keys. "
                f"Got {type(data)}"
            )
    
    elif suffix == ".pt":
        logger.info(f"Loading PyTorch tensor file: {input_path}")
        data = torch.load(input_path, map_location="cpu", weights_only=False)
        
        if isinstance(data, dict):
            return data
        elif isinstance(data, torch.Tensor):
            raise ValueError(
                "Got single tensor. Expected dict with 'hand', 'body', 'face' keys."
            )
        else:
            raise ValueError(f"Unexpected .pt format: {type(data)}")
    
    elif suffix == ".json":
        logger.info(f"Loading JSON file: {input_path}")
        with open(input_path, "r") as f:
            data = json.load(f)
        
        features = {}
        for key in ["hand", "body", "face"]:
            if key in data:
                features[key] = torch.tensor(data[key], dtype=torch.float32)
            else:
                raise KeyError(f"Missing required key '{key}' in JSON file")
        
        if "lengths" in data:
            features["lengths"] = torch.tensor(data["lengths"])
        
        return features
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .pkl, .pt, or .json")


def validate_features(features: dict) -> None:
    """Validate feature tensors have correct shape.
    
    Expected shapes:
    - hand: [T, 168] or [1, T, 168] (21 keypoints * 4 values * 2 hands)
    - body: [T, 132] or [1, T, 132] (33 keypoints * 4 values)
    - face: [T, 1872] or [1, T, 1872] (468 keypoints * 4 values)
    """
    required_keys = ["hand", "body", "face"]
    
    for key in required_keys:
        if key not in features:
            raise ValueError(f"Missing required feature: {key}")
        
        tensor = features[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Feature '{key}' must be a tensor, got {type(tensor)}")
        
        if tensor.dim() not in [2, 3]:
            raise ValueError(
                f"Feature '{key}' must be 2D [T, D] or 3D [B, T, D], "
                f"got shape {tensor.shape}"
            )
    
    # Check sequence lengths match
    hand_t = features["hand"].shape[-2] if features["hand"].dim() == 3 else features["hand"].shape[0]
    body_t = features["body"].shape[-2] if features["body"].dim() == 3 else features["body"].shape[0]
    face_t = features["face"].shape[-2] if features["face"].dim() == 3 else features["face"].shape[0]
    
    if not (hand_t == body_t == face_t):
        raise ValueError(
            f"Sequence lengths don't match: hand={hand_t}, body={body_t}, face={face_t}"
        )
    
    logger.info(f"  Features validated: T={hand_t} frames")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # =====================================================================
        # 1. Load Model
        # =====================================================================
        logger.info("=" * 60)
        logger.info("Loading model...")
        logger.info("=" * 60)
        
        loader = InferenceLoader(
            checkpoint_path=args.checkpoint,
            class_mapping_path=args.class_mapping,
            device=args.device
        )
        
        model, class_names, other_class_id = loader.load_all()
        model_info = loader.model_info
        
        if args.verbose:
            logger.info(f"  Checkpoint: {args.checkpoint}")
            logger.info(f"  Epoch: {model_info.epoch}")
            logger.info(f"  Classes: {model_info.num_classes}")
            logger.info(f"  TAIL→OTHER: {model_info.tail_to_other}")
            if model_info.tail_to_other:
                logger.info(f"  OTHER class ID: {other_class_id}")
            logger.info(f"  Device: {args.device}")
        
        # =====================================================================
        # 2. Load Input
        # =====================================================================
        logger.info("\nLoading input...")
        
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        features = load_input_file(args.input)
        validate_features(features)
        
        # =====================================================================
        # 3. Run Inference
        # =====================================================================
        logger.info("\nRunning inference...")
        
        predictor = Predictor(
            model=model,
            class_names=class_names,
            other_class_id=other_class_id,
            device=args.device,
            topk=args.topk
        )
        
        result = predictor.predict_from_features(features)
        
        # =====================================================================
        # 4. Output Results
        # =====================================================================
        if args.json:
            # JSON output
            output = result.to_dict()
            output["model_info"] = model_info.to_dict()
            print(json.dumps(output, indent=2))
        else:
            # Formatted text output
            print("\n" + result.format_output())
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except KeyError as e:
        logger.error(f"Missing key in input data: {e}")
        return 1
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
