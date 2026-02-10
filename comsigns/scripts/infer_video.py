#!/usr/bin/env python3
"""
Video inference script for ComSigns micro_v1 model.

Takes a video file, extracts keypoints using MediaPipe, and runs prediction.

Usage:
    python scripts/infer_video.py --video path/to/video.mp4
    python scripts/infer_video.py --video path/to/video.mp4 --topk 3
    
Examples:
    # Test with a "comer" video
    python scripts/infer_video.py --video ../data/raw/lsp_aec/Videos/SEGMENTED_SIGN/proteinas_porcentajes/comer_1001.mp4
    
    # Test with a "dos" video
    python scripts/infer_video.py --video ../data/raw/lsp_aec/Videos/SEGMENTED_SIGN/proteinas_porcentajes/dos_1373.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.encoder.model import MultimodalEncoder
from training.classifier import SignLanguageClassifier


def extract_keypoints_from_video(video_path: Path) -> dict:
    """
    Extract keypoints from video using MediaPipe.
    
    Returns dict with hand, body, face tensors of shape [T, dim].
    """
    from services.preprocessing.extract_keypoints import KeypointExtractor
    
    print(f"Extracting keypoints from: {video_path}")
    
    # Initialize extractor
    extractor = KeypointExtractor()
    
    # Use the built-in extract_from_video method
    feature_clip = extractor.extract_from_video(str(video_path))
    
    print(f"  Extracted {len(feature_clip.frames)} frames")
    
    # Convert FeatureClip to flat arrays
    hand_keypoints = []
    body_keypoints = []
    face_keypoints = []
    
    for frame in feature_clip.frames:
        # Flatten hand keypoints (2 hands * 21 keypoints * 4 values = 168)
        hand_vec = np.zeros(168, dtype=np.float32)
        if frame.hand_keypoints:
            for i, kp in enumerate(frame.hand_keypoints[:42]):  # Max 2 hands * 21
                if len(kp) >= 4:
                    hand_vec[i*4:i*4+4] = kp[:4]
                elif len(kp) >= 3:
                    hand_vec[i*4:i*4+3] = kp[:3]
                    hand_vec[i*4+3] = 1.0
        hand_keypoints.append(hand_vec)
        
        # Flatten body keypoints (33 keypoints * 4 values = 132)
        body_vec = np.zeros(132, dtype=np.float32)
        if frame.body_keypoints:
            for i, kp in enumerate(frame.body_keypoints[:33]):
                if len(kp) >= 4:
                    body_vec[i*4:i*4+4] = kp[:4]
                elif len(kp) >= 3:
                    body_vec[i*4:i*4+3] = kp[:3]
                    body_vec[i*4+3] = 1.0
        body_keypoints.append(body_vec)
        
        # Flatten face keypoints (468 keypoints * 4 values = 1872)
        face_vec = np.zeros(1872, dtype=np.float32)
        if frame.face_keypoints:
            for i, kp in enumerate(frame.face_keypoints[:468]):
                if len(kp) >= 4:
                    face_vec[i*4:i*4+4] = kp[:4]
                elif len(kp) >= 3:
                    face_vec[i*4:i*4+3] = kp[:3]
                    face_vec[i*4+3] = 1.0
        face_keypoints.append(face_vec)
    
    return {
        "hand": np.array(hand_keypoints, dtype=np.float32),
        "body": np.array(body_keypoints, dtype=np.float32),
        "face": np.array(face_keypoints, dtype=np.float32)
    }


def load_model(checkpoint_path: Path, class_mapping_path: Path, device: str = "cpu"):
    """Load the trained model and class mapping."""
    
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
    
    num_classes = class_mapping["config"]["vocabulary_size"]
    new_class_names = class_mapping["new_class_names"]
    
    encoder = MultimodalEncoder()
    model = SignLanguageClassifier(encoder=encoder, num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, new_class_names


def predict(model, keypoints: dict, class_names: dict, device: str = "cpu", topk: int = 5):
    """Run prediction on extracted keypoints."""
    
    hand = torch.tensor(keypoints["hand"], dtype=torch.float32).unsqueeze(0).to(device)
    body = torch.tensor(keypoints["body"], dtype=torch.float32).unsqueeze(0).to(device)
    face = torch.tensor(keypoints["face"], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(hand, body, face)
        probs = torch.softmax(logits, dim=-1)
    
    # Get top-k predictions
    topk_probs, topk_indices = torch.topk(probs[0], min(topk, len(class_names)))
    
    results = []
    for i in range(len(topk_indices)):
        idx = topk_indices[i].item()
        prob = topk_probs[i].item()
        name = class_names.get(str(idx), f"UNK_{idx}")
        results.append({"class_id": idx, "class_name": name, "confidence": prob})
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference on video file")
    parser.add_argument("--video", "-v", type=Path, required=True, help="Path to video file")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "experiments" / "micro_v1" / "best.pt")
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT / "experiments" / "micro_v1" / "class_mapping.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to show")
    args = parser.parse_args()
    
    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    
    print("=" * 60)
    print("COMSIGNS VIDEO INFERENCE")
    print("=" * 60)
    
    # Extract expected label from filename if possible
    video_name = args.video.stem
    expected_label = video_name.split("_")[0] if "_" in video_name else None
    
    # Load model
    print("\nLoading model...")
    model, class_names = load_model(args.model, args.mapping, args.device)
    print(f"  Model: {args.model.name}")
    print(f"  Classes: {list(class_names.values())}")
    
    # Extract keypoints
    print("\nExtracting keypoints...")
    keypoints = extract_keypoints_from_video(args.video)
    print(f"  Hand shape: {keypoints['hand'].shape}")
    print(f"  Body shape: {keypoints['body'].shape}")
    print(f"  Face shape: {keypoints['face'].shape}")
    
    # Run prediction
    print("\nRunning prediction...")
    results = predict(model, keypoints, class_names, args.device, args.topk)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    if expected_label:
        print(f"Video: {video_name}")
        print(f"Expected: {expected_label}")
        print()
    
    print(f"Top-{args.topk} Predictions:")
    for i, r in enumerate(results, 1):
        marker = "←" if expected_label and r["class_name"].lower() == expected_label.lower() else ""
        print(f"  {i}. {r['class_name']:10s} {r['confidence']:6.2%} {marker}")
    
    # Summary
    top1 = results[0]
    correct = expected_label and top1["class_name"].lower() == expected_label.lower()
    
    print()
    print("-" * 60)
    print(f"Prediction: {top1['class_name']} ({top1['confidence']:.2%})")
    if expected_label:
        status = "✓ CORRECT" if correct else "✗ INCORRECT"
        print(f"Result: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
