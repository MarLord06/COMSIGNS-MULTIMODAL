#!/usr/bin/env python3
"""Test semantic resolver with real data."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.semantic import create_semantic_resolver

def main():
    # Load with real data
    resolver = create_semantic_resolver(
        "experiments/run_20260122_010532/class_mapping.json",
        "../data/raw/lsp_aec/dict.json"
    )
    
    # Show statistics
    stats = resolver.loader.statistics
    print("=== Mapping Statistics ===")
    print(f"Original classes: {stats.num_classes_original}")
    print(f"Remapped classes: {stats.num_classes_remapped}")
    print(f"HEAD classes: {stats.head_count}")
    print(f"MID classes: {stats.mid_count}")
    print(f"TAIL (collapsed to OTHER): {stats.tail_count}")
    print(f"OTHER class ID: {stats.other_class_id}")
    
    print()
    print("=== Sample Resolutions ===")
    
    # Resolve some example predictions
    for new_id in [0, 28, 100, 141]:
        pred = resolver.resolve(new_id, 0.95)
        print(f"  new_class_id={new_id} -> {pred.bucket}/{pred.gloss} (old_id={pred.old_class_id})")
    
    print()
    print("=== TopK Example ===")
    class_ids = [28, 50, 141]
    scores = [0.45, 0.30, 0.15]
    topk = resolver.resolve_topk(class_ids, scores)
    for i, p in enumerate(topk.predictions):
        print(f"  {i+1}. {p.gloss} ({p.confidence:.0%}) - {p.bucket}")
    
    print()
    print("=== All Available Glosses (first 20) ===")
    glosses = resolver.get_all_glosses()
    for i, (new_id, gloss) in enumerate(sorted(glosses.items())[:20]):
        print(f"  {new_id}: {gloss}")

if __name__ == "__main__":
    main()
