"""Analyze gloss support for micro-vocabulary selection."""

import json
from pathlib import Path
from collections import Counter

def main():
    # Cargar split
    split_path = Path("../data/splits/aec_stratified.json")
    with open(split_path) as f:
        split_data = json.load(f)
    
    train_files = split_data["train"]
    val_files = split_data["val"]
    
    # Contar samples por glosa
    train_counts = Counter()
    val_counts = Counter()
    
    for name in train_files:
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            glosa = parts[0]
            train_counts[glosa] += 1
    
    for name in val_files:
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            glosa = parts[0]
            val_counts[glosa] += 1
    
    # Mostrar top 50 por soporte en train
    print("TOP 50 GLOSAS POR SOPORTE EN TRAIN:")
    print("=" * 60)
    print("{:<30} {:>8} {:>8} {:>8}".format("GLOSA", "TRAIN", "VAL", "TOTAL"))
    print("-" * 60)
    for glosa, count in train_counts.most_common(50):
        val = val_counts.get(glosa, 0)
        print("{:<30} {:>8} {:>8} {:>8}".format(glosa, count, val, count + val))
    
    # Estadísticas
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS:")
    print(f"  Total glosas: {len(train_counts)}")
    print(f"  Glosas con train >= 10: {sum(1 for c in train_counts.values() if c >= 10)}")
    print(f"  Glosas con train >= 15: {sum(1 for c in train_counts.values() if c >= 15)}")
    print(f"  Glosas con train >= 20: {sum(1 for c in train_counts.values() if c >= 20)}")

if __name__ == "__main__":
    main()
