# Tail Strategy Reference

> **IMPORTANT**: This document defines tail handling strategies for **REFERENCE ONLY**.  
> These strategies are **NOT IMPLEMENTED YET** - they serve as a decision framework  
> based on bucket analysis results.

## Context

In long-tail classification (like sign language with 505 glosses), classes are unevenly distributed:

| Bucket | Definition | Typical % of Classes | Typical % of Samples |
|--------|-----------|---------------------|---------------------|
| HEAD   | ≥ 10 samples | ~5-10% | ~30-50% |
| MID    | 3-9 samples | ~20-30% | ~30-40% |
| TAIL   | 1-2 samples | ~60-70% | ~15-25% |

The model learns HEAD classes well, struggles with MID, and often fails on TAIL.

---

## Strategy 1: TAIL → OTHER

### Description
Merge all TAIL classes (support ≤ 2) into a single "OTHER" class.

### Implementation Sketch (NOT IMPLEMENTED)
```python
# Pseudo-code only
def merge_tail_to_other(dataset, tail_class_ids, other_id):
    for sample in dataset:
        if sample.gloss_id in tail_class_ids:
            sample.gloss_id = other_id
    return dataset
```

### Pros
- **Simplifies model**: Reduces num_classes significantly (e.g., 505 → 200)
- **Improves HEAD/MID accuracy**: Less confusion from underrepresented classes
- **Cleaner training signal**: Model focuses on learnable classes
- **Faster convergence**: Fewer classes = simpler decision boundary

### Cons
- **Loses granularity**: Cannot distinguish between different rare glosses
- **Heterogeneous OTHER**: "OTHER" class becomes very diverse
- **Production limitations**: May not meet requirements for full vocabulary support
- **Evaluation bias**: Metrics may look artificially better

### When to Use
- Tail Accuracy@5 < 10% (model treats tail as noise)
- Tail Coverage@5 < 15%
- Primary goal is maximizing HEAD/MID accuracy
- Vocabulary completeness is not a hard requirement

### Decision Criteria
```
IF tail_accuracy_at_5 < 0.10 AND tail_is_noise == True:
    RECOMMEND: TAIL_TO_OTHER
```

---

## Strategy 2: TAIL EXCLUSION

### Description
Train only on HEAD + MID classes, completely excluding TAIL from training and evaluation.

### Implementation Sketch (NOT IMPLEMENTED)
```python
# Pseudo-code only
def exclude_tail(dataset, tail_class_ids):
    return [s for s in dataset if s.gloss_id not in tail_class_ids]
```

### Pros
- **Cleaner training**: Only classes with sufficient representation
- **Higher reported accuracy**: Metrics reflect model capability on learnable data
- **Simpler model**: Reduced output dimension
- **Focused evaluation**: Clear signal of model performance

### Cons
- **Loses vocabulary**: May exclude 50%+ of classes
- **No tail recognition**: Model cannot handle tail glosses at all
- **Production gaps**: Real-world data may contain tail glosses
- **Not suitable for full deployment**

### When to Use
- TAIL represents > 50% of classes
- Tail Accuracy@5 < 25%
- Primary goal is research/baseline establishment
- Full vocabulary is not required for current phase

### Decision Criteria
```
IF tail_class_percentage > 50% AND tail_accuracy_at_5 < 0.25:
    RECOMMEND: TAIL_EXCLUSION
```

---

## Strategy 3: TAIL FEW-SHOT (Future)

### Description
Separate handling for TAIL classes using few-shot learning or embedding-based retrieval.

### Concept (NOT IMPLEMENTED)
1. Train main classifier on HEAD + MID only
2. For TAIL classes, use embedding similarity:
   - Compute embedding for query sample
   - Compare against stored TAIL class prototypes
   - Use nearest neighbor classification

### Implementation Sketch (NOT IMPLEMENTED)
```python
# Pseudo-code only
class TailFewShotClassifier:
    def __init__(self, encoder, prototypes):
        self.encoder = encoder  # Pretrained encoder
        self.tail_prototypes = prototypes  # Dict[class_id, embedding]
    
    def predict_tail(self, sample):
        embedding = self.encoder.encode(sample)
        distances = {
            cid: cosine_distance(embedding, proto)
            for cid, proto in self.tail_prototypes.items()
        }
        return min(distances, key=distances.get)
```

### Pros
- **Preserves full vocabulary**: All 505 classes accessible
- **Specialized handling**: Different strategies for different buckets
- **Leverages embeddings**: Works even with 1-2 samples if embeddings are good
- **Scalable**: Can add new classes without retraining main classifier

### Cons
- **Complex architecture**: Two-stage inference
- **Requires good embeddings**: Depends on encoder quality
- **Slower inference**: Extra computation for tail detection
- **Tuning complexity**: Need to determine when to use few-shot

### When to Use
- Full vocabulary is required
- Embeddings show semantic structure (verified via t-SNE/UMAP)
- Tail Accuracy@5 is too low but embeddings cluster correctly
- Production requires handling rare glosses

### Decision Criteria
```
IF full_vocabulary_required AND embeddings_show_structure:
    RECOMMEND: TAIL_FEW_SHOT
```

---

## Strategy Selection Matrix

| Condition | Recommended Strategy |
|-----------|---------------------|
| Tail Acc@5 < 10%, Tail is noise | TAIL_TO_OTHER |
| Tail > 50% classes, Acc@5 < 25% | TAIL_EXCLUSION |
| Need full vocab, good embeddings | TAIL_FEW_SHOT |
| Tail Acc@5 > 25% | KEEP_TAIL (do nothing) |

---

## How to Use This Document

1. **Run bucket analysis** first:
   ```bash
   python scripts/run_bucket_analysis.py \
       --metrics experiments/run_001/metrics_by_class.json \
       --dataset-root data/raw/lsp_aec \
       --split-file data/processed/stratified_split.json \
       --output experiments/run_001/bucket_analysis.json
   ```

2. **Review the diagnosis** in the output:
   - `tail_is_learning`: True if model shows signal on tail
   - `tail_is_noise`: True if tail is essentially random
   - `recommendation`: Suggested strategy

3. **Make decision** based on:
   - Project requirements (full vocab vs. high accuracy)
   - Current model capabilities
   - Available resources for implementation

4. **Document decision** before implementing

---

## Implementation Status

| Strategy | Status | Implementation Location |
|----------|--------|------------------------|
| TAIL_TO_OTHER | ❌ Not Implemented | - |
| TAIL_EXCLUSION | ❌ Not Implemented | - |
| TAIL_FEW_SHOT | ❌ Not Implemented | - |
| Analysis Only | ✅ Implemented | `training/analysis/bucket_analysis.py` |

---

## References

- Long-Tail Learning: A Survey (2021)
- Decoupling Representation and Classifier for Long-Tailed Recognition (2019)
- Class-Balanced Loss Based on Effective Number of Samples (2019)
