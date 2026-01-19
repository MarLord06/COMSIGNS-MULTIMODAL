"""
Unit tests for the encoder collate function.

Tests verify:
- Correct padding behavior for variable-length sequences
- Shape consistency across hand, body, face tensors
- Proper mask generation
- Edge cases (single sample, empty batch)
- dtype correctness
"""

import pytest
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Import the collate function
from comsigns.core.data.loaders.collate import (
    encoder_collate_fn,
    create_encoder_collate_fn,
    _pad_sequence,
    _create_attention_mask,
)


@dataclass
class MockEncoderReadySample:
    """
    Mock sample for testing collate_fn without depending on AECDataset.
    
    Mirrors the interface of EncoderReadySample exactly.
    """
    gloss: str
    gloss_id: int
    hand_keypoints: np.ndarray  # [T, 168]
    body_keypoints: np.ndarray  # [T, 132]
    face_keypoints: np.ndarray  # [T, 1872]
    unique_name: str = "test_sample"
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def num_frames(self) -> int:
        return self.hand_keypoints.shape[0]


def create_mock_sample(
    gloss: str,
    gloss_id: int,
    num_frames: int,
    hand_dim: int = 168,
    body_dim: int = 132,
    face_dim: int = 1872,
    fill_value: float = 1.0
) -> MockEncoderReadySample:
    """
    Factory function to create mock samples with specified parameters.
    
    Args:
        gloss: Gloss label
        gloss_id: Integer class ID
        num_frames: Number of temporal frames (T)
        hand_dim: Hand keypoints dimension (default 168)
        body_dim: Body keypoints dimension (default 132)
        face_dim: Face keypoints dimension (default 1872)
        fill_value: Value to fill arrays with (default 1.0)
    
    Returns:
        MockEncoderReadySample instance
    """
    return MockEncoderReadySample(
        gloss=gloss,
        gloss_id=gloss_id,
        hand_keypoints=np.full((num_frames, hand_dim), fill_value, dtype=np.float32),
        body_keypoints=np.full((num_frames, body_dim), fill_value, dtype=np.float32),
        face_keypoints=np.full((num_frames, face_dim), fill_value, dtype=np.float32),
    )


class TestPadSequence:
    """Tests for the _pad_sequence helper function."""
    
    def test_basic_padding(self):
        """Test that arrays are padded correctly to max_len."""
        arr1 = np.ones((3, 10), dtype=np.float32)
        arr2 = np.ones((5, 10), dtype=np.float32)
        
        padded = _pad_sequence([arr1, arr2], max_len=5, pad_value=0.0)
        
        assert padded.shape == (2, 5, 10)
        # First array: 3 valid frames, 2 padding frames
        assert np.all(padded[0, :3, :] == 1.0)
        assert np.all(padded[0, 3:, :] == 0.0)
        # Second array: all 5 frames valid
        assert np.all(padded[1, :, :] == 1.0)
    
    def test_custom_pad_value(self):
        """Test padding with non-zero pad value."""
        arr = np.ones((2, 10), dtype=np.float32)
        
        padded = _pad_sequence([arr], max_len=5, pad_value=-99.0)
        
        assert np.all(padded[0, :2, :] == 1.0)
        assert np.all(padded[0, 2:, :] == -99.0)
    
    def test_single_array(self):
        """Test with a single array (batch_size=1)."""
        arr = np.ones((10, 50), dtype=np.float32)
        
        padded = _pad_sequence([arr], max_len=10)
        
        assert padded.shape == (1, 10, 50)
        assert np.all(padded == 1.0)
    
    def test_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            _pad_sequence([], max_len=5)
    
    def test_dtype_preservation(self):
        """Test that dtype is correctly set."""
        arr = np.ones((3, 10), dtype=np.float64)  # Input is float64
        
        padded = _pad_sequence([arr], max_len=5, dtype=np.float32)
        
        assert padded.dtype == np.float32


class TestCreateAttentionMask:
    """Tests for the _create_attention_mask helper function."""
    
    def test_basic_mask(self):
        """Test basic mask generation."""
        mask = _create_attention_mask([3, 5, 2], max_len=5)
        
        expected = np.array([
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False]
        ])
        
        assert mask.shape == (3, 5)
        assert np.array_equal(mask, expected)
    
    def test_all_same_length(self):
        """Test when all sequences have the same length."""
        mask = _create_attention_mask([4, 4, 4], max_len=4)
        
        assert mask.shape == (3, 4)
        assert np.all(mask)  # All True
    
    def test_single_sequence(self):
        """Test with a single sequence."""
        mask = _create_attention_mask([3], max_len=5)
        
        assert mask.shape == (1, 5)
        assert np.array_equal(mask[0], [True, True, True, False, False])


class TestEncoderCollateFn:
    """Tests for the main encoder_collate_fn."""
    
    def test_basic_collation(self):
        """Test basic batch collation with variable-length sequences."""
        samples = [
            create_mock_sample("hola", 0, num_frames=10),
            create_mock_sample("gracias", 1, num_frames=15),
            create_mock_sample("adios", 2, num_frames=8),
        ]
        
        batch = encoder_collate_fn(samples)
        
        # Check shapes
        assert batch["hand"].shape == (3, 15, 168)
        assert batch["body"].shape == (3, 15, 132)
        assert batch["face"].shape == (3, 15, 1872)
        assert batch["labels"].shape == (3,)
        assert batch["lengths"].shape == (3,)
        assert batch["mask"].shape == (3, 15)
    
    def test_dtypes(self):
        """Test that output tensors have correct dtypes."""
        samples = [create_mock_sample("test", 0, num_frames=5)]
        
        batch = encoder_collate_fn(samples)
        
        assert batch["hand"].dtype == torch.float32
        assert batch["body"].dtype == torch.float32
        assert batch["face"].dtype == torch.float32
        assert batch["labels"].dtype == torch.int64
        assert batch["lengths"].dtype == torch.int64
        assert batch["mask"].dtype == torch.bool
    
    def test_labels_correct(self):
        """Test that labels are preserved correctly."""
        samples = [
            create_mock_sample("a", 5, num_frames=3),
            create_mock_sample("b", 10, num_frames=4),
            create_mock_sample("c", 2, num_frames=5),
        ]
        
        batch = encoder_collate_fn(samples)
        
        assert batch["labels"].tolist() == [5, 10, 2]
    
    def test_lengths_correct(self):
        """Test that original lengths are preserved."""
        samples = [
            create_mock_sample("a", 0, num_frames=10),
            create_mock_sample("b", 1, num_frames=5),
            create_mock_sample("c", 2, num_frames=20),
        ]
        
        batch = encoder_collate_fn(samples)
        
        assert batch["lengths"].tolist() == [10, 5, 20]
    
    def test_padding_values(self):
        """Test that padding uses correct value (0.0)."""
        sample = create_mock_sample("test", 0, num_frames=3, fill_value=5.0)
        # Add another sample to force padding
        sample2 = create_mock_sample("test2", 1, num_frames=5, fill_value=5.0)
        
        batch = encoder_collate_fn([sample, sample2])
        
        # First sample: frames 0-2 should be 5.0, frames 3-4 should be 0.0
        hand = batch["hand"]
        assert torch.all(hand[0, :3, :] == 5.0)
        assert torch.all(hand[0, 3:, :] == 0.0)
    
    def test_mask_correctness(self):
        """Test that mask correctly identifies valid positions."""
        samples = [
            create_mock_sample("a", 0, num_frames=3),
            create_mock_sample("b", 1, num_frames=5),
        ]
        
        batch = encoder_collate_fn(samples)
        mask = batch["mask"]
        
        # First sample: 3 valid, 2 padding
        assert mask[0, :3].all()
        assert not mask[0, 3:].any()
        
        # Second sample: all 5 valid
        assert mask[1, :].all()
    
    def test_single_sample_batch(self):
        """Test collation with batch_size=1."""
        sample = create_mock_sample("single", 7, num_frames=12)
        
        batch = encoder_collate_fn([sample])
        
        assert batch["hand"].shape == (1, 12, 168)
        assert batch["labels"].tolist() == [7]
        assert batch["lengths"].tolist() == [12]
    
    def test_empty_batch_raises(self):
        """Test that empty batch raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            encoder_collate_fn([])
    
    def test_without_mask(self):
        """Test collation without mask generation."""
        samples = [create_mock_sample("test", 0, num_frames=5)]
        
        batch = encoder_collate_fn(samples, include_mask=False)
        
        assert batch["mask"].numel() == 0  # Empty tensor
    
    def test_custom_dimensions(self):
        """Test that dimensions are inferred, not hardcoded."""
        # Use non-standard dimensions to verify inference
        sample = MockEncoderReadySample(
            gloss="test",
            gloss_id=0,
            hand_keypoints=np.ones((5, 100), dtype=np.float32),  # Not 168
            body_keypoints=np.ones((5, 50), dtype=np.float32),   # Not 132
            face_keypoints=np.ones((5, 200), dtype=np.float32),  # Not 1872
        )
        
        batch = encoder_collate_fn([sample])
        
        assert batch["hand"].shape == (1, 5, 100)
        assert batch["body"].shape == (1, 5, 50)
        assert batch["face"].shape == (1, 5, 200)
    
    def test_missing_attribute_raises(self):
        """Test that missing attributes raise clear error."""
        bad_sample = {"gloss": "test"}  # Dict instead of proper sample
        
        with pytest.raises(AttributeError, match="hand_keypoints"):
            encoder_collate_fn([bad_sample])


class TestCreateEncoderCollateFn:
    """Tests for the factory function."""
    
    def test_factory_creates_function(self):
        """Test that factory returns a callable."""
        fn = create_encoder_collate_fn()
        assert callable(fn)
    
    def test_factory_with_custom_pad_value(self):
        """Test factory with custom padding value."""
        fn = create_encoder_collate_fn(pad_value=-1.0)
        
        sample1 = create_mock_sample("a", 0, num_frames=3, fill_value=5.0)
        sample2 = create_mock_sample("b", 1, num_frames=5, fill_value=5.0)
        
        batch = fn([sample1, sample2])
        
        # Padding should be -1.0
        assert torch.all(batch["hand"][0, 3:, :] == -1.0)
    
    def test_factory_without_mask(self):
        """Test factory with include_mask=False."""
        fn = create_encoder_collate_fn(include_mask=False)
        
        sample = create_mock_sample("test", 0, num_frames=5)
        batch = fn([sample])
        
        assert batch["mask"].numel() == 0


class TestIntegrationWithDataLoader:
    """Integration tests demonstrating usage with DataLoader."""
    
    def test_dataloader_iteration(self):
        """Test that collate_fn works correctly with DataLoader."""
        from torch.utils.data import Dataset, DataLoader
        
        class MockDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        # Create samples with varying lengths
        samples = [
            create_mock_sample(f"gloss_{i}", i, num_frames=np.random.randint(5, 20))
            for i in range(10)
        ]
        
        dataset = MockDataset(samples)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=encoder_collate_fn
        )
        
        # Iterate and validate batches
        total_samples = 0
        for batch in loader:
            batch_size = batch["hand"].shape[0]
            total_samples += batch_size
            
            # Validate shapes are consistent
            T_max = batch["hand"].shape[1]
            assert batch["body"].shape[1] == T_max
            assert batch["face"].shape[1] == T_max
            assert batch["mask"].shape[1] == T_max
            
            # Validate lengths match actual data
            for i in range(batch_size):
                length = batch["lengths"][i].item()
                # All frames up to length should have data
                assert batch["hand"][i, :length, :].abs().sum() > 0
        
        assert total_samples == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
