"""
Trainer class for sign language classification.

Provides a high-level interface for training that handles
model setup, optimization, and training loop orchestration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import logging

from .config import TrainerConfig
from .loops import train_one_epoch, train, validate_gradients

logger = logging.getLogger(__name__)


class Trainer:
    """
    High-level trainer for sign language classification.
    
    Orchestrates model training with sensible defaults.
    Handles device management, optimizer creation, and training loop.
    
    Attributes:
        model: The model to train
        config: Training configuration
        optimizer: Optimizer (created in fit() if not provided)
        loss_fn: Loss function (default: CrossEntropyLoss)
    
    Example:
        >>> model = SignLanguageClassifier(encoder, num_classes=100)
        >>> config = TrainerConfig(epochs=10, device="cuda")
        >>> trainer = Trainer(model, config)
        >>> history = trainer.fit(train_loader)
        >>> print(f"Final loss: {history['loss'][-1]:.4f}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train (e.g., SignLanguageClassifier)
            config: Training configuration. If None, uses defaults.
            optimizer: Optimizer. If None, creates AdamW in fit().
            loss_fn: Loss function. If None, uses CrossEntropyLoss.
        """
        self.model = model
        self.config = config or TrainerConfig()
        self._optimizer = optimizer
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # Move model to device
        self.device = self.config.get_torch_device()
        self.model = self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.history: Dict[str, List[float]] = {"loss": [], "epoch": []}
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Config: epochs={self.config.epochs}, lr={self.config.learning_rate}")
    
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get or create optimizer."""
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        return self._optimizer
    
    def fit(
        self,
        train_loader: DataLoader,
        epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader providing training batches.
                         Each batch must be a dict with keys:
                         "hand", "body", "face", "labels", "lengths"
            epochs: Override number of epochs from config
        
        Returns:
            Training history dictionary with:
            - "loss": List of average loss per epoch
            - "epoch": List of epoch numbers
        
        Example:
            >>> history = trainer.fit(train_loader)
            >>> plt.plot(history["epoch"], history["loss"])
        """
        if epochs is not None:
            self.config.epochs = epochs
        
        # Set seed for reproducibility
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.config.seed)
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"  Batch size: {train_loader.batch_size}")
        logger.info(f"  Dataset size: {len(train_loader.dataset)}")
        logger.info(f"  Steps per epoch: {len(train_loader)}")
        logger.info(f"  Overfit mode: {self.config.overfit_single_batch}")
        
        # Prepare overfit batch if needed
        overfit_batch = None
        if self.config.overfit_single_batch:
            overfit_batch = next(iter(train_loader))
            logger.info("Overfit mode enabled - using single batch")
        
        # Training loop
        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.config.epochs}")
            logger.info(f"{'='*50}")
            
            epoch_metrics = train_one_epoch(
                model=self.model,
                dataloader=train_loader,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                device=self.device,
                epoch=epoch,
                config=self.config,
                overfit_batch=overfit_batch
            )
            
            self.history["loss"].append(epoch_metrics["loss"])
            self.history["epoch"].append(epoch)
            self.global_step += epoch_metrics["num_steps"]
            
            logger.info(
                f"Epoch {epoch} | "
                f"Loss: {epoch_metrics['loss']:.4f} | "
                f"Steps: {epoch_metrics['num_steps']}"
            )
        
        logger.info(f"\nTraining complete!")
        logger.info(f"  Final loss: {self.history['loss'][-1]:.4f}")
        logger.info(f"  Total steps: {self.global_step}")
        
        return self.history
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Useful for custom training loops or debugging.
        
        Args:
            batch: Dictionary with "hand", "body", "face", "labels", "lengths"
        
        Returns:
            Dictionary with "loss" and optionally other metrics
        """
        self.model.train()
        
        # Move to device
        hand = batch["hand"].to(self.device)
        body = batch["body"].to(self.device)
        face = batch["face"].to(self.device)
        labels = batch["labels"].to(self.device)
        lengths = batch["lengths"].to(self.device)
        mask = batch.get("mask")
        if mask is not None and mask.numel() > 0:
            mask = mask.to(self.device)
        else:
            mask = None
        
        # Forward
        self.optimizer.zero_grad()
        logits = self.model(hand, body, face, lengths=lengths, mask=mask)
        loss = self.loss_fn(logits, labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_val
            )
        
        # Step
        self.optimizer.step()
        self.global_step += 1
        
        return {"loss": loss.item()}
    
    def validate_training(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Validate that training is working correctly.
        
        Performs one step and checks gradients.
        Useful for debugging.
        
        Args:
            batch: A single batch from the DataLoader
        
        Returns:
            Validation results including gradient info
        """
        # Perform one step
        metrics = self.train_step(batch)
        
        # Check gradients
        grad_info = validate_gradients(self.model)
        
        results = {
            "loss": metrics["loss"],
            "has_gradients": grad_info["has_gradients"],
            "non_zero_params": grad_info["non_zero_params"],
            "total_params": grad_info["total_params"],
        }
        
        # Log shapes for debugging
        logger.info("=== Training Validation ===")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Gradients: {grad_info['non_zero_params']}/{grad_info['total_params']} non-zero")
        logger.info(f"  Batch shapes:")
        logger.info(f"    hand: {batch['hand'].shape}")
        logger.info(f"    body: {batch['body'].shape}")
        logger.info(f"    face: {batch['face'].shape}")
        logger.info(f"    labels: {batch['labels'].shape}")
        
        return results
    
    def get_model(self) -> nn.Module:
        """Return the model (on the training device)."""
        return self.model
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get trainer state for checkpointing.
        
        Note: This is minimal - does not include full checkpoint support.
        """
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "history": self.history,
            "config": {
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "device": self.config.device,
            }
        }
