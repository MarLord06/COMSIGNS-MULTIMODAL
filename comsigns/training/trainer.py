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
from .loops import train_one_epoch, train, validate_gradients, validate_one_epoch
from .metrics import MetricsTracker

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
        loss_fn: Optional[nn.Module] = None,
        num_classes: Optional[int] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train (e.g., SignLanguageClassifier)
            config: Training configuration. If None, uses defaults.
            optimizer: Optimizer. If None, creates AdamW in fit().
            loss_fn: Loss function. If None, uses CrossEntropyLoss.
            num_classes: Number of classes for metrics computation. If None,
                        attempts to infer from model.num_classes attribute.
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
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "epoch": []
        }
        
        # Metrics tracking for validation
        # Infer num_classes from model if not provided
        self._num_classes = num_classes or getattr(model, 'num_classes', None)
        self._metrics_tracker: Optional[MetricsTracker] = None
        
        if self._num_classes is not None:
            self._metrics_tracker = MetricsTracker(
                num_classes=self._num_classes,
                topk=(1, 5, 10),
                device="cpu"  # Store on CPU to save GPU memory
            )
            logger.info(f"Metrics tracking enabled: {self._num_classes} classes, topk=(1, 5, 10)")
        else:
            logger.warning(
                "num_classes not provided and could not be inferred from model. "
                "Metrics tracking disabled. Provide num_classes to enable."
            )
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Config: epochs={self.config.epochs}, lr={self.config.learning_rate}")
        logger.info(f"Validation: {self.config.validate}")
    
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
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model with optional validation.
        
        Args:
            train_loader: DataLoader providing training batches.
                         Each batch must be a dict with keys:
                         "hand", "body", "face", "labels", "lengths"
            val_loader: Optional DataLoader for validation. If None and
                       config.validate=True, validation is skipped.
            epochs: Override number of epochs from config
        
        Returns:
            Training history dictionary with:
            - "train_loss": List of average training loss per epoch
            - "val_loss": List of average validation loss per epoch (empty if no validation)
            - "epoch": List of epoch numbers
        
        Example:
            >>> history = trainer.fit(train_loader, val_loader)
            >>> plt.plot(history["epoch"], history["train_loss"], label="train")
            >>> plt.plot(history["epoch"], history["val_loss"], label="val")
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
        logger.info(f"  Train dataset size: {len(train_loader.dataset)}")
        logger.info(f"  Train steps per epoch: {len(train_loader)}")
        if val_loader is not None:
            logger.info(f"  Val dataset size: {len(val_loader.dataset)}")
            logger.info(f"  Val steps per epoch: {len(val_loader)}")
        logger.info(f"  Overfit mode: {self.config.overfit_single_batch}")
        logger.info(f"  Validation enabled: {self.config.validate and val_loader is not None}")
        
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
            
            train_loss = epoch_metrics["loss"]
            self.history["train_loss"].append(train_loss)
            self.history["epoch"].append(epoch)
            self.global_step += epoch_metrics["num_steps"]
            
            # Validation with metrics tracking
            val_loss = None
            val_metrics_results = None
            
            if self.config.validate and val_loader is not None:
                val_loss, val_metrics_results = self._validate_with_metrics(val_loader)
                self.history["val_loss"].append(val_loss)
            
            # Logging
            if val_loss is not None:
                log_msg = (
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )
                # Append classification metrics if available
                if val_metrics_results is not None:
                    log_msg += (
                        f" | Acc: {val_metrics_results['accuracy']:.2%}"
                        f" | Top5: {val_metrics_results['top5_acc']:.2%}"
                        f" | F1: {val_metrics_results['f1_macro']:.4f}"
                    )
                logger.info(log_msg)
            else:
                logger.info(
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_loss:.4f}"
                )
        
        logger.info(f"\nTraining complete!")
        logger.info(f"  Train Loss: {self.history['train_loss'][0]:.4f} → {self.history['train_loss'][-1]:.4f}")
        if self.history["val_loss"]:
            logger.info(f"  Val Loss: {self.history['val_loss'][0]:.4f} → {self.history['val_loss'][-1]:.4f}")
        logger.info(f"  Total steps: {self.global_step}")
        
        return self.history
    
    def _validate_with_metrics(
        self,
        val_loader: DataLoader
    ) -> tuple:
        """
        Run validation with metrics tracking.
        
        Performs forward pass on validation set while accumulating:
        - Loss (for training monitoring)
        - Logits and labels (for classification metrics via MetricsTracker)
        
        Args:
            val_loader: Validation DataLoader
        
        Returns:
            Tuple of (avg_loss, metrics_dict or None)
            metrics_dict is None if MetricsTracker is not available.
        """
        self.model.eval()
        
        total_loss = 0.0
        num_steps = 0
        
        # Reset metrics tracker for this epoch
        if self._metrics_tracker is not None:
            self._metrics_tracker.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
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
                
                # Forward pass
                logits = self.model(hand, body, face, lengths=lengths, mask=mask)
                loss = self.loss_fn(logits, labels)
                
                # Accumulate loss
                total_loss += loss.item()
                num_steps += 1
                
                # Accumulate predictions for metrics (if tracker available)
                if self._metrics_tracker is not None:
                    self._metrics_tracker.update(logits, labels)
        
        # Compute average loss
        avg_loss = total_loss / max(num_steps, 1)
        
        # Compute classification metrics (once per epoch, not per batch)
        metrics_results = None
        if self._metrics_tracker is not None and self._metrics_tracker.num_samples > 0:
            metrics_results = self._metrics_tracker.compute()
        
        return avg_loss, metrics_results
    
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
