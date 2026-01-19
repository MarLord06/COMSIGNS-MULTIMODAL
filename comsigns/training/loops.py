"""
Training loops for sign language classification.

Provides pure functions for training that can be used
independently or via the Trainer class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Callable
import logging

from .config import TrainerConfig

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    config: TrainerConfig,
    overfit_batch: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Pure function that performs one full pass over the dataloader.
    
    Args:
        model: Model to train (must be in train mode)
        dataloader: Training data loader
        optimizer: Optimizer instance
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        device: Device to use
        epoch: Current epoch number (for logging)
        config: Training configuration
        overfit_batch: If provided, use this batch for all steps (debug mode)
    
    Returns:
        Dictionary with epoch metrics:
        - "loss": Average loss for the epoch
        - "num_steps": Number of training steps
    """
    model.train()
    
    total_loss = 0.0
    num_steps = 0
    
    # Use overfit batch or iterate normally
    if overfit_batch is not None:
        # Overfit mode: use same batch repeatedly
        num_iterations = len(dataloader)
        batches = [overfit_batch] * num_iterations
    else:
        batches = dataloader
    
    for step, batch in enumerate(batches):
        # Move batch to device
        hand = batch["hand"].to(device)
        body = batch["body"].to(device)
        face = batch["face"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        mask = batch.get("mask")
        if mask is not None and mask.numel() > 0:
            mask = mask.to(device)
        else:
            mask = None
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(hand, body, face, lengths=lengths, mask=mask)
        loss = loss_fn(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.gradient_clip_val
            )
        
        # Optimizer step
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        num_steps += 1
        
        # Logging
        if (step + 1) % config.log_every_n_steps == 0:
            avg_loss = total_loss / num_steps
            logger.info(
                f"Epoch {epoch} | Step {step + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}"
            )
    
    return {
        "loss": total_loss / max(num_steps, 1),
        "num_steps": num_steps
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: TrainerConfig
) -> Dict[str, List[float]]:
    """
    Full training loop for multiple epochs.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer instance
        loss_fn: Loss function
        config: Training configuration
    
    Returns:
        Training history with:
        - "loss": List of epoch losses
        - "step_losses": List of all step losses
    """
    device = config.get_torch_device()
    model = model.to(device)
    
    # Set random seed if specified
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
    
    history = {
        "loss": [],
        "epoch": []
    }
    
    # Prepare overfit batch if needed
    overfit_batch = None
    if config.overfit_single_batch:
        overfit_batch = next(iter(train_loader))
        logger.info("Overfit mode: Using single batch for all training")
        logger.info(f"  Batch size: {overfit_batch['hand'].shape[0]}")
        logger.info(f"  Max seq length: {overfit_batch['hand'].shape[1]}")
    
    # Training loop
    for epoch in range(1, config.epochs + 1):
        logger.info(f"=== Epoch {epoch}/{config.epochs} ===")
        
        epoch_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            config=config,
            overfit_batch=overfit_batch
        )
        
        history["loss"].append(epoch_metrics["loss"])
        history["epoch"].append(epoch)
        
        logger.info(
            f"Epoch {epoch} complete | "
            f"Loss: {epoch_metrics['loss']:.4f} | "
            f"Steps: {epoch_metrics['num_steps']}"
        )
    
    return history


def validate_gradients(model: nn.Module) -> Dict[str, Any]:
    """
    Check that gradients are non-zero after backward.
    
    Useful for debugging training issues.
    
    Args:
        model: Model after backward() has been called
    
    Returns:
        Dictionary with gradient statistics:
        - "has_gradients": bool, True if any param has gradients
        - "non_zero_params": int, count of params with non-zero gradients
        - "total_params": int, total trainable parameters
        - "grad_norms": dict mapping param name to gradient norm
    """
    grad_norms = {}
    non_zero_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_count += 1
            if param.grad is not None:
                norm = param.grad.norm().item()
                grad_norms[name] = norm
                if norm > 0:
                    non_zero_count += 1
    
    return {
        "has_gradients": non_zero_count > 0,
        "non_zero_params": non_zero_count,
        "total_params": total_count,
        "grad_norms": grad_norms
    }
