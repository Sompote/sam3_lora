"""
LoRA Training Script for SAM3

This script follows the same training procedure as sam3/train/trainer.py
but with LoRA-specific modifications.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '/workspace/sam3')

from sam3.model_builder import build_sam3_image_model
from sam3.train.utils.train_utils import AverageMeter, set_seeds, setup_distributed_backend

from ..lora.lora_utils import (
    LoRAConfig,
    get_lora_parameters,
    get_lora_state_dict,
    inject_lora_into_model,
    load_lora_state_dict,
    print_trainable_parameters,
)


class LoRATrainer:
    """
    Trainer for SAM3 with LoRA fine-tuning.

    This follows the same structure as sam3.train.trainer.Trainer but is
    optimized for LoRA fine-tuning.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        max_epochs: int = 20,
        val_epoch_freq: int = 5,
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
        save_freq: int = 5,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
    ):
        self.device = torch.device(device)
        self.max_epochs = max_epochs
        self.val_epoch_freq = val_epoch_freq
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

        # Setup directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Inject LoRA into model
        logging.info("Injecting LoRA into model...")
        self.model = inject_lora_into_model(model, lora_config, verbose=True)
        self.model = self.model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # AMP scaler
        amp_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        self.amp_dtype = amp_dtype_map.get(amp_dtype, torch.bfloat16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=log_dir)

        logging.info("Trainer initialized")
        print_trainable_parameters(self.model)

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "training.log")),
                logging.StreamHandler(),
            ],
        )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        loss_meter = AverageMeter("Loss", self.device, ":.4e")
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")

        end = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_to_device(batch)

            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, batch)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            # Update metrics
            loss_meter.update(loss.item() * self.gradient_accumulation_steps, batch["images"].size(0))
            batch_time_meter.update(time.time() - end)
            end = time.time()

            # Logging
            if batch_idx % 10 == 0:
                logging.info(
                    f"Epoch [{self.epoch}/{self.max_epochs}] "
                    f"Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss_meter.avg:.4f} "
                    f"Time: {batch_time_meter.avg:.2f}s"
                )

                # Tensorboard logging
                self.writer.add_scalar("train/loss", loss_meter.val, self.global_step)
                if self.scheduler is not None:
                    self.writer.add_scalar(
                        "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                    )

        return {"loss": loss_meter.avg}

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        loss_meter = AverageMeter("Val Loss", self.device, ":.4e")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move batch to device
                batch = self._move_to_device(batch)

                # Forward pass with AMP
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(batch)
                    loss = self._compute_loss(outputs, batch)

                # Update metrics
                loss_meter.update(loss.item(), batch["images"].size(0))

                if batch_idx % 10 == 0:
                    logging.info(
                        f"Validation [{batch_idx}/{len(self.val_loader)}] "
                        f"Loss: {loss_meter.avg:.4f}"
                    )

        # Tensorboard logging
        self.writer.add_scalar("val/loss", loss_meter.avg, self.epoch)

        return {"loss": loss_meter.avg}

    def train(self):
        """Main training loop."""
        logging.info("Starting training...")

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch()
            logging.info(f"Epoch {epoch} training metrics: {train_metrics}")

            # Validation
            if epoch % self.val_epoch_freq == 0 or epoch == self.max_epochs - 1:
                val_metrics = self.validate()
                logging.info(f"Epoch {epoch} validation metrics: {val_metrics}")

                # Save best model
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint("best")
                    logging.info(f"Saved best model with val loss: {self.best_val_loss:.4f}")

            # Save checkpoint
            if epoch % self.save_freq == 0 or epoch == self.max_epochs - 1:
                self.save_checkpoint(f"epoch_{epoch}")

        logging.info("Training completed!")
        self.writer.close()

    def save_checkpoint(self, name: str):
        """
        Save checkpoint.

        Args:
            name: Checkpoint name
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pt")

        # Get LoRA state dict (only save LoRA parameters)
        lora_state_dict = get_lora_state_dict(self.model)

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "lora_state_dict": lora_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        logging.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load LoRA weights
        load_lora_state_dict(self.model, checkpoint["lora_state_dict"])

        # Load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scaler
        if self.use_amp and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        # Load scheduler
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        # Load training state
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logging.info(f"Loaded checkpoint from epoch {self.epoch}")

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        if isinstance(batch["images"], torch.Tensor):
            batch["images"] = batch["images"].to(self.device, non_blocking=True)
        return batch

    def _compute_loss(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss. This is a placeholder - you should implement the actual
        loss computation based on your task.

        Args:
            outputs: Model outputs
            batch: Input batch

        Returns:
            Loss tensor
        """
        # This is a placeholder - replace with actual SAM3 loss computation
        # For example, you would use the SAM3 loss functions from sam3.train.loss

        # Dummy loss for demonstration
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        else:
            # If the model doesn't return loss, you need to implement it
            raise NotImplementedError(
                "Loss computation not implemented. "
                "Please implement _compute_loss method with SAM3 loss functions."
            )
