"""Training utilities for LSLM model."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        output_dir,
        use_wandb=True
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb
        
        # Setup training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.max_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(project="lslm", config=vars(config))
            
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch in pbar:
                # Move to device
                audio_features = batch["audio_features"].to(self.device)
                text_input_ids = batch["text_input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(audio_features, text_input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    text_input_ids.view(-1)
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )
                self.optimizer.step()
                self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                
                if self.use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })
                    
        return total_loss / len(self.train_loader)
        
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move to device
            audio_features = batch["audio_features"].to(self.device)
            text_input_ids = batch["text_input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass
            logits = self.model(audio_features, text_input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                text_input_ids.view(-1)
            )
            total_loss += loss.item()
            
        val_loss = total_loss / len(self.val_loader)
        if self.use_wandb:
            wandb.log({"val_loss": val_loss})
            
        return val_loss
        
    def train(self):
        """Train the model."""
        best_val_loss = float("inf")
        
        for epoch in range(self.config.max_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} train loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            logger.info(f"Epoch {epoch} val loss: {val_loss:.4f}")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}") 