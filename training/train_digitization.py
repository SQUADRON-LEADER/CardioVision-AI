"""
Training Script for ECG Image Digitization
Trains model to convert ECG images to time-series signals
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datetime import datetime
import json

from ECG_Digitization_Model import (
    ECGDigitizationModel,
    AdvancedECGDigitizationModel,
    ECGDigitizationLoss,
    create_digitization_model
)
from ECG_Digitization_Dataset import create_digitization_dataloaders


class ECGDigitizationTrainer:
    """
    Trainer for ECG digitization models
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        checkpoint_dir='digitization_checkpoints',
        log_dir='digitization_logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'train_mae': [],
            'val_mae': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_mse = 0
        total_mae = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, signals, metadata) in enumerate(pbar):
            images = images.to(self.device)
            signals = signals.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_signals = self.model(images)
            
            # Calculate loss
            loss, loss_dict = self.criterion(pred_signals, signals)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                mse = nn.functional.mse_loss(pred_signals, signals).item()
                mae = nn.functional.l1_loss(pred_signals, signals).item()
            
            total_loss += loss.item()
            total_mse += mse
            total_mae += mae
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse:.4f}',
                'mae': f'{mae:.4f}'
            })
        
        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mse, avg_mae
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        total_mse = 0
        total_mae = 0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for images, signals, metadata in pbar:
                images = images.to(self.device)
                signals = signals.to(self.device)
                
                # Forward pass
                pred_signals = self.model(images)
                
                # Calculate loss
                loss, loss_dict = self.criterion(pred_signals, signals)
                
                # Calculate metrics
                mse = nn.functional.mse_loss(pred_signals, signals).item()
                mae = nn.functional.l1_loss(pred_signals, signals).item()
                
                total_loss += loss.item()
                total_mse += mse
                total_mae += mae
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mse': f'{mse:.4f}',
                    'mae': f'{mae:.4f}'
                })
        
        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mse, avg_mae
    
    def train(self, num_epochs, save_freq=5):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs to train
            save_freq: Save checkpoint every N epochs
        """
        print("=" * 80)
        print("Starting ECG Digitization Training")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Epochs: {num_epochs}")
        print("=" * 80)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_mse, train_mae = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_mse, val_mae = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mse'].append(train_mse)
            self.history['val_mse'].append(val_mse)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model! Val Loss: {val_loss:.4f}")
            
            # Save regular checkpoint
            if epoch % save_freq == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Save training plots
            if epoch % 5 == 0:
                self.plot_training_history()
        
        print("\n" + "=" * 80)
        print(f"Training Complete!")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print("=" * 80)
        
        # Save final history
        self.save_training_history()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_digitization_model.pth')
            print(f"  Saving best model to {path}")
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            print(f"  Saving checkpoint to {path}")
        
        torch.save(checkpoint, path)
    
    def save_training_history(self):
        """Save training history to CSV"""
        df = pd.DataFrame(self.history)
        path = os.path.join(self.log_dir, 'training_history.csv')
        df.to_csv(path, index=False)
        print(f"Training history saved to {path}")
    
    def plot_training_history(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Loss', fontweight='bold')
        axes[0].set_title('Training & Validation Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MSE
        axes[1].plot(epochs, self.history['train_mse'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.history['val_mse'], 'r-', label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('MSE', fontweight='bold')
        axes[1].set_title('Mean Squared Error', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # MAE
        axes[2].plot(epochs, self.history['train_mae'], 'b-', label='Train', linewidth=2)
        axes[2].plot(epochs, self.history['val_mae'], 'r-', label='Val', linewidth=2)
        axes[2].set_xlabel('Epoch', fontweight='bold')
        axes[2].set_ylabel('MAE', fontweight='bold')
        axes[2].set_title('Mean Absolute Error', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ECG Digitization Model')
    parser.add_argument('--ptbxl_dir', type=str, 
                       default=r'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3',
                       help='Path to PTB-XL dataset')
    parser.add_argument('--image_dir', type=str, default='ecg_data',
                       help='Path to ECG images directory')
    parser.add_argument('--model_type', type=str, default='advanced', choices=['basic', 'advanced'],
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--signal_length', type=int, default=1000,
                       help='Signal length (samples)')
    parser.add_argument('--sampling_rate', type=int, default=100, choices=[100, 500],
                       help='Sampling rate (Hz)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_digitization_dataloaders(
        ptbxl_dir=args.ptbxl_dir,
        image_dir=args.image_dir if os.path.exists(args.image_dir) else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampling_rate=args.sampling_rate,
        signal_length=args.signal_length
    )
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = create_digitization_model(
        model_type=args.model_type,
        signal_length=args.signal_length,
        num_leads=12,
        dropout=0.3
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function
    criterion = ECGDigitizationLoss(mse_weight=1.0, shape_weight=0.3, freq_weight=0.2)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create trainer
    trainer = ECGDigitizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir='digitization_checkpoints',
        log_dir='digitization_logs'
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs, save_freq=5)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
