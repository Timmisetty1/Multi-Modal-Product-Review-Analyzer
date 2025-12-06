"""
Training script for Multi-Modal Product Review Analyzer.

Fine-tunes pre-trained models on Amazon review data.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    # When run as script
    from model import create_model
    from data_loader import create_data_loaders
except ImportError:
    # When imported as module
    from .model import create_model
    from .data_loader import create_data_loaders


class Trainer:
    """
    Trainer class for multi-modal review analyzer.
    
    Handles training loop, validation, and model checkpointing.
    Addresses computational constraints through mixed precision training.
    """
    
    def __init__(self, model, train_loader, val_loader, config):
        """
        Initialize trainer.
        
        Args:
            model: Multi-modal model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision training for computational efficiency
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    logits = self.model(images, input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images, input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        logits = self.model(images, input_ids, attention_mask)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images, input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=[f'Rating {i+1}' for i in range(5)],
            output_dict=True
        )
        
        return avg_loss, accuracy, report
    
    def train(self, num_epochs, save_dir='models'):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training on {self.device}")
        print(f"Model size: {self.model.get_model_size():.2f} MB")
        print(f"Using mixed precision: {self.use_amp}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validation phase
            val_loss, val_acc, report = self.validate()
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
                print(f"Saved best model with accuracy: {val_acc:.4f}")
        
        # Save final model and history
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")
        
        return self.history


def main():
    """Main training function."""
    # Configuration
    config = {
        'image_model': 'vit',
        'text_model': 'distilbert',
        'fusion_dim': 512,
        'num_classes': 5,
        'dropout': 0.3,
        'freeze_backbones': True,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'num_epochs': 10,
        'use_amp': True
    }
    
    # Data paths (adjust based on your data)
    train_path = 'data/train.csv'
    val_path = 'data/val.csv'
    test_path = 'data/test.csv'
    image_dir = 'data/images'
    
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path, val_path, test_path, image_dir,
        batch_size=config['batch_size']
    )
    
    print("Creating model...")
    model = create_model(config)
    
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.train(num_epochs=config['num_epochs'])
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
