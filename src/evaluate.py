"""
Evaluation and inference script for Multi-Modal Product Review Analyzer.
"""

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    # When run as script
    from model import create_model
    from data_loader import create_data_loaders
except ImportError:
    # When imported as module
    from .model import create_model
    from .data_loader import create_data_loaders


class Evaluator:
    """Evaluator class for testing the multi-modal model."""
    
    def __init__(self, model, test_loader, device=None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model instance
            test_loader: Test data loader
            device: Device to use for evaluation
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self):
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluating'):
                # Move data to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(images, input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Collect predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=[f'Rating {i+1}' for i in range(5)],
            output_dict=True
        )
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs)
        }
        
        return results
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'Rating {i+1}' for i in range(5)],
                   yticklabels=[f'Rating {i+1}' for i in range(5)])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_report(self, results):
        """Print evaluation report."""
        print("\n" + "="*50)
        print("Multi-Modal Review Analyzer - Evaluation Results")
        print("="*50)
        print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
        
        print("\nPer-Class Performance:")
        print("-" * 50)
        report = results['classification_report']
        for i in range(5):
            class_name = f'Rating {i+1}'
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1-score']:.4f}")
                print(f"  Support:   {int(metrics['support'])}")
        
        print("\n" + "="*50)


class Predictor:
    """Predictor class for inference on new samples."""
    
    def __init__(self, model, device=None):
        """
        Initialize predictor.
        
        Args:
            model: Trained model instance
            device: Device to use for prediction
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image, input_ids, attention_mask):
        """
        Make prediction on a single sample.
        
        Args:
            image: Image tensor
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            
        Returns:
            Predicted rating (1-5) and confidence scores
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            
            # Move to device
            image = image.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            logits = self.model(image, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            
            # Convert to rating (1-5)
            rating = pred.item() + 1
            confidence = probs[0].cpu().numpy()
        
        return rating, confidence


def load_model(checkpoint_path, config=None):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration (if not in checkpoint)
        
    Returns:
        Loaded model
    """
    if config is None:
        # Default config
        config = {
            'image_model': 'vit',
            'text_model': 'distilbert',
            'fusion_dim': 512,
            'num_classes': 5,
            'dropout': 0.3,
            'freeze_backbones': True
        }
    
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'config' in checkpoint:
            print(f"Model trained with config: {checkpoint['config']}")
    else:
        model.load_state_dict(checkpoint)
    
    return model


def main():
    """Main evaluation function."""
    # Configuration
    config = {
        'image_model': 'vit',
        'text_model': 'distilbert',
        'fusion_dim': 512,
        'num_classes': 5,
        'dropout': 0.3,
        'freeze_backbones': True
    }
    
    # Data paths
    train_path = 'data/train.csv'
    val_path = 'data/val.csv'
    test_path = 'data/test.csv'
    image_dir = 'data/images'
    
    print("Loading data...")
    _, _, test_loader = create_data_loaders(
        train_path, val_path, test_path, image_dir,
        batch_size=32
    )
    
    print("Loading model...")
    model_path = 'models/best_model.pt'
    if os.path.exists(model_path):
        model = load_model(model_path, config)
    else:
        print("Model checkpoint not found. Using untrained model.")
        model = create_model(config)
    
    print("Evaluating model...")
    evaluator = Evaluator(model, test_loader)
    results = evaluator.evaluate()
    
    # Print results
    evaluator.print_report(results)
    
    # Save confusion matrix
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        save_path='models/confusion_matrix.png'
    )
    print("\nConfusion matrix saved to models/confusion_matrix.png")


if __name__ == '__main__':
    main()
