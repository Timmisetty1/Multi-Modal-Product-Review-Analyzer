"""
Data loading and preprocessing for Amazon product reviews.

Handles multi-modal data with images and text.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer, ViTImageProcessor
from torchvision import transforms
import pandas as pd


class AmazonReviewDataset(Dataset):
    """
    Dataset class for Amazon product reviews with images and text.
    
    Supports both images and text for multi-modal learning.
    """
    
    def __init__(self, 
                 data_path,
                 image_dir,
                 text_model='distilbert-base-uncased',
                 image_model='google/vit-base-patch16-224-in21k',
                 max_length=128,
                 split='train'):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to CSV or JSON file with review data
            image_dir: Directory containing product images
            text_model: Name of pre-trained text model for tokenization
            image_model: Name of pre-trained image model for preprocessing
            max_length: Maximum sequence length for text
            split: Dataset split (train/val/test)
        """
        self.data_path = data_path
        self.image_dir = image_dir
        self.max_length = max_length
        self.split = split
        
        # Load data
        self.data = self._load_data()
        
        # Initialize tokenizer and image processor
        self.tokenizer = BertTokenizer.from_pretrained(text_model)
        
        # Image preprocessing
        if 'vit' in image_model.lower():
            self.image_processor = ViTImageProcessor.from_pretrained(image_model)
            self.transform = None
        else:
            # For ResNet and other models
            self.image_processor = None
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_data(self):
        """Load review data from file."""
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.json'):
            df = pd.read_json(self.data_path, lines=True)
        else:
            raise ValueError("Data file must be CSV or JSON")
        
        # Filter for the split if column exists
        if 'split' in df.columns:
            df = df[df['split'] == self.split]
        
        return df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        row = self.data.iloc[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, row['image_id'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # If image not found, create a blank image
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.image_processor:
            # ViT preprocessing
            image_tensor = self.image_processor(images=image, return_tensors='pt')['pixel_values'][0]
        else:
            # ResNet preprocessing
            image_tensor = self.transform(image)
        
        # Process text
        text = str(row['review_text'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Rating (1-5 scale, convert to 0-4 for classification)
        rating = int(row['rating']) - 1
        
        return {
            'image': image_tensor,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(rating, dtype=torch.long)
        }


def create_data_loaders(train_path, val_path, test_path, image_dir,
                       batch_size=32, num_workers=4, **kwargs):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        image_dir: Directory with images
        batch_size: Batch size for training
        num_workers: Number of worker processes
        **kwargs: Additional arguments for dataset
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = AmazonReviewDataset(
        train_path, image_dir, split='train', **kwargs
    )
    val_dataset = AmazonReviewDataset(
        val_path, image_dir, split='val', **kwargs
    )
    test_dataset = AmazonReviewDataset(
        test_path, image_dir, split='test', **kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def prepare_sample_data(output_dir='data', num_samples=100):
    """
    Create sample data for demonstration purposes.
    
    Args:
        output_dir: Directory to save sample data
        num_samples: Number of samples to generate
    """
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Generate sample reviews
    sample_texts = [
        "This product is amazing! Highly recommend it.",
        "Not satisfied with the quality. Expected better.",
        "Good value for money. Works as described.",
        "Excellent product! Best purchase this year.",
        "Terrible experience. Do not buy.",
        "Average product. Nothing special.",
        "Love it! Exactly what I needed.",
        "Poor quality. Broke after one use.",
        "Great features and easy to use.",
        "Disappointed with the performance."
    ]
    
    data = []
    for i in range(num_samples):
        # Random rating (1-5)
        rating = np.random.randint(1, 6)
        
        # Select text based on rating tendency
        if rating >= 4:
            text = np.random.choice(sample_texts[:3] + sample_texts[6:9])
        elif rating <= 2:
            text = np.random.choice(sample_texts[1:2] + sample_texts[4:6] + sample_texts[7:8])
        else:
            text = np.random.choice(sample_texts[2:3] + sample_texts[5:6] + sample_texts[8:10])
        
        # Create dummy image
        img = Image.new('RGB', (224, 224), 
                       color=(np.random.randint(0, 255),
                             np.random.randint(0, 255),
                             np.random.randint(0, 255)))
        img.save(os.path.join(output_dir, 'images', f'product_{i}.jpg'))
        
        data.append({
            'image_id': f'product_{i}.jpg',
            'review_text': text,
            'rating': rating,
            'split': 'train' if i < 70 else ('val' if i < 85 else 'test')
        })
    
    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'reviews.csv'), index=False)
    
    print(f"Created {num_samples} sample reviews in {output_dir}")
    return df
