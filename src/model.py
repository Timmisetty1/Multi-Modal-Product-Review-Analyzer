"""
Multi-Modal Product Review Analyzer Model

Combines image and text features using pre-trained models to predict product ratings.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel, ViTImageProcessor
from torchvision import models


class ImageFeatureExtractor(nn.Module):
    """Extract features from product images using pre-trained vision models."""
    
    def __init__(self, model_name='vit', freeze_backbone=True):
        super(ImageFeatureExtractor, self).__init__()
        self.model_name = model_name
        
        if model_name == 'vit':
            # Vision Transformer for better feature extraction
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.feature_dim = 768
        elif model_name == 'resnet':
            # ResNet50 as alternative
            self.backbone = models.resnet50(pretrained=True)
            # Remove final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone if specified (for computational efficiency)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        """Extract image features."""
        if self.model_name == 'vit':
            outputs = self.backbone(pixel_values=images)
            # Use CLS token representation
            features = outputs.last_hidden_state[:, 0, :]
        else:  # resnet
            features = self.backbone(images)
            features = features.view(features.size(0), -1)
        return features


class TextFeatureExtractor(nn.Module):
    """Extract features from review text using pre-trained language models."""
    
    def __init__(self, model_name='distilbert', freeze_backbone=True):
        super(TextFeatureExtractor, self).__init__()
        
        if model_name == 'distilbert':
            self.backbone = BertModel.from_pretrained('distilbert-base-uncased')
            self.feature_dim = 768
        elif model_name == 'bert':
            self.backbone = BertModel.from_pretrained('bert-base-uncased')
            self.feature_dim = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone if specified (for computational efficiency)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """Extract text features."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation
        features = outputs.last_hidden_state[:, 0, :]
        return features


class MultiModalFusion(nn.Module):
    """
    Fusion module to combine image and text features.
    
    Addresses the challenge of feature alignment between different modalities.
    """
    
    def __init__(self, image_dim, text_dim, fusion_dim=512, dropout=0.3):
        super(MultiModalFusion, self).__init__()
        
        # Project features to common dimension for alignment
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for adaptive fusion
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, image_features, text_features):
        """Fuse image and text features using attention mechanism."""
        # Project to common dimension
        img_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)
        
        # Concatenate for attention
        concat_features = torch.cat([img_proj, text_proj], dim=1)
        
        # Calculate attention weights
        attention_weights = self.attention(concat_features)
        
        # Apply attention weights
        img_attended = img_proj * attention_weights[:, 0:1]
        text_attended = text_proj * attention_weights[:, 1:2]
        
        # Fused features
        fused = img_attended + text_attended
        
        return fused


class MultiModalReviewAnalyzer(nn.Module):
    """
    Complete Multi-Modal Product Review Analyzer.
    
    Predicts 5-class product ratings from images and text reviews.
    """
    
    def __init__(self, 
                 image_model='vit',
                 text_model='distilbert',
                 fusion_dim=512,
                 num_classes=5,
                 dropout=0.3,
                 freeze_backbones=True):
        super(MultiModalReviewAnalyzer, self).__init__()
        
        # Feature extractors
        self.image_extractor = ImageFeatureExtractor(
            model_name=image_model,
            freeze_backbone=freeze_backbones
        )
        self.text_extractor = TextFeatureExtractor(
            model_name=text_model,
            freeze_backbone=freeze_backbones
        )
        
        # Fusion module
        self.fusion = MultiModalFusion(
            image_dim=self.image_extractor.feature_dim,
            text_dim=self.text_extractor.feature_dim,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, input_ids, attention_mask):
        """Forward pass through the multi-modal network."""
        # Extract features from both modalities
        image_features = self.image_extractor(images)
        text_features = self.text_extractor(input_ids, attention_mask)
        
        # Fuse features
        fused_features = self.fusion(image_features, text_features)
        
        # Predict rating
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_model_size(self):
        """Calculate model size in MB."""
        param_size = sum(p.numel() for p in self.parameters()) * 4 / (1024 ** 2)
        return param_size


def create_model(config=None):
    """
    Factory function to create the multi-modal model.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        MultiModalReviewAnalyzer instance
    """
    if config is None:
        config = {
            'image_model': 'vit',
            'text_model': 'distilbert',
            'fusion_dim': 512,
            'num_classes': 5,
            'dropout': 0.3,
            'freeze_backbones': True
        }
    
    model = MultiModalReviewAnalyzer(**config)
    return model
