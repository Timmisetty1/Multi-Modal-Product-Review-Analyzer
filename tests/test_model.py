"""
Unit tests for multi-modal model components.
"""

import unittest
import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import (
    ImageFeatureExtractor,
    TextFeatureExtractor,
    MultiModalFusion,
    MultiModalReviewAnalyzer,
    create_model
)


class TestModelComponents(unittest.TestCase):
    """Test individual model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.image_size = 224
        self.seq_length = 128
    
    def test_image_feature_extractor_vit(self):
        """Test ViT image feature extractor."""
        extractor = ImageFeatureExtractor(model_name='vit', freeze_backbone=True)
        
        # Create dummy input
        images = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        # Forward pass
        features = extractor(images)
        
        # Check output shape
        self.assertEqual(features.shape, (self.batch_size, 768))
        self.assertEqual(extractor.feature_dim, 768)
    
    def test_text_feature_extractor_distilbert(self):
        """Test DistilBERT text feature extractor."""
        extractor = TextFeatureExtractor(model_name='distilbert', freeze_backbone=True)
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        
        # Forward pass
        features = extractor(input_ids, attention_mask)
        
        # Check output shape
        self.assertEqual(features.shape, (self.batch_size, 768))
        self.assertEqual(extractor.feature_dim, 768)
    
    def test_multimodal_fusion(self):
        """Test multi-modal fusion module."""
        fusion = MultiModalFusion(image_dim=768, text_dim=768, fusion_dim=512)
        
        # Create dummy features
        image_features = torch.randn(self.batch_size, 768)
        text_features = torch.randn(self.batch_size, 768)
        
        # Forward pass
        fused = fusion(image_features, text_features)
        
        # Check output shape
        self.assertEqual(fused.shape, (self.batch_size, 512))
    
    def test_full_model(self):
        """Test complete multi-modal model."""
        model = MultiModalReviewAnalyzer(
            image_model='vit',
            text_model='distilbert',
            fusion_dim=512,
            num_classes=5
        )
        
        # Create dummy input
        images = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        
        # Forward pass
        logits = model(images, input_ids, attention_mask)
        
        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, 5))
    
    def test_create_model_factory(self):
        """Test model factory function."""
        config = {
            'image_model': 'vit',
            'text_model': 'distilbert',
            'fusion_dim': 512,
            'num_classes': 5
        }
        
        model = create_model(config)
        
        # Check model type
        self.assertIsInstance(model, MultiModalReviewAnalyzer)
    
    def test_model_size_calculation(self):
        """Test model size calculation."""
        model = create_model()
        size_mb = model.get_model_size()
        
        # Model should be reasonably sized (between 100MB and 1GB)
        self.assertGreater(size_mb, 100)
        self.assertLess(size_mb, 1000)


class TestModelTraining(unittest.TestCase):
    """Test model training capabilities."""
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        model = create_model()
        
        # Create dummy input
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)
        labels = torch.tensor([0, 1])
        
        # Forward pass
        logits = model(images, input_ids, attention_mask)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for trainable parameters
        has_gradients = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break
        
        self.assertTrue(has_gradients, "Model should have gradients after backward pass")


if __name__ == '__main__':
    unittest.main()
