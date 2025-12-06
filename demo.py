#!/usr/bin/env python
"""
Quick demo script to demonstrate the Multi-Modal Product Review Analyzer.
"""

import sys
sys.path.insert(0, 'src')

import torch
import pandas as pd
from model import create_model


def main():
    print("=" * 60)
    print("Multi-Modal Product Review Analyzer - Demo")
    print("=" * 60)
    
    # 1. Check PyTorch
    print(f"\n1. Environment Check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    # 2. Load sample data
    print(f"\n2. Loading Sample Data:")
    try:
        df = pd.read_csv('data/train.csv')
        print(f"   Training samples: {len(df)}")
        print(f"   Rating distribution:")
        for rating in range(1, 6):
            count = (df['rating'] == rating).sum()
            print(f"     {rating} stars: {count}")
    except FileNotFoundError:
        print("   No sample data found. Run: python scripts/prepare_data.py --create-sample")
    
    # 3. Create model (without loading pre-trained weights)
    print(f"\n3. Model Architecture:")
    print(f"   Creating multi-modal model configuration...")
    
    config = {
        'image_model': 'vit',
        'text_model': 'distilbert',
        'fusion_dim': 512,
        'num_classes': 5,
        'dropout': 0.3,
        'freeze_backbones': True
    }
    
    print(f"   Configuration:")
    for key, value in config.items():
        print(f"     - {key}: {value}")
    
    print(f"\n   Note: Model initialization requires downloading pre-trained")
    print(f"   weights from HuggingFace (~400MB). This will happen on first use.")
    
    # 4. Show capabilities
    print(f"\n4. System Capabilities:")
    print(f"   ✓ Multi-modal learning (Image + Text)")
    print(f"   ✓ 5-class rating prediction (1-5 stars)")
    print(f"   ✓ Pre-trained vision models (ViT, ResNet)")
    print(f"   ✓ Pre-trained language models (BERT, DistilBERT)")
    print(f"   ✓ Attention-based feature fusion")
    print(f"   ✓ Mixed precision training")
    print(f"   ✓ Target accuracy: ~0.68 on 20K samples")
    
    # 5. Usage instructions
    print(f"\n5. Quick Start:")
    print(f"   a) Install dependencies: pip install -r requirements.txt")
    print(f"   b) Prepare data: python scripts/prepare_data.py --create-sample")
    print(f"   c) Train model: python src/train.py")
    print(f"   d) Evaluate: python src/evaluate.py")
    print(f"   e) Use notebook: jupyter notebook notebooks/demo.ipynb")
    
    print(f"\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
