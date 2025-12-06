# Multi-Modal Product Review Analyzer

A deep learning system that combines image and text features to predict product ratings from Amazon reviews. This project demonstrates practical multi-modal learning, achieving **~0.68 accuracy** on 5-class rating prediction.

## 🎯 Project Overview

This system addresses the challenge of analyzing product reviews by leveraging both visual and textual information. By fine-tuning pre-trained vision and language models, it learns to:
- Extract meaningful features from product images
- Understand sentiment and content from review text
- Fuse multi-modal features for accurate rating prediction
- Handle practical challenges like feature alignment and computational constraints

## 🏗️ Architecture

The model consists of three main components:

1. **Image Feature Extractor**: Vision Transformer (ViT) or ResNet for extracting visual features
2. **Text Feature Extractor**: DistilBERT or BERT for processing review text
3. **Multi-Modal Fusion Module**: Attention-based mechanism for feature alignment and fusion

```
Product Image ──► Vision Model (ViT) ──┐
                                        ├──► Fusion Module ──► Classifier ──► Rating (1-5)
Review Text ───► Language Model (BERT) ─┘
```

## 📊 Features

- **Multi-Modal Learning**: Combines visual and textual information
- **Pre-trained Models**: Leverages ViT and BERT for transfer learning
- **Attention Mechanism**: Adaptive fusion of features from different modalities
- **Computational Efficiency**: Mixed precision training and frozen backbones
- **5-Class Classification**: Predicts ratings from 1 to 5 stars
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/Timmisetty1/Multi-Modal-Product-Review-Analyzer.git
cd Multi-Modal-Product-Review-Analyzer

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

#### 1. Prepare Sample Data

```bash
# Create sample dataset for demonstration
python scripts/prepare_data.py --create-sample --num-samples 1000
```

#### 2. Train the Model

```bash
# Train with default configuration
python src/train.py
```

#### 3. Evaluate

```bash
# Evaluate on test set
python src/evaluate.py
```

## 📁 Project Structure

```
Multi-Modal-Product-Review-Analyzer/
├── src/
│   ├── model.py           # Multi-modal model architecture
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── train.py           # Training script
│   └── evaluate.py        # Evaluation and inference
├── scripts/
│   └── prepare_data.py    # Data preparation utilities
├── data/                  # Dataset directory
├── models/                # Saved model checkpoints
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Python dependencies
└── README.md
```

## 🔧 Usage

### Training with Custom Data

```python
from src.model import create_model
from src.data_loader import create_data_loaders
from src.train import Trainer

# Configuration
config = {
    'image_model': 'vit',
    'text_model': 'distilbert',
    'fusion_dim': 512,
    'num_classes': 5,
    'learning_rate': 1e-4,
    'batch_size': 32
}

# Load data
train_loader, val_loader, test_loader = create_data_loaders(
    'data/train.csv', 'data/val.csv', 'data/test.csv',
    'data/images', batch_size=32
)

# Create and train model
model = create_model(config)
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train(num_epochs=10)
```

### Making Predictions

```python
from src.evaluate import load_model, Predictor
from transformers import BertTokenizer, ViTImageProcessor
from PIL import Image

# Load trained model
model = load_model('models/best_model.pt')
predictor = Predictor(model)

# Prepare input
image = Image.open('product.jpg')
text = "Great product! Highly recommend."

# Preprocess
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

image_tensor = image_processor(images=image, return_tensors='pt')['pixel_values'][0]
encoding = tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)

# Predict
rating, confidence = predictor.predict(
    image_tensor, 
    encoding['input_ids'][0], 
    encoding['attention_mask'][0]
)

print(f"Predicted Rating: {rating}/5")
print(f"Confidence: {confidence}")
```

## 📈 Performance

Expected performance on Amazon review dataset (20K samples):
- **Overall Accuracy**: ~0.68
- **Training Time**: ~2-3 hours on single GPU
- **Model Size**: ~400MB

Performance varies based on:
- Dataset quality and balance
- Training hyperparameters
- Backbone freezing strategy

## 🎓 Key Learnings

This project demonstrates practical challenges in multi-modal learning:

1. **Feature Alignment**: Different modalities (vision/language) have different feature spaces
2. **Computational Constraints**: Pre-trained models are large; freezing helps
3. **Data Quality**: Both image and text quality affect performance
4. **Fusion Strategy**: Attention mechanisms help weight modality importance
5. **Class Imbalance**: Rating distributions may be skewed

## 🛠️ Model Components

### Image Feature Extractor
- **Vision Transformer (ViT)**: 768-dim features from CLS token
- **ResNet50**: 2048-dim features from global average pooling
- Frozen backbones for computational efficiency

### Text Feature Extractor
- **DistilBERT**: Efficient 768-dim text representations
- **BERT**: Full 768-dim contextualized embeddings
- Max sequence length: 128 tokens

### Multi-Modal Fusion
- Projection layers for feature alignment
- Attention mechanism for adaptive fusion
- Dropout for regularization

## 📝 Data Format

The dataset should be organized as follows:

```
data/
├── train.csv
├── val.csv
├── test.csv
└── images/
    ├── product_1.jpg
    ├── product_2.jpg
    └── ...
```

CSV format:
```csv
image_id,review_text,rating
product_1.jpg,"Great product!",5
product_2.jpg,"Not satisfied",2
...
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional fusion strategies
- More pre-trained model options
- Data augmentation techniques
- Hyperparameter optimization

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Pre-trained models from Hugging Face Transformers
- PyTorch and torchvision for deep learning framework
- Amazon review dataset for multi-modal learning research

## 📧 Contact

For questions or feedback, please open an issue on GitHub.