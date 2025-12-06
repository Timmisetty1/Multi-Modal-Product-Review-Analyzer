"""
Multi-Modal Product Review Analyzer

A deep learning system for predicting product ratings from images and text reviews.
"""

__version__ = '1.0.0'

from .model import (
    ImageFeatureExtractor,
    TextFeatureExtractor,
    MultiModalFusion,
    MultiModalReviewAnalyzer,
    create_model
)

from .data_loader import (
    AmazonReviewDataset,
    create_data_loaders,
    prepare_sample_data
)

from .train import Trainer

from .evaluate import (
    Evaluator,
    Predictor,
    load_model
)

__all__ = [
    'ImageFeatureExtractor',
    'TextFeatureExtractor',
    'MultiModalFusion',
    'MultiModalReviewAnalyzer',
    'create_model',
    'AmazonReviewDataset',
    'create_data_loaders',
    'prepare_sample_data',
    'Trainer',
    'Evaluator',
    'Predictor',
    'load_model'
]
