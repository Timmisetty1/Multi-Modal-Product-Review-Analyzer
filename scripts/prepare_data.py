"""
Script to prepare Amazon review dataset.

This script helps prepare the data for multi-modal learning.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import prepare_sample_data


def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test sets.
    
    Args:
        df: DataFrame with reviews
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    test_df = df[val_end:].copy()
    
    return train_df, val_df, test_df


def prepare_amazon_data(input_path, output_dir, image_dir):
    """
    Prepare Amazon review data from raw format.
    
    Args:
        input_path: Path to raw data file
        output_dir: Directory to save processed data
        image_dir: Directory containing product images
    """
    print(f"Loading data from {input_path}...")
    
    # Load data
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.json'):
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError("Input file must be CSV or JSON")
    
    print(f"Loaded {len(df)} reviews")
    
    # Validate required columns
    required_cols = ['image_id', 'review_text', 'rating']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Clean data
    print("Cleaning data...")
    
    # Remove rows with missing values
    df = df.dropna(subset=required_cols)
    
    # Ensure ratings are in range 1-5
    df = df[df['rating'].between(1, 5)]
    
    # Convert rating to int
    df['rating'] = df['rating'].astype(int)
    
    print(f"After cleaning: {len(df)} reviews")
    
    # Split data
    print("Splitting data...")
    train_df, val_df, test_df = split_data(df)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Data saved to {output_dir}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    for name, data in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name} Set:")
        print(f"  Total samples: {len(data)}")
        print(f"  Rating distribution:")
        for rating in range(1, 6):
            count = (data['rating'] == rating).sum()
            pct = count / len(data) * 100
            print(f"    Rating {rating}: {count} ({pct:.1f}%)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Prepare Amazon review data for multi-modal learning'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input data file (CSV or JSON)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/images',
        help='Directory containing product images'
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample data for demonstration'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of sample reviews to generate (if --create-sample)'
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        print(f"Creating {args.num_samples} sample reviews...")
        prepare_sample_data(args.output, args.num_samples)
        print("\nSample data created successfully!")
        print(f"Location: {args.output}")
        print(f"  - Reviews: {args.output}/reviews.csv")
        print(f"  - Images: {args.output}/images/")
        
        # Split the sample data
        df = pd.read_csv(os.path.join(args.output, 'reviews.csv'))
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        
        train_df.to_csv(os.path.join(args.output, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(args.output, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(args.output, 'test.csv'), index=False)
        
    elif args.input:
        prepare_amazon_data(args.input, args.output, args.image_dir)
    else:
        parser.print_help()
        print("\nError: Either --input or --create-sample must be specified")


if __name__ == '__main__':
    main()
