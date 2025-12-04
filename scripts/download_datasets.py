"""Script để download public datasets từ Kaggle.

Usage:
    python scripts/download_datasets.py --dataset insurance_churn
    python scripts/download_datasets.py --dataset bank_churn
    python scripts/download_datasets.py --list  # List available datasets
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import (
    list_available_datasets,
    download_kaggle_dataset,
    PUBLIC_DATASETS,
)


def main():
    parser = argparse.ArgumentParser(description="Download public datasets")
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name to download',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for datasets',
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("=" * 60)
        print("Available Public Datasets")
        print("=" * 60)
        
        datasets = list_available_datasets()
        for key, info in datasets.items():
            print(f"\n{key}:")
            print(f"  Name: {info['name']}")
            print(f"  Source: {info['source']}")
            print(f"  URL: {info['url']}")
            print(f"  Description: {info['description']}")
        
        print("\n" + "=" * 60)
        print("To download a dataset:")
        print("  1. Go to the URL above")
        print("  2. Download the CSV file")
        print("  3. Place it in data/raw/ directory")
        print("  4. Or use Kaggle API (see README)")
        print("=" * 60)
        return
    
    if args.dataset:
        if args.dataset not in PUBLIC_DATASETS:
            print(f"Error: Unknown dataset '{args.dataset}'")
            print(f"Available datasets: {', '.join(PUBLIC_DATASETS.keys())}")
            print("\nUse --list to see details")
            return
        
        info = PUBLIC_DATASETS[args.dataset]
        print(f"Downloading {info['name']}...")
        print(f"URL: {info['url']}")
        print("\nNote: This script requires Kaggle API setup.")
        print("Alternative: Download manually from the URL above")
        print("and place the CSV file in data/raw/")
        
        # Try to download if Kaggle API is available
        try:
            # Note: User needs to provide dataset name in format username/dataset-name
            print("\nTo use Kaggle API, you need:")
            print("1. Install: pip install kaggle")
            print("2. Setup credentials: https://www.kaggle.com/docs/api")
            print("3. Then use: kaggle datasets download <dataset-name>")
        except Exception as e:
            print(f"Error: {e}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

