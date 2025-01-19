import argparse


def get_train_args():
    parser = argparse.ArgumentParser(description='Custom Dataset Training')

    # Data related arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing all images')
    parser.add_argument('--annotation-file', type=str, required=True,
                        help='Path to COCO format annotation file')
    parser.add_argument('--train-dir', type=str,
                        help='Directory containing training data (if already split)')
    parser.add_argument('--test-dir', type=str,
                        help='Directory containing test data (if already split)')
    parser.add_argument('--no-split', action='store_true',
                        help='Skip dataset splitting (use existing train/test dirs)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')

    # Training related arguments
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')

    # キャッシュ関連の引数
    parser.add_argument('--use-cache', action='store_true',
                        help='Enable dataset caching for faster loading')
    parser.add_argument('--cache-dir', type=str,
                        help='Directory to store cached data (default: data_dir/.cache)')

    args = parser.parse_args()
    return args
