#!/usr/bin/env python3
"""
PyTorch Dataset for CFG-based Malware Classification
"""

import torch
import os
import logging
from pathlib import Path
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MalwareCFGDataset(Dataset):
    """
    Dataset for loading pre-processed CFG data

    Args:
        benign_dir: Directory containing benign .pt files
        malware_dir: Directory containing malware .pt files
        transform: Optional transform to apply to data
        pre_transform: Optional pre-transform to apply to data
    """

    def __init__(self, benign_dir, malware_dir, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)

        self.benign_dir = benign_dir
        self.malware_dir = malware_dir

        # Load file paths
        self.file_paths = []
        self.labels = []

        # Load benign samples
        if os.path.exists(benign_dir):
            benign_files = [
                os.path.join(benign_dir, f)
                for f in os.listdir(benign_dir)
                if f.endswith('.pt')
            ]
            self.file_paths.extend(benign_files)
            self.labels.extend([0] * len(benign_files))
            logger.info(f"Loaded {len(benign_files)} benign samples")

        # Load malware samples
        if os.path.exists(malware_dir):
            malware_files = [
                os.path.join(malware_dir, f)
                for f in os.listdir(malware_dir)
                if f.endswith('.pt')
            ]
            self.file_paths.extend(malware_files)
            self.labels.extend([1] * len(malware_files))
            logger.info(f"Loaded {len(malware_files)} malware samples")

        logger.info(f"Total dataset size: {len(self.file_paths)}")

        if len(self.file_paths) == 0:
            logger.warning("No data files found!")

    def len(self):
        """Return the number of samples in the dataset"""
        return len(self.file_paths)

    def get(self, idx):
        """
        Get a single data sample

        Args:
            idx: Index of the sample

        Returns:
            Data: PyTorch Geometric Data object
        """
        file_path = self.file_paths[idx]

        try:
            # Load the data
            data = torch.load(file_path)
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            # Return a dummy data object in case of error
            return Data(
                x=torch.zeros((1, 10)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                y=torch.tensor([self.labels[idx]], dtype=torch.long)
            )

    def get_labels(self):
        """Return all labels"""
        return self.labels

    def get_class_weights(self):
        """
        Calculate class weights for imbalanced datasets

        Returns:
            torch.Tensor: Class weights [weight_benign, weight_malware]
        """
        num_benign = self.labels.count(0)
        num_malware = self.labels.count(1)
        total = len(self.labels)

        if num_benign == 0 or num_malware == 0:
            return torch.tensor([1.0, 1.0])

        # Inverse frequency weighting
        weight_benign = total / (2 * num_benign)
        weight_malware = total / (2 * num_malware)

        return torch.tensor([weight_benign, weight_malware])

    def get_statistics(self):
        """
        Get dataset statistics

        Returns:
            dict: Dataset statistics
        """
        num_benign = self.labels.count(0)
        num_malware = self.labels.count(1)

        stats = {
            'total_samples': len(self.labels),
            'benign_samples': num_benign,
            'malware_samples': num_malware,
            'benign_ratio': num_benign / len(self.labels) if len(self.labels) > 0 else 0,
            'malware_ratio': num_malware / len(self.labels) if len(self.labels) > 0 else 0
        }

        return stats


def create_data_loaders(benign_dir, malware_dir, batch_size=32,
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       num_workers=0, seed=42):
    """
    Create train/val/test data loaders with stratified split

    Args:
        benign_dir: Directory with benign .pt files
        malware_dir: Directory with malware .pt files
        batch_size: Batch size for data loaders
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility

    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train, val, and test ratios must sum to 1.0"

    # Load full dataset
    dataset = MalwareCFGDataset(benign_dir, malware_dir)

    # Get dataset statistics
    stats = dataset.get_statistics()
    logger.info(f"Dataset statistics: {stats}")

    if len(dataset) == 0:
        logger.error("Dataset is empty!")
        return None, None, None, dataset

    # Get labels for stratified split
    labels = dataset.get_labels()

    # Create indices
    indices = list(range(len(dataset)))

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )

    # Second split: val vs test
    temp_labels = [labels[i] for i in temp_indices]
    val_size = val_ratio / (val_ratio + test_ratio)

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=seed
    )

    logger.info(f"Split sizes - Train: {len(train_indices)}, "
                f"Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, dataset


def main():
    """Test the dataset"""
    import argparse

    parser = argparse.ArgumentParser(description='Test CFG dataset')
    parser.add_argument('--benign-dir', type=str, default='data/processed/benign',
                        help='Directory with benign .pt files')
    parser.add_argument('--malware-dir', type=str, default='data/processed/malware',
                        help='Directory with malware .pt files')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')

    args = parser.parse_args()

    logger.info("Testing dataset loader...")

    # Create data loaders
    train_loader, val_loader, test_loader, dataset = create_data_loaders(
        args.benign_dir,
        args.malware_dir,
        batch_size=args.batch_size
    )

    if train_loader is None:
        logger.error("Failed to create data loaders")
        return

    # Test loading a batch
    logger.info("\nTesting batch loading:")
    for batch in train_loader:
        logger.info(f"Batch shape: x={batch.x.shape}, "
                   f"edge_index={batch.edge_index.shape}, "
                   f"y={batch.y.shape}")
        logger.info(f"Labels in batch: {batch.y.tolist()}")
        logger.info(f"Number of graphs in batch: {batch.num_graphs}")
        break

    # Get class weights
    class_weights = dataset.get_class_weights()
    logger.info(f"\nClass weights: {class_weights}")

    logger.info("\nDataset test complete!")


if __name__ == '__main__':
    main()
