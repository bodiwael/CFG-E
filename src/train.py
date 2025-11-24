#!/usr/bin/env python3
"""
Training Script for Malware Classification using GNN
"""

import torch
import torch.nn.functional as F
import os
import yaml
import logging
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np

from dataset import create_data_loaders
from model import create_model, count_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, optimizer, device, class_weights=None):
    """
    Train for one epoch

    Args:
        model: The neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        class_weights: Optional class weights for imbalanced data

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Set class weights if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(data)

        # Compute loss
        if class_weights is not None:
            loss = F.nll_loss(out, data.y, weight=class_weights)
        else:
            loss = F.nll_loss(out, data.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None):
    """
    Evaluate the model

    Args:
        model: The neural network model
        loader: Data loader (validation or test)
        device: Device to evaluate on
        class_weights: Optional class weights

    Returns:
        tuple: (average_loss, accuracy, predictions, labels)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    # Set class weights if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)

    for data in loader:
        data = data.to(device)

        # Forward pass
        out = model(data)

        # Compute loss
        if class_weights is not None:
            loss = F.nll_loss(out, data.y, weight=class_weights)
        else:
            loss = F.nll_loss(out, data.y)

        # Get predictions
        pred = out.argmax(dim=1)
        probs = torch.exp(out)  # Convert log softmax to probabilities

        # Store results
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_model(config):
    """
    Main training function

    Args:
        config: Configuration dictionary
    """
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output_dir'], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Create data loaders
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, dataset = create_data_loaders(
        benign_dir=config['data']['benign_dir'],
        malware_dir=config['data']['malware_dir'],
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        num_workers=config['training']['num_workers'],
        seed=config['seed']
    )

    if train_loader is None:
        logger.error("Failed to create data loaders")
        return

    # Get class weights for imbalanced data
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = dataset.get_class_weights()
        logger.info(f"Class weights: {class_weights}")

    # Create model
    logger.info(f"Creating model: {config['model']['type']}")
    model = create_model(
        model_type=config['model']['type'],
        num_node_features=config['model']['num_features'],
        hidden_channels=config['model']['hidden_channels'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        pooling=config['model']['pooling']
    ).to(device)

    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience']
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    best_val_acc = 0.0
    best_epoch = 0

    # Training loop
    logger.info("Starting training...")
    logger.info("=" * 60)

    for epoch in range(config['training']['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, class_weights
        )

        # Validate
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, device, class_weights
        )

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Log progress
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']} | "
                   f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                   f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                   f"LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pt'))
            logger.info(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    logger.info("=" * 60)
    logger.info(f"Training complete! Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set with best model
    logger.info("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, device, class_weights
    )

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'predictions': test_preds.tolist(),
        'labels': test_labels.tolist(),
        'probabilities': test_probs.tolist()
    }

    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"\nAll results saved to: {output_dir}")

    return model, history, output_dir


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train GNN for malware classification')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')

    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.error(f"Config file not found: {args.config}")
        return

    # Train model
    train_model(config)


if __name__ == '__main__':
    main()
