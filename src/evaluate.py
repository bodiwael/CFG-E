#!/usr/bin/env python3
"""
Evaluation Script for Malware Classification Model
Provides detailed metrics, confusion matrix, and ROC curve
"""

import torch
import numpy as np
import json
import os
import logging
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm, output_path, class_names=['Benign', 'Malware']):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        output_path: Path to save the plot
        class_names: Names of classes
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Add percentages
    total = np.sum(cm)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(labels, probs, output_path):
    """
    Plot ROC curve

    Args:
        labels: True labels
        probs: Predicted probabilities for positive class
        output_path: Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")

    return roc_auc


def plot_precision_recall_curve(labels, probs, output_path):
    """
    Plot Precision-Recall curve

    Args:
        labels: True labels
        probs: Predicted probabilities for positive class
        output_path: Path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Precision-Recall curve saved to {output_path}")

    return avg_precision


def plot_training_history(history_path, output_dir):
    """
    Plot training history (loss and accuracy curves)

    Args:
        history_path: Path to history JSON file
        output_dir: Directory to save plots
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training history saved to {output_path}")


def evaluate_model(results_dir):
    """
    Evaluate model results and generate detailed metrics

    Args:
        results_dir: Directory containing model results
    """
    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)

    # Load test results
    results_path = os.path.join(results_dir, 'test_results.json')
    if not os.path.exists(results_path):
        logger.error(f"Test results not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    labels = np.array(results['labels'])
    predictions = np.array(results['predictions'])
    probabilities = np.array(results['probabilities'])

    # Get probabilities for malware class (class 1)
    malware_probs = probabilities[:, 1]

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("-" * 60)
    class_names = ['Benign', 'Malware']
    report = classification_report(labels, predictions,
                                   target_names=class_names,
                                   digits=4)
    logger.info("\n" + report)

    # Save classification report
    report_dict = classification_report(labels, predictions,
                                       target_names=class_names,
                                       output_dict=True)
    with open(os.path.join(results_dir, 'classification_report.json'), 'w') as f:
        json.dump(report_dict, f, indent=2)

    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    logger.info("-" * 60)
    cm = confusion_matrix(labels, predictions)
    logger.info(f"\n{cm}")
    logger.info(f"\nTrue Negatives (TN):  {cm[0, 0]}")
    logger.info(f"False Positives (FP): {cm[0, 1]} (Benign misclassified as Malware)")
    logger.info(f"False Negatives (FN): {cm[1, 0]} (Malware misclassified as Benign)")
    logger.info(f"True Positives (TP):  {cm[1, 1]}")

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
    # False Negative Rate (Miss Rate)
    fnr_value = fn / (fn + tp) if (fn + tp) > 0 else 0
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    logger.info(f"\nAdditional Metrics:")
    logger.info(f"False Positive Rate: {fpr_value:.4f}")
    logger.info(f"False Negative Rate: {fnr_value:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")

    # ROC curve and AUC
    roc_auc = plot_roc_curve(
        labels, malware_probs,
        os.path.join(results_dir, 'roc_curve.png')
    )
    logger.info(f"\nROC AUC Score: {roc_auc:.4f}")

    # Precision-Recall curve
    avg_precision = plot_precision_recall_curve(
        labels, malware_probs,
        os.path.join(results_dir, 'precision_recall_curve.png')
    )
    logger.info(f"Average Precision Score: {avg_precision:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        os.path.join(results_dir, 'confusion_matrix.png'),
        class_names=class_names
    )

    # Plot training history if available
    history_path = os.path.join(results_dir, 'history.json')
    if os.path.exists(history_path):
        plot_training_history(history_path, results_dir)

    # Summary metrics
    summary = {
        'accuracy': results['test_accuracy'],
        'precision_benign': report_dict['Benign']['precision'],
        'precision_malware': report_dict['Malware']['precision'],
        'recall_benign': report_dict['Benign']['recall'],
        'recall_malware': report_dict['Malware']['recall'],
        'f1_benign': report_dict['Benign']['f1-score'],
        'f1_malware': report_dict['Malware']['f1-score'],
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'false_positive_rate': fpr_value,
        'false_negative_rate': fnr_value,
        'specificity': specificity,
        'confusion_matrix': cm.tolist()
    }

    with open(os.path.join(results_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Evaluation complete! Results saved to: {results_dir}")
    logger.info("=" * 60)

    return summary


def compare_models(results_dirs, model_names, output_dir):
    """
    Compare multiple models

    Args:
        results_dirs: List of result directories
        model_names: List of model names
        output_dir: Directory to save comparison plots
    """
    logger.info("Comparing models...")

    os.makedirs(output_dir, exist_ok=True)

    # Collect metrics from all models
    metrics = []
    for results_dir in results_dirs:
        summary_path = os.path.join(results_dir, 'summary_metrics.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                metrics.append(json.load(f))
        else:
            logger.warning(f"Summary not found: {summary_path}")

    if not metrics:
        logger.error("No metrics found for comparison")
        return

    # Plot comparison
    metric_names = ['accuracy', 'precision_malware', 'recall_malware',
                   'f1_malware', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']

    x = np.arange(len(model_names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (metric_name, label) in enumerate(zip(metric_names, metric_labels)):
        values = [m[metric_name] for m in metrics]
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Model comparison saved to {output_dir}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing model results')

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        logger.error(f"Results directory not found: {args.results_dir}")
        return

    # Evaluate model
    evaluate_model(args.results_dir)


if __name__ == '__main__':
    main()
