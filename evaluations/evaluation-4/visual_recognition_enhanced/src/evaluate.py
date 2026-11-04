import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import load_test
from src.preprocessing import get_preprocessing_pipeline
from src.model import load_model


def evaluate_model(
    model_path: str = 'models/best_model.keras',
    batch_size: int = 32
) -> Dict[str, Any]:
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print("Loading test data...")
    test_ds, class_names = load_test(batch_size=batch_size)
    
    print("Applying preprocessing pipeline...")
    preprocess_pipeline = get_preprocessing_pipeline(is_training=False)
    test_ds = preprocess_pipeline(test_ds)
    
    print("Evaluating model...")
    results = model.evaluate(test_ds, verbose=1)
    
    print("Generating predictions...")
    y_pred = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    y_true = []
    for _, label in test_ds:
        y_true.extend(label.numpy())
    y_true = np.array(y_true)
    
    print("Generating metrics...")
    metrics = generate_metrics(y_true, y_pred_classes, class_names)
    
    print("Generating plots...")
    plot_confusion_matrix(y_true, y_pred_classes, class_names)
    
    save_evaluation_results(metrics, results)
    
    return {
        'model_metrics': metrics,
        'evaluation_results': results,
        'class_names': class_names,
        'predictions': y_pred,
        'true_labels': y_true,
        'predicted_labels': y_pred_classes
    }


def generate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    # calcular las metricas
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    # metricas por clase
    precision = {}
    recall = {}
    f1_score = {}
    
    for i, class_name in enumerate(class_names):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        
        precision[class_name] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[class_name] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[class_name] = 2 * (precision[class_name] * recall[class_name]) / (
            precision[class_name] + recall[class_name]
        ) if (precision[class_name] + recall[class_name]) > 0 else 0
    
    # metricas generales
    avg_precision = np.mean(list(precision.values()))
    avg_recall = np.mean(list(recall.values()))
    avg_f1 = np.mean(list(f1_score.values()))
    
    # reporte de clasificacion
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'classification_report': report
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()


def save_evaluation_results(
    metrics: Dict[str, Any],
    evaluation_results: List[float]
) -> None:
    results_path = Path('results/evaluation_results.txt')
    
    with open(results_path, 'w') as f:
        f.write('=== Coffee Roast Classification Model Evaluation ===\n\n')
        
        f.write(f'Loss: {evaluation_results[0]:.4f}\n')
        f.write(f'Accuracy: {evaluation_results[1]:.4f}\n\n')
        
        f.write('=== Per-Class Metrics ===\n')
        for class_name in metrics['precision'].keys():
            f.write(f'{class_name}:\n')
            f.write(f'  Precision: {metrics["precision"][class_name]:.4f}\n')
            f.write(f'  Recall: {metrics["recall"][class_name]:.4f}\n')
            f.write(f'  F1-Score: {metrics["f1_score"][class_name]:.4f}\n\n')
        
        f.write(f'Average Precision: {metrics["avg_precision"]:.4f}\n')
        f.write(f'Average Recall: {metrics["avg_recall"]:.4f}\n')
        f.write(f'Average F1-Score: {metrics["avg_f1"]:.4f}\n')
    
    print(f"Evaluation results saved to {results_path}")


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    print("\n=== Evaluation Summary ===")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average Precision: {metrics['avg_precision']:.4f}")
    print(f"Average Recall: {metrics['avg_recall']:.4f}")
    print(f"Average F1-Score: {metrics['avg_f1']:.4f}")
    
    print("\n=== Per-Class Metrics ===")
    for class_name in metrics['precision'].keys():
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision'][class_name]:.4f}")
        print(f"  Recall: {metrics['recall'][class_name]:.4f}")
        print(f"  F1-Score: {metrics['f1_score'][class_name]:.4f}")


if __name__ == "__main__":
    print("Starting model evaluation...")
    results = evaluate_model()
    print_evaluation_summary(results['model_metrics'])
    print("Evaluation completed successfully!")