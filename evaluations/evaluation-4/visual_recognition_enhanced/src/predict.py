import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from src.data_loader import load_single_image
from src.model import load_model


def predict_image(
    image_path: str,
    model_path: str = 'models/best_model.keras',
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"Loading image from {image_path}...")
    img = load_single_image(image_path)
    
    # agregar dimension de batch
    img = tf.expand_dims(img, axis=0)
    
    print("Making prediction...")
    predictions = model.predict(img)
    
    # obtener clase predicha y confianza
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # obtener las 4 predicciones mas confiables
    top_4_indices = np.argsort(predictions[0])[-4:][::-1]
    top_4_classes = []
    top_4_probabilities = []
    
    # usar nombres de clases por defecto si no se proporcionan
    if class_names is None:
        class_names = ["Dark", "Green", "Light", "Medium"]
    
    for idx in top_4_indices:
        class_name = class_names[idx]
        top_4_classes.append(class_name)
        top_4_probabilities.append(float(predictions[0][idx]))
    
    # determinar nombre de clase predicha
    predicted_class = class_names[predicted_class_idx]
    
    results = {
        'image_path': image_path,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top_4_classes': top_4_classes,
        'top_4_probabilities': top_4_probabilities,
        'all_predictions': predictions[0].tolist(),
        'class_names': class_names
    }
    
    return results


def print_prediction_results(results: Dict[str, Any]) -> None:
    print("\n=== Prediction Results ===")
    print(f"Image: {results['image_path']}")
    print(f"Predicted Class: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.4f}")
    
    print("\nTop 4 Predictions:")
    for i, (class_name, prob) in enumerate(
        zip(results['top_4_classes'], results['top_4_probabilities']),
        start=1
    ):
        print(f"  {i}. {class_name}: {prob:.4f}")


def save_prediction_results(results: Dict[str, Any], output_path: str = None) -> None:
    if output_path is None:
        output_path = Path("results/predictions.txt")
    
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"=== Prediction Results ===\n")
        f.write(f"Image: {results['image_path']}\n")
        f.write(f"Predicted Class: {results['predicted_class']}\n")
        f.write(f"Confidence: {results['confidence']:.4f}\n\n")
        
        f.write("Top 4 Predictions:\n")
        for i, (class_name, prob) in enumerate(
            zip(results['top_4_classes'], results['top_4_probabilities']),
            start=1
        ):
            f.write(f"  {i}. {class_name}: {prob:.4f}\n")
    
    print(f"Prediction results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict coffee roast class from image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model_path', default='models/best_model.keras', help='Path to the trained model')
    
    args = parser.parse_args()
    
    print("Starting image prediction...")
    results = predict_image(args.image_path, args.model_path)
    print_prediction_results(results)
    save_prediction_results(results)
    print("Prediction completed successfully!")