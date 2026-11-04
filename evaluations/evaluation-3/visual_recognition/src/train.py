import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
from src.model import create_cnn_model, compile_model, save_model
from src.data_loader import load_train, load_test


def setup_directories() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)


def train_model(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Dict[str, Any]:
    setup_directories()
    
    train_ds, class_names = load_train(batch_size=batch_size)
    test_ds, _ = load_test(batch_size=batch_size)
    
    model = create_cnn_model()
    model = compile_model(model, learning_rate=learning_rate)
    
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        verbose=0
    )
    
    save_model(model, 'models/coffee_model.keras')
    
    plot_training_history(history)
    
    return {
        'history': history,
        'model': model,
        'class_names': class_names
    }


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión del entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de validación')
    plt.title('Precisión del modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida del entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title('Pérdida del modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_plots.png')
    plt.close()


if __name__ == "__main__":
    print("Comienza el entrenamiento del modelo.")
    results = train_model(
        epochs=10,
        batch_size=32,
        learning_rate=0.001
    )
    print("Entrenamiento completado.")
    print(f"Precisión más alta validada: {max(results['history'].history['val_accuracy']):.4f}")