import tensorflow as tf
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Dict, Any, List
from src.model import create_cnn_model, compile_model, save_model
from src.data_loader import load_train, load_test
from src.preprocessing import get_preprocessing_pipeline
from src.augmentation import apply_augmentation
import numpy as np


def setup_directories() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)


def create_callbacks() -> List[tf.keras.callbacks.Callback]:
    # early stopping para prevenir sobreajuste
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
    
    # reducir tasa de aprendizaje cuando el loss de validacion se estabiliza
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # checkpoint para guardar el mejor modelo
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    return [early_stopping, reduce_lr, checkpoint]


def train_model(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_augmentation: bool = True,
    use_transfer_learning: bool = False
) -> Dict[str, Any]:
    print("Setting up directories...")
    setup_directories()
    
    print("Loading training data...")
    train_ds, class_names = load_train(batch_size=batch_size)

    # verificar distribucion de etiquetas
    all_labels = []

    for _, labels in train_ds:
        all_labels.extend(labels.numpy())

    unique, counts = np.unique(all_labels, return_counts=True)

    print("Distribución de clases:")
    print(dict(zip(unique, counts)))
    
    print("Loading test data...")
    test_ds, _ = load_test(batch_size=batch_size)
    
    print(f"Class names: {class_names}")
    
    # aplicar pipeline de preprocesamiento
    print("Applying preprocessing pipeline...")
    preprocess_pipeline = get_preprocessing_pipeline(is_training=True)
    train_ds = preprocess_pipeline(train_ds)
    test_ds = preprocess_pipeline(test_ds)
    
    # aplicar data augmentation si esta habilitado
    if use_augmentation:
        print("Applying data augmentation...")
        train_ds = apply_augmentation(train_ds, is_training=True)
    
    print("Creating model...")
    model = create_cnn_model(
        use_transfer_learning=use_transfer_learning,
        dropout_rate=0.5
    )
    
    print("Compiling model...")
    model = compile_model(model, learning_rate=learning_rate)
    
    model.summary()
    
    print("Creating callbacks...")
    callbacks = create_callbacks()
    
    print("Starting training...")
    for images, labels in train_ds.take(1):
        print("Shape:", images.shape)
        print("Labels:", labels.numpy()[:20])
        print("Min:", tf.reduce_min(images).numpy())
        print("Max:", tf.reduce_max(images).numpy())
        print("Primer pixel:", images[0, 0, 0].numpy())
        
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Saving final model...")
    save_model(model, 'models/coffee_model.keras')
    
    print("Generating training plots...")
    plot_training_history(history)
    
    return {
        'history': history,
        'model': model,
        'class_names': class_names
    }


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    # plotear y guardar historial de entrenamiento
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # plotear loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_plots.png')
    plt.close()


def load_trained_model(model_path: str) -> tf.keras.Model:
    # cargar modelo entrenado desde archivo
    return tf.keras.models.load_model(model_path)


if __name__ == "__main__":
    print("Starting coffee roast classification model training...")
    results = train_model(
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        use_augmentation=True,
        use_transfer_learning=False
    )
    print("Training completed successfully!")
    print(f"Best validation accuracy: {max(results['history'].history['val_accuracy']):.4f}")