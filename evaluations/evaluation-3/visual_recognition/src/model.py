import tensorflow as tf
from typing import Tuple


def create_cnn_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    dropout_rate: float = 0.5
) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255, input_shape=input_shape),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
    loss: str = 'sparse_categorical_crossentropy'
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def save_model(model: tf.keras.Model, filepath: str) -> None:
    model.save(filepath)


def load_model(filepath: str) -> tf.keras.Model:
    return tf.keras.models.load_model(filepath)