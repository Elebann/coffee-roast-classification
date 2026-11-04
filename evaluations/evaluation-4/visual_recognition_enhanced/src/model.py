import tensorflow as tf
from typing import Optional, Tuple, List


def create_cnn_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    dropout_rate: float = 0.5,
    use_transfer_learning: bool = False,
    base_model_name: str = "MobileNetV2"
) -> tf.keras.Model:
    if use_transfer_learning:
        return _create_transfer_learning_model(
            input_shape, num_classes, base_model_name
        )
    else:
        return _create_custom_cnn_model(input_shape, num_classes, dropout_rate)


def _create_custom_cnn_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float
) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255, input_shape=input_shape),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def _create_transfer_learning_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    base_model_name: str
) -> tf.keras.Model:
    if base_model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Rescaling(1.0 / 255),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
    loss: str = 'sparse_categorical_crossentropy',
    metrics: Optional[List[str]] = None
) -> tf.keras.Model:
    if metrics is None:
        metrics = ['accuracy']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model


def save_model(model: tf.keras.Model, filepath: str) -> None:
    model.save(filepath)


def load_model(filepath: str) -> tf.keras.Model:
    return tf.keras.models.load_model(filepath)