import tensorflow as tf
from typing import Callable, Tuple


def create_augmentation_pipeline() -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    
    def augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = data_augmentation(image, training=True)
        return image, label
    
    return augment


def apply_augmentation(
    dataset: tf.data.Dataset,
    is_training: bool = True
) -> tf.data.Dataset:
    if is_training:
        augmentation = create_augmentation_pipeline()
        dataset = dataset.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset


def get_augmentation_pipeline() -> Callable:
    return create_augmentation_pipeline()