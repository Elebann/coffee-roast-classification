import tensorflow as tf
from typing import Callable, Tuple


def create_preprocessing_pipeline(
    is_training: bool = False,
    target_size: Tuple[int, int] = (224, 224),
    normalization: bool = True,
    cache: bool = True,
    prefetch: bool = True
) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # redimensionar imagen
        image = tf.image.resize(image, target_size)
        
        return image, label
    
    def pipeline(dataset: tf.data.Dataset) -> tf.data.Dataset:
        if cache:
            dataset = dataset.cache()
        
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if prefetch:
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    
    return pipeline


def get_preprocessing_pipeline(
    is_training: bool = False,
    target_size: Tuple[int, int] = (224, 224)
) -> Callable:
    if is_training:
        return create_preprocessing_pipeline(
            is_training=True,
            target_size=target_size,
            normalization=True,
            cache=True,
            prefetch=True
        )
    else:
        return create_preprocessing_pipeline(
            is_training=False,
            target_size=target_size,
            normalization=True,
            cache=True,
            prefetch=True
        )