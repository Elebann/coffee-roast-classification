import tensorflow as tf
from pathlib import Path
from typing import Tuple, List


def load_train(batch_size: int = 32) -> Tuple[tf.data.Dataset, List[str]]:
    dataset_path = Path("dataset/train")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="int",
        class_names=["Dark", "Green", "Light", "Medium"],
        shuffle=True
    )
    
    class_names = train_ds.class_names
    return train_ds, class_names


def load_test(batch_size: int = 32) -> Tuple[tf.data.Dataset, List[str]]:
    dataset_path = Path("dataset/test")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="int",
        class_names=["Dark", "Green", "Light", "Medium"],
        shuffle=False
    )
    
    class_names = test_ds.class_names
    return test_ds, class_names


def load_single_image(image_path: str) -> tf.Tensor:
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    return img