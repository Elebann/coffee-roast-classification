"""
imágenes para mostrar en el informe y explicar la arquitectura del modelo
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import Dict, List, Tuple

CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]
SAMPLE_IMAGE = "dataset/test/Dark/dark (1).png"
MODEL_PATH = "models/coffee_model.keras"
OUTPUT_DIR = Path("results/architecture")


def load_sample() -> Tuple[np.ndarray, np.ndarray]:
    # carga imagen de ejemplo en rangos 0-255 y 0-1
    img_uint8 = tf.keras.utils.load_img(SAMPLE_IMAGE, target_size=(224, 224))
    img_255 = tf.keras.utils.img_to_array(img_uint8)
    img_01 = img_255 / 255.0
    return img_255, img_01


def get_layer_outputs(model: tf.keras.Model, img: np.ndarray) -> Dict[int, np.ndarray]:
    # ejecuta la imagen por capas y devuelve salidas intermedias (Keras 3)
    batch = np.expand_dims(img, axis=0)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = inputs
    outputs: Dict[int, tf.Tensor] = {}

    for i, layer in enumerate(model.layers):
        x = layer(x)
        outputs[i] = x

    intermediate = tf.keras.Model(inputs=inputs, outputs=list(outputs.values()))
    values = intermediate.predict(batch, verbose=0)

    return {i: values[i] for i in range(len(model.layers))}


def save_fig(name: str) -> None:
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Guardado: {path}")


def plot_architecture_overview() -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")

    blocks = [
        ("Entrada\n224×224×3", "#E8F4FD"),
        ("Rescaling\n÷255", "#D4EDDA"),
        ("Conv2D\n32 filtros", "#FFF3CD"),
        ("MaxPool\n2×2", "#F8D7DA"),
        ("Conv2D\n64 filtros", "#FFF3CD"),
        ("MaxPool\n2×2", "#F8D7DA"),
        ("GAP", "#E2D5F1"),
        ("Dense\n128 ReLU", "#D1ECF1"),
        ("Dropout\n50%", "#F5C6CB"),
        ("Softmax\n4 clases", "#C3E6CB"),
    ]

    x = 0.3
    for i, (label, color) in enumerate(blocks):
        box = FancyBboxPatch(
            (x, 0.8), 1.15, 1.4,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + 0.575, 1.5, label, ha="center", va="center", fontsize=8, fontweight="bold")
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + 1.25, 1.5), xytext=(x + 1.15, 1.5),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))
        x += 1.35

    ax.set_title(
        "Arquitectura secuencial de la CNN para clasificación de tueste de café",
        fontsize=13, fontweight="bold", pad=12,
    )
    save_fig("01_arquitectura_general.png")


def plot_rescaling(img_255: np.ndarray, img_01: np.ndarray) -> None:
    # rescaling de píxeles
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(img_255.astype(np.uint8))
    axes[0, 0].set_title("Imagen original (valores 0–255)", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_01)
    axes[0, 1].set_title("Tras Rescaling (valores 0–1)", fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].hist(img_255.ravel(), bins=50, color="#4C72B0", edgecolor="white")
    axes[1, 0].set_title("Distribución de píxeles antes del Rescaling")
    axes[1, 0].set_xlabel("Intensidad del píxel")
    axes[1, 0].set_ylabel("Frecuencia")
    axes[1, 0].set_xlim(0, 255)

    axes[1, 1].hist(img_01.ravel(), bins=50, color="#55A868", edgecolor="white")
    axes[1, 1].set_title("Distribución de píxeles después del Rescaling")
    axes[1, 1].set_xlabel("Intensidad del píxel")
    axes[1, 1].set_ylabel("Frecuencia")
    axes[1, 1].set_xlim(0, 1)

    fig.suptitle(
        "Capa Rescaling: escala automática de píxeles para estabilizar el entrenamiento",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    save_fig("02_rescaling.png")


def plot_conv_blocks(img_01: np.ndarray, outputs: Dict[int, np.ndarray]) -> None:
    # bloques convolucionales, extracción de características
    conv1 = outputs[1][0]
    pool1 = outputs[2][0]
    conv2 = outputs[3][0]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.2], hspace=0.35, wspace=0.2)

    ax_in = fig.add_subplot(gs[0, :])
    ax_in.imshow(img_01)
    ax_in.set_title("Imagen de entrada: grano de café (clase Dark)", fontweight="bold")
    ax_in.axis("off")

    ax_c1 = fig.add_subplot(gs[1, 0])
    _show_feature_grid(ax_c1, conv1, n=16, title="1.er bloque — Conv2D (32 filtros 3×3)\nCaracterísticas básicas: bordes, formas, color")

    ax_p1 = fig.add_subplot(gs[1, 1])
    _show_feature_grid(ax_p1, pool1, n=16, title="Tras MaxPooling2D (2×2)\nReducción espacial: 224×224 → 112×112")

    ax_c2 = fig.add_subplot(gs[2, :])
    _show_feature_grid(ax_c2, conv2, n=16, cols=8, title="2.º bloque — Conv2D (64 filtros 3×3)\nPatrones más complejos: textura y nivel de tueste")

    fig.suptitle("Bloques convolucionales: extracción jerárquica de características visuales", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("03_bloques_convolucionales.png")


def _show_feature_grid(ax, features: np.ndarray, n: int, title: str, cols: int = 4) -> None:
    rows = int(np.ceil(n / cols))
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.axis("off")

    inset_size = 0.9 / cols
    for i in range(n):
        row, col = divmod(i, cols)
        left = 0.05 + col * inset_size
        bottom = 1.0 - (row + 1) * (0.9 / rows) - 0.05
        inset = ax.inset_axes([left, bottom, inset_size * 0.92, (0.9 / rows) * 0.92])
        inset.imshow(features[:, :, i], cmap="viridis")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(f"F{i}", fontsize=6)


def plot_global_average_pooling(outputs: Dict[int, np.ndarray]) -> None:
    # global average pooling 2D
    pool2 = outputs[4][0]
    gap = outputs[5][0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    channel = 10
    spatial_map = pool2[:, :, channel]
    axes[0].imshow(spatial_map, cmap="viridis")
    axes[0].set_title(f"Mapa espacial antes del GAP\n(canal {channel}, 56×56)", fontweight="bold")
    axes[0].axis("off")

    avg_val = float(np.mean(spatial_map))
    axes[1].barh([0], [avg_val], color="#8172B3", height=0.4)
    axes[1].set_xlim(0, max(gap) * 1.1)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Valor promedio del canal")
    axes[1].set_title(f"Promedio global del canal {channel}\n= {avg_val:.3f}", fontweight="bold")

    axes[2].bar(range(len(gap)), gap, color="#8172B3", edgecolor="white")
    axes[2].set_title("Vector de 64 características tras GAP", fontweight="bold")
    axes[2].set_xlabel("Índice de característica")
    axes[2].set_ylabel("Valor promedio")

    fig.suptitle(
        "GlobalAveragePooling2D: resume cada mapa espacial en un valor, reduciendo parámetros",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    save_fig("04_global_average_pooling.png")


def plot_dense_layer(outputs: Dict[int, np.ndarray]) -> None:
    # capa Dense 128 neuronas
    dense_out = outputs[6][0]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in dense_out]
    ax.bar(range(len(dense_out)), dense_out, color=colors, edgecolor="none", width=1.0)
    ax.set_title("Activaciones ReLU de la capa Dense (128 neuronas)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Neurona")
    ax.set_ylabel("Activación")
    ax.axhline(0, color="black", linewidth=0.5)

    top_idx = np.argsort(dense_out)[-5:][::-1]
    for idx in top_idx:
        ax.annotate(f"n{idx}", (idx, dense_out[idx]), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=7, color="#333333")

    fig.suptitle(
        "Capa completamente conectada: combina las características para la clasificación",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_fig("05_capa_densa.png")


def plot_dropout(outputs: Dict[int, np.ndarray]) -> None:
    # dropout al 50%
    dense_out = outputs[6][0]
    rng = np.random.default_rng(42)
    mask = rng.random(len(dense_out)) > 0.5
    dropped = dense_out.copy()
    dropped[~mask] = 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(range(len(dense_out)), dense_out, color="#4C72B0", edgecolor="none", width=1.0)
    axes[0].set_title("Activaciones completas (128 neuronas)", fontweight="bold")
    axes[0].set_xlabel("Neurona")
    axes[0].set_ylabel("Activación")

    colors = ["#C44E52" if m else "#DDDDDD" for m in mask]
    axes[1].bar(range(len(dropped)), dropped, color=colors, edgecolor="none", width=1.0)
    axes[1].set_title("Tras Dropout (50%) — solo en entrenamiento", fontweight="bold")
    axes[1].set_xlabel("Neurona")
    axes[1].set_ylabel("Activación")

    active = int(mask.sum())
    axes[1].text(0.98, 0.95, f"Neuronas activas: {active}/128",
                 transform=axes[1].transAxes, ha="right", va="top",
                 bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#856404"))

    fig.suptitle(
        "Dropout (50%): desactiva neuronas aleatoriamente para mejorar la generalización",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    save_fig("06_dropout.png")


def plot_softmax_output(img_01: np.ndarray, outputs: Dict[int, np.ndarray]) -> None:
    # softmax con probabilidades por clase
    probs = outputs[8][0]
    predicted = CLASS_NAMES[int(np.argmax(probs))]

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.3)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img_01)
    ax_img.set_title("Imagen evaluada", fontweight="bold")
    ax_img.axis("off")

    ax_bar = fig.add_subplot(gs[1])
    bar_colors = ["#55A868" if name == predicted else "#4C72B0" for name in CLASS_NAMES]
    bars = ax_bar.bar(CLASS_NAMES, probs, color=bar_colors, edgecolor="white")
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Probabilidad")
    ax_bar.set_title(f"Salida Softmax — Predicción: {predicted}", fontweight="bold")

    for bar, prob in zip(bars, probs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{prob:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Capa de salida: distribución de probabilidades sobre las 4 clases de tueste",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    save_fig("07_softmax_salida.png")


def generate_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando modelo e imagen de ejemplo...")
    model = tf.keras.models.load_model(MODEL_PATH)
    img_255, img_01 = load_sample()
    outputs = get_layer_outputs(model, img_01)

    print("Generando visualizaciones:")
    plot_architecture_overview()
    plot_rescaling(img_255, img_01)
    plot_conv_blocks(img_01, outputs)
    plot_global_average_pooling(outputs)
    plot_dense_layer(outputs)
    plot_dropout(outputs)
    plot_softmax_output(img_01, outputs)

    print(f"\nListo. {len(list(OUTPUT_DIR.glob('*.png')))} imágenes en {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_all()
