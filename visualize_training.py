"""
Herramienta para visualizar gráficamente el historial de entrenamiento
a partir de un checkpoint guardado por ChessTrainer.

Uso:
    python visualize_training.py --checkpoint models/final_model.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch


def load_history(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint.get("training_history")
    if history is None:
        raise KeyError(
            "El checkpoint no contiene 'training_history'. "
            "Asegúrate de que fue generado por ChessTrainer."
        )
    return history


def plot_history(history: dict):
    epochs = list(range(1, len(history["total_loss"]) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["policy_loss"], label="Policy Loss")
    plt.plot(epochs, history["value_loss"], label="Value Loss")
    plt.plot(epochs, history["total_loss"], label="Total Loss", linewidth=2)

    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Historial de entrenamiento del bot de ajedrez")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualizar gráficamente el historial de entrenamiento desde un checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al archivo .pt guardado (por ejemplo: models/final_model.pt)",
    )

    args = parser.parse_args()

    history = load_history(args.checkpoint)
    plot_history(history)


if __name__ == "__main__":
    main()

