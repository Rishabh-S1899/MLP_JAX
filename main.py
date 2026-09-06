"""Run a small JAX MLP on deterministically generated synthetic data."""

from dataclasses import dataclass

import numpy as np

from mlp_jax.data import (
    assign_class_labels,
    one_hot_encode_labels,
    sample_input_features,
)
from mlp_jax.model import init_mlp_params, mlp_forward, predict_classes
from mlp_jax.random import make_prng_key, split_prng_key
from mlp_jax.training import (
    classification_accuracy,
    cross_entropy_loss,
    train_mlp,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the example training run."""

    seed: int = 0
    batch_size: int = 32
    num_features: int = 8
    num_classes: int = 4
    hidden_size: int = 16
    learning_rate: float = 0.1
    num_epochs: int = 200


def run_demo(config: TrainingConfig = TrainingConfig()) -> None:
    """Train and evaluate an MLP on a simple synthetic classification task."""
    root_key = make_prng_key(config.seed)
    data_key, init_key = split_prng_key(root_key, 2)

    x = sample_input_features(data_key, config.batch_size, config.num_features)
    labels = assign_class_labels(x, config.num_classes)
    one_hot_targets = one_hot_encode_labels(labels, config.num_classes)

    layer_sizes = [
        config.num_features,
        config.hidden_size,
        config.hidden_size,
        config.num_classes,
    ]
    params = init_mlp_params(init_key, layer_sizes)

    initial_logits = mlp_forward(params, x)
    print("Input shape:", x.shape)
    print("Labels[:8]:", np.asarray(labels[:8]).tolist())
    print("One-hot[0]:", np.asarray(one_hot_targets[0]).tolist())
    print("Num layers:", len(params))
    print(
        "Layer shapes:",
        [(layer["W"].shape, layer["b"].shape) for layer in params],
    )
    print(f"Initial loss:     {float(cross_entropy_loss(initial_logits, one_hot_targets)):.4f}")
    print(f"Initial accuracy: {float(classification_accuracy(initial_logits, labels)):.4f}")

    trained_params = train_mlp(
        params,
        x,
        one_hot_targets,
        config.learning_rate,
        config.num_epochs,
    )
    final_logits = mlp_forward(trained_params, x)
    predictions = predict_classes(trained_params, x)
    print(f"Final loss:       {float(cross_entropy_loss(final_logits, one_hot_targets)):.4f}")
    print(f"Final accuracy:   {float(classification_accuracy(final_logits, labels)):.4f}")
    print("Preds[:8]: ", np.asarray(predictions[:8]).tolist())
    print("Labels[:8]:", np.asarray(labels[:8]).tolist())


if __name__ == "__main__":
    run_demo()
