# MLP in JAX

A compact, from-scratch implementation of a multi-layer perceptron (MLP) using
[JAX](https://docs.jax.dev/). The project generates a synthetic classification
dataset, trains a fully connected neural network with gradient descent, and
prints metrics before and after training.

The implementation is intended as a readable learning example. It uses JAX for
array operations, automatic differentiation, and deterministic pseudo-random
number generation; it does not depend on a high-level neural-network framework.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) (recommended), or another Python package
  manager

JAX's installed build determines the available hardware backend. The project
runs on the CPU with the dependencies specified in `pyproject.toml`.

## Setup

From the repository root, create the project environment and install the locked
dependencies:

```powershell
uv sync
```

If you are not using uv, create and activate a virtual environment, then install
the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Run the example

With uv:

```powershell
uv run python main.py
```

Or, after activating the virtual environment:

```powershell
python main.py
```

The output includes the input and parameter shapes, initial loss and accuracy,
and final loss and accuracy. With the default deterministic seed, the model
should learn the generated batch to high accuracy.

## Configuration

`main.py` defines a `TrainingConfig` dataclass. Its default values are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `seed` | 0 | Seed used for JAX pseudo-random keys |
| `batch_size` | 32 | Number of synthetic samples |
| `num_features` | 8 | Features in each sample |
| `num_classes` | 4 | Output classes |
| `hidden_size` | 16 | Units in each of the two hidden layers |
| `learning_rate` | 0.1 | SGD update step size |
| `num_epochs` | 200 | Number of full-batch updates |

To run with another configuration, import `run_demo` and provide a config:

```python
from main import TrainingConfig, run_demo

run_demo(TrainingConfig(hidden_size=32, learning_rate=0.05, num_epochs=500))
```

## Project structure

```text
main.py                 Executable demo and training configuration
mlp_jax/
  random.py             PRNG keys and normal sampling
  data.py               Synthetic features, labels, and one-hot targets
  layers.py             Dense-layer initialization, affine pass, and ReLU
  model.py              MLP initialization, forward pass, softmax, prediction
  training.py           Loss, accuracy, gradients, SGD, and training loop
```

## Implementation approach

The training pipeline is deliberately split into small, independently reusable
functions:

1. A root JAX PRNG key is created and split into separate keys for data
   generation and parameter initialization. This makes every run reproducible.
2. Input features are sampled from a standard-normal distribution. Labels are
   assigned using the index of the largest value among the first
   `num_classes` features, producing a learnable synthetic task.
3. The network is built from dense layers with weight matrices and bias vectors.
   Hidden layers use ReLU activation; the final layer returns unnormalized
   logits.
4. Cross-entropy is evaluated using `jax.nn.log_softmax`, which provides a
   numerically stable loss calculation. Accuracy compares the highest-logit
   class against the assigned label.
5. `jax.grad` differentiates the loss with respect to every weight and bias.
   Full-batch stochastic gradient descent applies the resulting gradients for
   the configured number of epochs.

Parameters are represented as a list of dictionaries, one dictionary per
layer, with `"W"` for weights and `"b"` for biases. This is a native JAX
pytree, so it can be passed directly to `jax.grad`.

## Useful entry points

- `mlp_jax.model.init_mlp_params`: creates the weights and biases for an MLP.
- `mlp_jax.model.mlp_forward`: computes logits for a batch of features.
- `mlp_jax.training.train_mlp`: trains parameters using full-batch SGD.
- `mlp_jax.model.predict_classes`: returns the highest-logit class for each
  sample.
