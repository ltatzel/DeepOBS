"""This script runs SGD on the noise-free quadratic problem."""

import torch
from deepobs import pytorch as pt

optimizer_class = torch.optim.SGD
hyperparams = {
    "lr": {"type": float},
    "momentum": {"type": float, "default": 0.99},
    "nesterov": {"type": bool, "default": False},
}
runner = pt.runners.StandardRunner(optimizer_class, hyperparams)

runner.run(
    testproblem="noise_free_quadratic",
    hyperparams={"lr": 1e-3},
    batch_size=8,
    num_epochs=10,
)
