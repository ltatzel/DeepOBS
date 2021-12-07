from deepobs.pytorch.testproblems.cifar10_resnet32 import net_cifar10_resnet32

"""Example run script using StandardRunner."""

from torch.optim import SGD

from deepobs import pytorch as pt

optimizer_class = SGD
hyperparams = {
    "lr": {"type": float},
    "momentum": {"type": float, "default": 0.99},
    "nesterov": {"type": bool, "default": False},
}

runner = pt.runners.StandardRunner(optimizer_class, hyperparams)
runner.run("cifar10_resnet32", batch_size=128, num_epochs=2)
