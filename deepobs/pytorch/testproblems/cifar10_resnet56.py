"""Testproblem for ResNet56 on CIFAR-10."""

from .cifar10_resnet_base import cifar10_resnet
from .testproblems_modules_resnets import resnet56


class cifar10_resnet56(cifar10_resnet):
    """DeepOBS problem for ResNet56 on CIFAR-10.

    NOTE: Since the ResNet56 architecture uses batch normalizuation layers, the
    model has to be set to evaluation mode (``model.eval()``) when using
    BackPACK.
    """

    net_fn = resnet56
