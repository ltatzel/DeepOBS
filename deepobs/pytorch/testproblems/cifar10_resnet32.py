"""Testproblem for ResNet32 on CIFAR-10."""

from .cifar10_resnet_base import cifar10_resnet
from .testproblems_modules_resnets import resnet32


class cifar10_resnet32(cifar10_resnet):
    """DeepOBS problem for ResNet32 on CIFAR-10.

    NOTE: Since the ResNet32 architecture uses batch normalizuation layers, the
    model has to be set to evaluation mode (``model.eval()``) when using
    BackPACK.
    """

    net_fn = resnet32
