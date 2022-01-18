"""Testproblem for ResNet110 on CIFAR-10."""

from torch.nn import Module

from .cifar10_resnet_base import cifar10_resnet
from .testproblems_modules_resnets import resnet110


class cifar10_resnet110(cifar10_resnet):
    """DeepOBS problem for ResNet110 on CIFAR-10.

    NOTE: Since the ResNet110 architecture uses batch normalizuation layers, the
    model has to be set to evaluation mode (``model.eval()``) when using
    BackPACK.
    """

    @staticmethod
    def net_fn() -> Module:
        return resnet110()
