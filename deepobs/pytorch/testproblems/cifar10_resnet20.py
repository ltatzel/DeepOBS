"""Testproblem for ResNet20 on CIFAR-10."""

from .cifar10_resnet_base import cifar10_resnet
from .testproblems_modules_resnets import resnet20


class cifar10_resnet20(cifar10_resnet):
    """DeepOBS problem for ResNet20 on CIFAR-10.

    NOTE: Since the ResNet20 architecture uses batch normalizuation layers, the
    model has to be set to evaluation mode (``model.eval()``) when using
    BackPACK.
    """

    net_fn = resnet20
