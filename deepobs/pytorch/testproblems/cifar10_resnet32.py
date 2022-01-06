"""Testproblem for ResNet32 on CIFAR-10."""

from torch.nn import CrossEntropyLoss

from ..datasets.cifar10 import cifar10
from .testproblem import UnregularizedTestproblem
from .testproblems_modules_resnets import resnet32


class cifar10_resnet32(UnregularizedTestproblem):
    """DeepOBS problem for ResNet32 on CIFAR-10.

    NOTE: Since the ResNet32 architecture uses batch normalizuation layers, the
    model has to be set to evaluation mode (``model.eval()``) when using
    BackPACK.
    """

    def set_up(self):
        """Set up the testproblem, i.e. (data, loss_function and network)"""
        self.data = cifar10(self._batch_size)
        self.loss_function = CrossEntropyLoss
        self.net = resnet32()
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
