"""Base class for ResNet testproblems on CIFAR-10.

Used to implement the testproblems

- ``cifar10_resnet20``
- ``cifar10_resnet32``
- ``cifar10_resnet4``
- ``cifar10_resnet56``
- ``cifar10_resnet110``
- ``cifar10_resnet1202``
"""

from typing import Callable

from torch.nn import CrossEntropyLoss, Module

from ..datasets.cifar10 import cifar10
from .testproblem import UnregularizedTestproblem


class cifar10_resnet(UnregularizedTestproblem):
    """Base class for DeepOBS ResNet problems on CIFAR-10.

    NOTE: Since the ResNet architectures have batch normalizuation layers, the
    model has to be set to evaluation mode (``model.eval()``) when using
    BackPACK.
    """

    def set_up(self):
        """Set up the testproblem, i.e. (data, loss_function and network)"""
        self.data = cifar10(self._batch_size)
        self.loss_function = CrossEntropyLoss
        self.net = self.net_fn()
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

    @staticmethod
    def net_fn() -> Module:
        raise NotImplementedError("Must be implemented by child class.")
