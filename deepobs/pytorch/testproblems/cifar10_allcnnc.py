# -*- coding: utf-8 -*-
"""The ALL-CNN-C architecture for CIFAR-10."""

from torch import nn

from ..datasets.cifar10 import cifar10
from .testproblem import WeightRegularizedTestproblem
from .testproblems_modules import net_cifar100_allcnnc_wo_dropout


class cifar10_allcnnc_wo_dropout(WeightRegularizedTestproblem):
    """Same as `cifar100_allcnnc_wo_dropout` but uses the CIFAR-10 dataset."""

    def __init__(self, batch_size, l2_reg=0.001):
        super(cifar10_allcnnc_wo_dropout, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the All CNN C test problem on Cifar-100."""
        self.data = cifar10(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_cifar100_allcnnc_wo_dropout(num_classes=10)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
