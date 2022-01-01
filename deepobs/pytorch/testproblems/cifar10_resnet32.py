"""This script implements the ``cifar10_resnet32`` testproblem that uses a 
resnet-architecture with 32 layers. The original paper that introduced resnets 
can be found at https://arxiv.org/abs/1512.03385

The code is copied from Yerlan Idelbayev's git repo "Proper ResNet Implemen-
tation for CIFAR10/CIFAR100 in Pytorch", see
https://github.com/akamaster/pytorch_resnet_cifar10 (accessed December 7th, 
2021). 

Changelog:
- Added some documentation (e.g. some docstrings)
- Replaced in-place operation in residual block 
- Replaced functions from ``torch.nn.functional`` by modules from ``torch.nn``
(for BackPACK compatibility). 

TODO
- Replace batch-normalization layers by group normalization, because per-sample
quantities don't make sense with batch-normalization.  
"""

import torch.nn as nn
import torch.nn.init as init

from ..datasets.cifar10 import cifar10
from .testproblem import UnregularizedTestproblem


def _weights_init(m):
    """Initialization for linear and conv layers"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    """The basic block consists of two convolutional blocks, each followed by
    a Batch-Norm layer and the ReLu activation function.

    For option ``A`` (which is used with CIFAR-10), the shortcut/skip-connection
    is the identity with extra zero padding. It basically adds the original
    input ``x`` to the input of the second ReLu activation.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()

        # First conv- and batch-normalization layer
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

        # Second conv- and batch-normalization layer
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        # The shortcut/skip-connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """For CIFAR10 ResNet paper uses this option A."""
                padding = (0, 0, 0, 0, planes // 4, planes // 4)
                self.shortcut = nn.ZeroPad2d(padding)
            elif option == "B":
                raise RuntimeError("Option B not supported.")

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if isinstance(self.shortcut, nn.Sequential):
            shortcut_input = x
        else:  # case A above
            shortcut_input = x[:, :, ::2, ::2]

        out = out + self.shortcut(shortcut_input)
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    """The residual network consists of ``BasicBlock``s."""

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        kernel_size = 8  # works for cifar-10, resnet32
        self.avgpool = nn.AvgPool2d(kernel_size)

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


class net_cifar10_resnet32(ResNet):
    """The resnet with 32 layers"""

    def __init__(self):
        super().__init__(BasicBlock, [5, 5, 5], num_classes=10)


class cifar10_resnet32(UnregularizedTestproblem):
    """DeepOBS testproblem (without regularization) for CIFAR-10 using a resnet
    with 32 layers.
    """

    def __init__(self, batch_size):
        super(cifar10_resnet32, self).__init__(batch_size)

    def set_up(self):
        """Set up the testproblem, i.e. (data, loss_function and network)"""
        self.data = cifar10(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_cifar10_resnet32()
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
