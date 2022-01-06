"""Implements the ResNets from 
https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py 
with BackPACK compatibility.
"""

from collections import OrderedDict
from typing import Callable, List, OrderedDict

import torch.nn as nn
import torch.nn.init as init
from backpack.custom_module.branching import Parallel
from backpack.custom_module.pad import Pad
from backpack.custom_module.slicing import Slicing

__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Sequential):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        layers = self.basic_block(in_planes, planes, stride=stride, option=option)
        super().__init__(layers)

    @staticmethod
    def basic_block(
        in_planes: int, planes: int, stride: int = 1, option: str = "A"
    ) -> OrderedDict[str, nn.Module]:
        expansion = 1

        layers = [
            (
                "conv1",
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
            ),
            ("bn1", nn.BatchNorm2d(planes)),
            ("relu1", nn.ReLU()),
            (
                "conv2",
                nn.Conv2d(
                    planes, planes, kernel_size=3, stride=1, padding=1, bias=False
                ),
            ),
            ("bn2", nn.BatchNorm2d(planes)),
        ]

        shortcut = nn.Identity()

        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                shortcut = nn.Sequential(
                    Slicing(
                        (
                            slice(None),
                            slice(None),
                            slice(None, None, 2),
                            slice(None, None, 2),
                        )
                    ),
                    Pad(
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        mode="constant",
                        value=0.0,
                    ),
                )
            elif option == "B":
                shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(expansion * planes),
                )

        block = Parallel(nn.Sequential(OrderedDict(layers)), shortcut)

        return OrderedDict([("block", block), ("relu2", nn.ReLU())])


class ResNet(nn.Sequential):
    def __init__(
        self,
        block: Callable[[int, int, int], nn.Module],
        num_blocks: List[int],
        num_classes: int = 10,
    ):
        self.in_planes = 16

        layers = OrderedDict(
            [
                (
                    "conv1",
                    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                ),
                ("bn1", nn.BatchNorm2d(16)),
                ("relu1", nn.ReLU()),
                ("layer1", self._make_layer(block, 16, num_blocks[0], stride=1)),
                ("layer2", self._make_layer(block, 32, num_blocks[1], stride=2)),
                ("layer3", self._make_layer(block, 64, num_blocks[2], stride=2)),
                (
                    "avgpool",
                    nn.AvgPool2d(kernel_size=8),
                ),  # works for cifar-10, resnet32
                ("flatten", nn.Flatten()),
                ("linear", nn.Linear(64, num_classes)),
            ]
        )

        super().__init__(layers)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np

    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(
                    lambda p: p.requires_grad and len(p.data.size()) > 1,
                    net.parameters(),
                )
            )
        ),
    )


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("resnet"):
            print(net_name)
            test(globals()[net_name]())
            print()