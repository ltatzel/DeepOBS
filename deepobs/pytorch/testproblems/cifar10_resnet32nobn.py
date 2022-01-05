"""Testproblem for ResNet32 with Identity instead of BN on CIFAR-10."""

from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    CrossEntropyLoss,
    Identity,
    Module,
)

from ..datasets.cifar10 import cifar10
from .testproblem import UnregularizedTestproblem
from .testproblems_modules_resnets import resnet32


def net_cifar10_resnet32nobn() -> Module:
    """ResNet32 with BatchNormalization layers replaced by Identities."""
    model = resnet32()

    def is_batch_norm(module: Module) -> bool:
        """Return True for BN layers."""
        return isinstance(module, (BatchNorm1d, BatchNorm2d, BatchNorm3d))

    def make_identity(module: Module) -> Identity:
        """Instantiate an identity layer."""
        return Identity()

    replace(model, is_batch_norm, make_identity)

    return model


def replace(module: Module, trigger, make_new):
    """Replace layer when trigger (m) is True, replace with make_new(m)."""

    def has_children(module):
        return bool(list(module.children()))

    for name, mod in module.named_children():
        if has_children(mod):
            replace(mod, trigger, make_new)
        else:
            if trigger(mod):
                new_mod = make_new(mod)
                setattr(module, name, new_mod)


class cifar10_resnet32nobn(UnregularizedTestproblem):
    """DeepOBS problem for ResNet32 (Identities instead of BN) on CIFAR-10."""

    def set_up(self):
        """Set up the testproblem, i.e. (data, loss_function and network)"""
        self.data = cifar10(self._batch_size)
        self.loss_function = CrossEntropyLoss
        self.net = net_cifar10_resnet32nobn()
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
