"""Implementation of the DeepOBS test problem `imagenet_resnet50`.

NOTE: This implementation is based on torch 1.12.1 and torchvision 0.13.1 and
might not work with other versions. It is extracted from:
https://github.com/pytorch/vision/tree/release/0.13/references/classification
"""

from torch import nn
from torchvision.models import resnet50

from deepobs.pytorch.testproblems.testproblem import TestProblem


class imagenet_resnet50(TestProblem):
    """DeepOBS test problem class for the ResNet50 network on ImageNet.

    TODO
    """

    def __init__(self, batch_size=32, l2_reg=1e-4, pretrained=True):
        """Create a new problem instance.

        TODO
        """
        super().__init__(batch_size, l2_reg)
        self.pretrained = pretrained

    @staticmethod
    def _set_weight_decay(
        model,
        weight_decay,
        norm_weight_decay=None,
        norm_classes=None,
        custom_keys_weight_decay=None,
    ):
        """Create a list of dicts. Each dict contains two keys: `"weight_decay"`
        (the value specifies the L2 regularization constant) and `"params"`
        (which represents the corresponding parameters).

        This is a copy of https://github.com/pytorch/vision/blob/bddbd7e6d65ecacc2e40cf6c9e2059669b8dbd44/references/classification/utils.py#L406-L465.

        If `norm_weight_decay`, `norm_classes` and `custom_keys_weight_decay`
        are all `None`, this function simplifies to:
        ```
        params = []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            params.append(p)
        return [{"params": params, "weight_decay": weight_decay}]
        ```
        That means, all trainable parameters are regularized with the same
        regularization constant in this case.
        """  # noqa: E501

        if not norm_classes:
            norm_classes = [
                nn.modules.batchnorm._BatchNorm,
                nn.LayerNorm,
                nn.GroupNorm,
                nn.modules.instancenorm._InstanceNorm,
                nn.LocalResponseNorm,
            ]
        norm_classes = tuple(norm_classes)

        params = {
            "other": [],
            "norm": [],
        }
        params_weight_decay = {
            "other": weight_decay,
            "norm": norm_weight_decay,
        }
        custom_keys = []
        if custom_keys_weight_decay is not None:
            for key, weight_decay in custom_keys_weight_decay:
                params[key] = []
                params_weight_decay[key] = weight_decay
                custom_keys.append(key)

        def _add_params(module, prefix=""):
            for name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                is_custom_key = False
                for key in custom_keys:
                    target_name = (
                        f"{prefix}.{name}"
                        if prefix != "" and "." in key
                        else name
                    )
                    if key == target_name:
                        params[key].append(p)
                        is_custom_key = True
                        break
                if not is_custom_key:
                    if norm_weight_decay is not None and isinstance(
                        module, norm_classes
                    ):
                        params["norm"].append(p)
                    else:
                        params["other"].append(p)

            for child_name, child_module in module.named_children():
                child_prefix = (
                    f"{prefix}.{child_name}" if prefix != "" else child_name
                )
                _add_params(child_module, prefix=child_prefix)

        _add_params(model)

        param_groups = []
        for key in params:
            if len(params[key]) > 0:
                param_groups.append(
                    {
                        "params": params[key],
                        "weight_decay": params_weight_decay[key],
                    }
                )
        return param_groups

    def get_regularization_groups(self):
        """Add constant regularization `self._l2_reg` to all trainable
        parameters. For details, see the `_set_weight_decay` method.
        """

        param_groups = self._set_weight_decay(
            self.net,
            self._l2_reg,
            norm_weight_decay=None,
            norm_classes=None,
            custom_keys_weight_decay=None,
        )
        assert len(param_groups) == 1
        assert len(param_groups[0]) == 2  # two keys: "weight_decay", "params"

        # Re-organize `param_groups` for compatibility with DeepOBS
        weight_decay = param_groups[0]["weight_decay"]
        assert weight_decay == self._l2_reg
        return {weight_decay: param_groups[0]["params"]}

    def set_up(self):
        """Set up the test problem."""
        # self.data = imagenet(self._batch_size)

        # Define loss function
        self.loss_function = nn.CrossEntropyLoss

        # Define model
        weights = "IMAGENET1K_V1" if self.pretrained else None
        self.net = resnet50(weights=weights)
        self.net.to(self._device)

        # Define parameter groups for regularization
        self.regularization_groups = self.get_regularization_groups()


if __name__ == "__main__":

    print("\nSet up testproblem")

    tp = imagenet_resnet50()
    tp.set_up()

    print("Regularization loss = ", tp.get_regularization_loss())
