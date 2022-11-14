"""Implementation of the DeepOBS test problem `imagenet_resnet50`.

NOTE: This implementation is based on torch 1.12.1 and torchvision 0.13.1 and
might not work with other versions. It is extracted from:
https://github.com/pytorch/vision/tree/release/0.13/references/classification
"""

from copy import deepcopy

from torch import allclose, nn
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

    def get_regularization_groups(self):
        """TODO: So far, no regularization. This needs to be fixed!!!"""
        no = 0.0
        group_dict = {no: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize no parameters
            group_dict[no].append(parameters)
        return group_dict

    def set_up(self):
        """Set up the test problem."""
        # self.data = imagenet(self._batch_size)

        # Define loss function
        self.loss_function = nn.CrossEntropyLoss

        # Define model
        weights = "IMAGENET1K_V1" if self.pretrained else None
        self.net = resnet50(weights=weights)
        self.net.to(self._device)

        # Define parameter groups for regularization (TODO: Needs fixing!!)
        self.regularization_groups = self.get_regularization_groups()


def set_weight_decay_ref(
    model,
    weight_decay,
    norm_weight_decay=None,
    norm_classes=None,
    custom_keys_weight_decay=None,
):

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
                    f"{prefix}.{name}" if prefix != "" and "." in key else name
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


def set_weight_decay(
    model,
    weight_decay,
):

    params = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        params.append(p)

    return [
        {
            "params": params,
            "weight_decay": weight_decay,
        }
    ]


if __name__ == "__main__":

    print("\nSet up testproblem")

    tp = imagenet_resnet50()
    tp.set_up()

    # Test regularization
    model_ref = resnet50(weights="IMAGENET1K_V1")
    model = deepcopy(model_ref)
    weight_decay = 1e-2

    print("\nCall set_weight_decay_ref")
    param_groups_ref = set_weight_decay_ref(model_ref, weight_decay)

    print("Call set_weight_decay")
    param_groups = set_weight_decay(model, weight_decay)

    # Compare
    print("Compare results")
    assert len(param_groups) == len(param_groups_ref) == 1
    params = param_groups[0]
    params_ref = param_groups_ref[0]

    assert len(params) == len(params_ref) == 2
    assert params["weight_decay"] == params_ref["weight_decay"]
    assert len(params["params"]) == len(params_ref["params"]) > 0
    for p, p_ref in zip(params["params"], params_ref["params"]):
        assert allclose(p, p_ref)
