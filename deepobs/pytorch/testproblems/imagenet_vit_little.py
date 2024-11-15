"""Implementation of the DeepOBS test problem `imagenet_vit_little`."""

import timm

import torch
import torchvision
from torch import nn
from timm.data.transforms_factory import transforms_imagenet_eval
from deepobs.pytorch.datasets.dataset import DataSet
from deepobs.pytorch.testproblems.testproblem import TestProblem


# ==============================================================================
# Auxiliary functions and classes
# (required by the `load_data` method below)
# ==============================================================================
class EmptyDataset(torch.utils.data.IterableDataset):
    """An empty Pytorch dataset"""

    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __iter__(self):
        return iter([])


# ==============================================================================
# ImageNet data set
# ==============================================================================

# Paths to ImageNet dataset on Slurm. Use compute notes, where data sets are
# available locally on the compute node (for faster I/O) by using `sbatch` with
# `--constraint=ImageNet2012`.
TRAINSET_PATH = r"/mnt/lustre/datasets/ImageNet2012/train"
VALSET_PATH = r"/mnt/lustre/datasets/ImageNet2012/val"


class imagenet_data(DataSet):
    """DeepOBS data set class for the `ImageNet` data set"""

    def __init__(self, batch_size, train_eval_size=None):
        """Create a new data set instance. `train_eval_size` is the size used
        for the training evaluation set and for the validation set. If `None`,
        `train_eval_size` is set to the size of the test set.
        """

        self._name = "imagenet"

        # Create data sets and samplers
        train_data, test_data, train_sampler, test_sampler = self.load_data(
            TRAINSET_PATH, VALSET_PATH
        )
        self._train_data = train_data
        self._test_data = test_data
        self._train_sampler = train_sampler
        self._test_sampler = test_sampler

        # Determine size of train eval data
        self._train_eval_size = train_eval_size
        if train_eval_size is None:  # same as test set size
            self._train_eval_size = len(self._test_data)

        super().__init__(batch_size)

    @staticmethod
    def load_data(traindir, valdir):
        """Set up the data sets and data samplers. This is a copy of
        https://github.com/pytorch/vision/blob/bddbd7e6d65ecacc2e40cf6c9e2059669b8dbd44/references/classification/train.py#L113-L179.
        """  # noqa: E501
        # Load training data (used as test data)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms_imagenet_eval(
                img_size=256,
                crop_pct=0.95,
                interpolation="bilinear",
                use_prefetcher=False,
            ),
        )

        # Load validation data (used as test data)
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms_imagenet_eval(
                img_size=256,
                crop_pct=0.95,
                interpolation="bilinear",
                use_prefetcher=False,
            ),
        )

        # Create data samplers
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        return dataset, dataset_test, train_sampler, test_sampler

    def _make_train_and_valid_dataloader(self):
        """Create the training and validation data loader."""

        train_loader = self._make_dataloader(
            self._train_data, sampler=self._train_sampler, drop_last=True
        )
        self._train_indices = list(range(len(self._train_data)))

        # No validation data
        valid_loader = torch.utils.data.DataLoader(EmptyDataset())
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        """Create the test data loader."""

        test_loader = self._make_dataloader(
            self._test_data, sampler=self._test_sampler, shuffle=False
        )
        return test_loader


# ==============================================================================
# ImageNet-ViT-Little test problem
# ==============================================================================


class imagenet_vit_little(TestProblem):
    """DeepOBS test problem class for the ViT-Little network on ImageNet data."""

    def __init__(self, batch_size=256, l2_reg=1e-4, pretrained=True):
        """Create a new problem instance.

        NOTE: Note that the default batch size is set to `256`, because the
        original training script uses `--nproc_per_node=8` GPUs with a batch
        size of `32`. The gradients are averaged over all GPUs, i.e. the
        effective batch size is `8 * 32 = 256`, see
        https://github.com/pytorch/vision/blob/bddbd7e6d65ecacc2e40cf6c9e2059669b8dbd44/references/classification/train.py#L378.
        """  # noqa: E501
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
        """Define weight decay constants for the parameters. Return a list of
        dicts. Each dict contains two keys: `"weight_decay"` (the value
        specifies the L2 regularization constant) and `"params"` (which
        represents the corresponding parameters).

        This is a copy of
        https://github.com/pytorch/vision/blob/bddbd7e6d65ecacc2e40cf6c9e2059669b8dbd44/references/classification/utils.py#L406-L465.

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
                child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
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
        # Set up data loaders
        self.data = imagenet_data(self._batch_size)

        # Define loss function
        self.loss_function = nn.CrossEntropyLoss

        # Define model
        self.net = timm.create_model(
            "vit_little_patch16_reg4_gap_256.sbb_in1k", pretrained=True
        )
        self.net.to(self._device)

        # Define parameter groups for regularization
        self.regularization_groups = self.get_regularization_groups()
