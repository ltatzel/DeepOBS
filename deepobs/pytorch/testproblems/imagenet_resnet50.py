"""Implementation of the DeepOBS test problem `imagenet_resnet50`.

REQUIREMENTS: 
This implementation is based on torch 1.12.1 and torchvision 0.13.1 and might 
not work with other versions. The function `check_torch_torchvision_versions`
checks wether the correct versions are used. 

NOTE: 
The test problem is entirely extracted from
https://github.com/pytorch/vision/tree/release/0.13/references/classification.
The most important functionality is implemented in `train.py`. Some auxiliary
functions and classes were taken from `sampler.py`, `presets.py`,
`transforms.py`, and `utils.py`.

LOG: 
- The training routine is called via `torchrun --nproc_per_node=8 train.py
  --model "resnet50"`. So, it uses the default parameters that can be extracted
  from the `get_args_parser` function in `train.py`. Note that the "effective"
  batch size depends on the number of GPUs - in our case, it is `8 * 32 =
  256`.
- Data: Loading the data is implemented in `train.py` in the `load_data`
  function. The data sets have to be downloaded manually - we use the data that
  is already on Slurm (see `TRAINSET_PATH` and `VALSET_PATH`). The ImageNet data
  set is available at `/mnt/qb/datasets/ImageNet2012`. However, it is also
  available locally (this makes IO faster) on a subset of the compute nodes
  accessible via the `sbatch`-option `constraint=ImageNet2012`.
- Model: The model is simply taken from `torchvision`. By default, the
  pre-trained model is used.
- Loss-function: The loss function is defined as cross entropy loss (see line
  230 in `train.py`).
- Regularizer: The regularizer is defined in `train.py` in lines 232-243. This
  code calls `utils.set_weight_decay` with all parameters set to `None` except
  `model` and `weight_decay`. We simply copy this function. The default weight
  decay is set to `1e-4`.

VALIDATION: 
At https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50,
a test set accuracy of 76.13 % is reported. We get the exact same test accuracy
(for seed `0`), which is an indication of the correctness of our implementation.
"""  # noqa: E501

import errno
import math
import os
from types import SimpleNamespace
from warnings import warn

import torch
import torch.distributed as dist
import torchvision
from torch import nn
from torchvision.models import resnet50
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from deepobs.pytorch.datasets.dataset import DataSet
from deepobs.pytorch.testproblems.testproblem import TestProblem


# ==============================================================================
# Auxiliary functions and classes
# (required by the `load_data` method below)
# ==============================================================================


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for
    distributed, with repeated augmentation. It ensures that different each
    augmented version of a sample will be visible to a different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'. This is borrowed
    from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        repetitions=3,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available!"
                )
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available!"
                )
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(
                len(self.dataset) * float(repetitions) / self.num_replicas
            )
        )
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = [
            transforms.RandomResizedCrop(crop_size, interpolation=interpolation)
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(
                    autoaugment.RandAugment(interpolation=interpolation)
                )
            elif auto_augment_policy == "ta_wide":
                trans.append(
                    autoaugment.TrivialAugmentWide(interpolation=interpolation)
                )
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(
                    autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=interpolation
                    )
                )
        trans.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


class EmptyDataset(torch.utils.data.IterableDataset):
    """An empty Pytorch dataset"""
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0
    
    def __iter__(self):
        return iter([])


def check_torch_torchvision_versions():
    """Check if torch and torchvision versions are correct."""

    # torch
    torch_vers = torch.__version__
    if not torch_vers == "1.12.1":
        warn(f"Expected torch version 1.12.1 but found {torch_vers}.")

    # torchvision
    vision_vers = torchvision.__version__
    if not vision_vers == "0.13.1":
        warn(f"Expected torchvision version 0.13.1 but found {vision_vers}.")


# ==============================================================================
# ImageNet data set
# ==============================================================================

# Paths to ImageNet dataset on Slurm. Use compute notes, where data sets are
# available locally on the compute node (for faster I/O) by using `sbatch` with
# `--constraint=ImageNet2012`.
TRAINSET_PATH = r"/mnt/lustre/datasets/ImageNet2012/train"
VALSET_PATH = r"/mnt/lustre/datasets/ImageNet2012/val"

# Some defualt parameters extracted from `get_args_parser` in `train.py`
DEFAULT_ARGS = {
    "val_resize_size": 256,  # resize images
    "val_crop_size": 224,  # crop images, use only center
    "train_crop_size": 224,  # crop images, use only center
    "interpolation": "bilinear",  # resize images using bilinear interpolation
    "cache_dataset": False,  # cache the datasets
    "weights": "IMAGENET1K_V1",  # weight for pre-trained model
    "test_only": False,  # only test the model
    "distributed": False,  # so far, distributed mode is not supported
}
DEFAULT_ARGS = SimpleNamespace(**DEFAULT_ARGS)


class imagenet_data(DataSet):
    """DeepOBS data set class for the `ImageNet` data set"""

    def __init__(self, batch_size, train_eval_size=None):
        """Create a new data set instance. `train_eval_size` is the size used
        for the training evaluation set and for the validation set. If `None`,
        `train_eval_size` is set to the size of the test set.
        """

        self._name = "imagenet"

        # Check torch and torchvision version
        check_torch_torchvision_versions()

        # Create data sets and samplers
        train_data, test_data, train_sampler, test_sampler = self.load_data(
            TRAINSET_PATH, VALSET_PATH, DEFAULT_ARGS
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
    def load_data(traindir, valdir, args):
        """Set up the data sets and data samplers. This is a copy of
        https://github.com/pytorch/vision/blob/bddbd7e6d65ecacc2e40cf6c9e2059669b8dbd44/references/classification/train.py#L113-L179.
        """  # noqa: E501

        # Extract parameters
        val_resize_size = args.val_resize_size
        val_crop_size = args.val_crop_size
        train_crop_size = args.train_crop_size
        interpolation = InterpolationMode(args.interpolation)

        # Load training data
        cache_path = _get_cache_path(traindir)
        if args.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print(f"Loading dataset_train from {cache_path}")
            dataset, _ = torch.load(cache_path)
        else:
            auto_augment_policy = getattr(args, "auto_augment", None)
            random_erase_prob = getattr(args, "random_erase", 0.0)
            dataset = torchvision.datasets.ImageFolder(
                traindir,
                ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                ),
            )
            if args.cache_dataset:
                print(f"Saving dataset_train to {cache_path}")
                mkdir(os.path.dirname(cache_path))
                save_on_master((dataset, traindir), cache_path)

        # Load validation data (used as test data)
        cache_path = _get_cache_path(valdir)
        if args.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print(f"Loading dataset_test from {cache_path}")
            dataset_test, _ = torch.load(cache_path)
        else:
            if args.weights and args.test_only:
                weights = torchvision.models.get_weight(args.weights)
                preprocessing = weights.transforms()
            else:
                preprocessing = ClassificationPresetEval(
                    crop_size=val_crop_size,
                    resize_size=val_resize_size,
                    interpolation=interpolation,
                )

            dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                preprocessing,
            )
            if args.cache_dataset:
                print(f"Saving dataset_test to {cache_path}")
                mkdir(os.path.dirname(cache_path))
                save_on_master((dataset_test, valdir), cache_path)

        # Create data samplers
        assert not args.distributed, "Distributed setting not supported"
        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = RASampler(
                    dataset, shuffle=True, repetitions=args.ra_reps
                )
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset
                )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test, shuffle=False
            )
        else:
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
# ImageNet-ResNet50 test problem
# ==============================================================================


class imagenet_resnet50(TestProblem):
    """DeepOBS test problem class for the ResNet50 network on ImageNet data."""

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
        # Set up data loaders
        self.data = imagenet_data(self._batch_size)

        # Define loss function
        self.loss_function = nn.CrossEntropyLoss

        # Define model
        weights = "IMAGENET1K_V1" if self.pretrained else None
        self.net = resnet50(weights=weights)
        self.net.to(self._device)

        # Define parameter groups for regularization
        self.regularization_groups = self.get_regularization_groups()
