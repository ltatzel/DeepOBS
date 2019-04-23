# -*- coding: utf-8 -*-
"""CIFAR-10 DeepOBS dataset."""

import os
import numpy as np
from . import dataset
from .. import config
from torch.utils import data as dat
from torchvision import datasets
from torchvision import transforms


class cifar10(dataset.DataSet):
    """DeepOBS data set class for the `CIFAR-10\
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_ data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``50 000`` for train, ``10 000``
        for test) the remainder is dropped in each epoch (after shuffling).
    data_augmentation (bool): If ``True`` some data augmentation operations
        (random crop window, horizontal flipping, lighting augmentation) are
        applied to the training data (but not the test data).
    train_eval_size (int): Size of the train eval data set.
        Defaults to ``10 000`` the size of the test set.

  Attributes:
    batch: A tuple ``(x, y)`` of tensors, yielding batches of CIFAR-10 images
        (``x`` with shape ``(batch_size, 32, 32, 3)``) and corresponding one-hot
        label vectors (``y`` with shape ``(batch_size, 10)``). Executing these
        tensors raises a ``tf.errors.OutOfRangeError`` after one epoch.
    train_init_op: A tensorflow operation initializing the dataset for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the testproblem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the testproblem for
        evaluating on test data.
    phase: A string-value tf.Variable that is set to ``train``, ``train_eval``
        or ``test``, depending on the current phase. This can be used by testproblems
        to adapt their behavior to this phase.
  """

    def __init__(self,
                 batch_size,
                 data_augmentation=True,
                 train_eval_size=10000):
        """Creates a new CIFAR-10 instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (``50 000`` for train, ``10 000``
          for test) the remainder is dropped in each epoch (after shuffling).
      data_augmentation (bool): If ``True`` some data augmentation operations
          (random crop window, horizontal flipping, lighting augmentation) are
          applied to the training data (but not the test data).
      train_eval_size (int): Size of the train eval data set.
          Defaults to ``10 000`` the size of the test set.
    """
        self._name = "cifar10"
        self._data_augmentation = data_augmentation
        self._train_eval_size = train_eval_size
        super(cifar10, self).__init__(batch_size)

    def _make_dataloader(self, train = True, shuffle = True, data_augmentation = False, sampler = None):
        if data_augmentation:
            transform = transforms.Compose([
                    transforms.Pad(padding=2),
                    transforms.RandomCrop(size=(32,32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=63. / 255.,saturation=[0.5,1.5], contrast=[0.2,1.8]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513, 0.26158784))
                    ])
        else:
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513, 0.26158784))
                    ])
        dataset = datasets.CIFAR10(root = config.get_data_dir(), train = train, download = True, transform = transform)
        loader = dat.DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle, drop_last=True, pin_memory=True, num_workers=4, sampler=sampler)
        return loader

    def _make_train_dataloader(self):
        return self._make_dataloader(train=True, shuffle = True, data_augmentation = self._data_augmentation, sampler=None)

    def _make_test_dataloader(self):
        return self._make_dataloader(train=False, shuffle = False, data_augmentation = False, sampler=None)

    def _make_train_eval_dataloader(self):
        indices = np.random.choice(len(self._train_dataloader.dataset), size= self._train_eval_size, replace=False)
        sampler = dat.SubsetRandomSampler(indices)
        return self._make_dataloader(train=True, shuffle=False, data_augmentation=False, sampler=sampler)