"""AFHQ DeepOBS data set for PyTorch."""

import os

from torchvision import datasets, transforms

from deepobs.config import get_data_dir
from deepobs.datasets.pytorch._dataset import DataSet


class AFHQ(DataSet):
    """DeepOBS data set class for the `AFHQ\
    <https://github.com/clovaai/stargan-v2/blob/master/README.md\
    #animal-faces-hq-dataset-afhq>`_ data set.

    Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size the remainder is dropped in each
            epoch (after shuffling).
        image_resize (bool): If ``True`` images are resized to 64x64 for the
            training data (but not the test data).
        train_eval_size (int): Size of the train eval data set.
            Defaults to ``19,962`` the size of the test set.
    """

    def __init__(self, batch_size, image_resize=False, train_eval_size=1500):
        """Creates a new CelebA instance.

        Args:
            batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
                is not a divider of the dataset size the remainder is dropped in each
                epoch (after shuffling).
            image_resize (bool): If ``True`` images are resized to 64x64 for the
                training data (but not the test data).
            train_eval_size (int): Size of the train eval data set.
                Defaults to ``19,962`` the size of the test set.
        """
        self._image_resize = image_resize
        self._transform_no_resize = transforms.Compose([transforms.ToTensor()])

        self._transform_resize = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        super().__init__(batch_size, train_eval_size)

    def _make_train_and_valid_dataloader(self):
        if self._image_resize:
            transform = self._transform_resize
        else:
            transform = self._transform_no_resize

        train_dataset = datasets.ImageFolder(
            root=os.path.join(get_data_dir(), self._name, "train"),
            transform=transform,
        )
        valid_dataset = datasets.ImageFolder(
            root=os.path.join(get_data_dir(), self._name, "train"),
            transform=self._transform_no_resize,
        )
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(
            train_dataset, valid_dataset, shuffle_dataset=True
        )
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        transform = self._transform_no_resize
        test_dataset = datasets.ImageFolder(
            root=os.path.join(get_data_dir(), self._name, "val"),
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None)
