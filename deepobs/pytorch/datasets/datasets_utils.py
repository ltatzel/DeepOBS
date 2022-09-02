# -*- coding: utf-8 -*-

"""Helper classes and functions for the DeepOBS datasets."""

from random import shuffle

import torch.utils.data.sampler as s


class train_eval_sampler(s.Sampler):
    """A subclass of torch Sampler to easily draw the train eval set."""

    def __init__(self, indices, sub_size):
        """
        Args:
            indices (list): A list of indices (i.e. a list of integers). 
                `__iter__` creates and returns a random subset of these indices.
            sub_size (int): The size of the subset which is to be drawn from
                the original index set `indices`.
        """
        self.indices = indices.copy()  # copy because of in-place shuffle
        self.sub_size = sub_size

    def __iter__(self):
        """Create a random subset of `self.indices`."""
        shuffle(self.indices)
        sub_indices = self.indices[0 : self.sub_size]
        return iter(sub_indices)

    def __len__(self):
        return self.sub_size


# Example
# if __name__ == "__main__":

#     # Create indices
#     indices = list(range(10))
#     print("indices = ", indices)

#     example_sampler = train_eval_sampler(indices, 3)
#     print("sub_indices = ", list(example_sampler.__iter__()))
#     print("sub_indices = ", list(example_sampler.__iter__()))
#     print("sub_indices = ", list(example_sampler.__iter__()))
