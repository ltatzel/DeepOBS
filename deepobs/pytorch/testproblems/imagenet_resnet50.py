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


if __name__ == "__main__":

    print("\n===== START =====")

    tp = imagenet_resnet50()
    tp.set_up()

    print("===== DONE =====\n")
