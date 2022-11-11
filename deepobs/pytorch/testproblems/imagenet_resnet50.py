"""Implementation of the DeepOBS test problem `imagenet_resnet50_pretrained`.

This is extracted from (note the version `0.13`):
https://github.com/pytorch/vision/tree/release/0.13/references/classification
"""

import torch
from torchvision.models import resnet50, ResNet50_Weights
from deepobs.pytorch.testproblems.testproblem import TestProblem



class imagenet_resnet50_pretrained(TestProblem):
    """DeepOBS test problem class for the ResNet50 network on ImageNet.
    
    TODO
    """

    def __init__(self, batch_size, l2_reg=1e-4):
        """Create a new problem instance.

        TODO
        """
        super().__init__(batch_size, l2_reg)

    def get_regularization_groups(self):
        """TODO: So far, no regularization"""
        no = 0.0
        group_dict = {no: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize no parameters
            group_dict[no].append(parameters)
        return group_dict

    def set_up(self):
        """Set up the test problem."""
        # self.data = imagenet(self._batch_size)
        

        self.loss_function = nn.CrossEntropyLoss

        # Pre-trained model 
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.net.to(self._device)

        # Define parameter groups for regularization
        self.regularization_groups = self.get_regularization_groups()


if __name__ == "__main__":
    
    print("\n===== START =====")
    
    tp = imagenet_resnet50_pretrained(batch_size=32)
    tp.set_up()
    
    print("\n===== DONE =====")
