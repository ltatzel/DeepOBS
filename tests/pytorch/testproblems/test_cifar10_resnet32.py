"""Test the CIFAR-10 testproblem with the resnet32 by checking that BackPACK's
extensions and ViViT works with this architecture. Also check compatibility with
DeepOBS by running the StandardRunner.
"""

import torch
from backpack import extend
from torch.optim import SGD

from deepobs import pytorch as pt
from deepobs.pytorch.testproblems.cifar10_resnet32 import (
    cifar10_resnet32,
    net_cifar10_resnet32,
)

# from backpack.extensions import BatchGrad, DiagGGNExact


# ------------------------------------------------------------------------------
# Print some information about the resnet and check if it works with BackPACK
# ------------------------------------------------------------------------------

model = net_cifar10_resnet32()
print("model = \n", model)

# Get some CIFAR-10 data
testproblem = cifar10_resnet32(batch_size=128)
testproblem.set_up()
train_loader, _ = testproblem.data._make_train_and_valid_dataloader()
inputs, labels = next(iter(train_loader))
print("inputs.shape = ", inputs.shape)
print("labels.shape = ", labels.shape)

# Put the model in evaluation mode (because batch normalization layers that are
# not supported in train mode)
model = model.eval()

# Extend model and loss function
lossfunc = extend(torch.nn.CrossEntropyLoss())
# model = extend(model, use_converter=True)

# Test first-order extention
# print("\nTest first-order extension")
# loss = lossfunc(model(inputs), labels)
# with backpack(BatchGrad()):
#    loss.backward()

# Test second-order extension
# print("\nTest second-order extension")
# loss = lossfunc(model(inputs), labels)
# with backpack(DiagGGNExact()):
#     loss.backward()

# Test ViViT
# TODO


# ------------------------------------------------------------------------------
# Running an optimizer on the testproblem
# ------------------------------------------------------------------------------
optimizer_class = SGD
hyperparams = {
    "lr": {"type": float, "default": 0.1},
    "momentum": {"type": float, "default": 0.90},
    "nesterov": {"type": bool, "default": False},
}

runner = pt.runners.StandardRunner(optimizer_class, hyperparams)
runner.run("cifar10_resnet32", batch_size=128, num_epochs=2)
