"""Test the CIFAR-10 testproblem with the resnet32 by checking that BackPACK's
extensions and ViViT works with this architecture. Also check compatibility with
DeepOBS by running the StandardRunner.
"""

import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad, DiagGGNExact
from torch.optim import SGD

from deepobs import pytorch as pt
from deepobs.pytorch.testproblems.cifar10_resnet32 import cifar10_resnet32

# Set up testproblem
testproblem = cifar10_resnet32(batch_size=128)
torch.manual_seed(0)
testproblem.set_up()

# Extract model
model = testproblem.net
# print("model = \n", model)
model = model.eval()  # batch normalization layers not supproted in train mode

# Data
# train_loader, _ = testproblem.data._make_train_and_valid_dataloader()
# batch_data = next(iter(train_loader))
# torch.save(batch_data, "batch_data.pt")
batch_data = torch.load("batch_data.pt")
inputs, labels = batch_data

# Lossfunc
lossfunc = testproblem.loss_function()


# ------------------------------------------------------------------------------
# Test forward pass
# ------------------------------------------------------------------------------
print("\n===== Test: Forward pass =====")

# Forward pass
outputs = model(inputs)
print("outputs.shape = ", outputs.shape)
print("loss = ", lossfunc(outputs, labels))


# ------------------------------------------------------------------------------
# Test BackPACK
# ------------------------------------------------------------------------------

# Extend model and loss function
lossfunc = extend(lossfunc)

print("\n===== Converting model =====")
model = extend(model, use_converter=True, debug=False)
print("Done")

# Test first-order extention
print("\n===== Test: First-order extension =====")
loss = lossfunc(model(inputs), labels)
with backpack(BatchGrad()):
    loss.backward()
print("Done")

# Test second-order extension
print("\n===== Test: Second-order extension =====")
loss = lossfunc(model(inputs), labels)
with backpack(DiagGGNExact()):
    loss.backward()
print("Done")


# ------------------------------------------------------------------------------
# Test ViViT
# ------------------------------------------------------------------------------
# TODO


# ------------------------------------------------------------------------------
# Test DeepOBS
# ------------------------------------------------------------------------------
optimizer_class = SGD
hyperparams = {
    "lr": {"type": float, "default": 0.1},
    "momentum": {"type": float, "default": 0.90},
    "nesterov": {"type": bool, "default": False},
}

runner = pt.runners.StandardRunner(optimizer_class, hyperparams)
# runner.run("cifar10_resnet32", batch_size=128, num_epochs=2)
print("Done")
