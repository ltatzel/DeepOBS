"""Test the CIFAR-10 problem with the ResNet18 (BN replaced by Identity).

Do so by checking that BackPACK's extensions and ViViT works with this
architecture. Also check compatibility with DeepOBS by running the
StandardRunner.
"""

from copy import deepcopy

import torch
from backpack import backpack, extend
from backpack.core.derivatives.convnd import weight_jac_t_save_memory
from backpack.extensions import BatchGrad, DiagGGNExact
from deepobs import pytorch as pt
from deepobs.pytorch.testproblems.cifar10_resnet18nobn import cifar10_resnet18nobn
from torch.optim import SGD

# Set up testproblem
testproblem = cifar10_resnet18nobn(batch_size=16)
torch.manual_seed(0)
testproblem.set_up()


# Extract model
model = testproblem.net
# print("model = \n", model)
# model = model.eval()  # batch normalization layers not supproted in train mode

# Data
train_loader, _ = testproblem.data._make_train_and_valid_dataloader()
batch_data = next(iter(train_loader))
# torch.save(batch_data, "batch_data.pt")
# batch_data = torch.load("batch_data.pt")
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
# Test BackPACK extend
# ------------------------------------------------------------------------------

extended_model = extend(deepcopy(model), use_converter=True)
extended_lossfunc = extend(deepcopy(lossfunc))

torch.manual_seed(0)
outputs = model(inputs)
loss = lossfunc(outputs, labels)

torch.manual_seed(0)
extended_outputs = extended_model(inputs)
extended_loss = extended_lossfunc(extended_outputs, labels)

assert torch.allclose(outputs, extended_outputs)
assert torch.allclose(loss, extended_loss)


# ------------------------------------------------------------------------------
# Test BackPACK first-order extension
# ------------------------------------------------------------------------------

print("\n===== Test: First-order extension =====")
loss = extended_lossfunc(extended_model(inputs), labels)
with backpack(BatchGrad()):
    loss.backward()
print("Done")

# -------------------------------s----------------------------------------------
# Test BackPACK second-order extension
# ------------------------------------------------------------------------------

print("\n===== Test: Second-order extension =====")
loss = extended_lossfunc(extended_model(inputs), labels)
with backpack(DiagGGNExact()), weight_jac_t_save_memory(True):
    loss.backward()
print("Done")


raise Exception
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
# runner.run("cifar10_resnet18nobn", batch_size=128, num_epochs=2)
print("Done")
