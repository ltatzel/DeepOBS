"""Test the CIFAR-10 problem with the ResNet32.

Check that BackPACK's extensions and ViViT works with this architecture. 
Also check compatibility with DeepOBS by running the StandardRunner.
"""

from copy import deepcopy

import torch
from backpack import backpack, extend
from backpack.core.derivatives.convnd import weight_jac_t_save_memory
from backpack.extensions import BatchGrad, DiagGGNExact
from torch.optim import SGD

from deepobs import pytorch as pt
from deepobs.pytorch.testproblems.cifar10_resnet32 import cifar10_resnet32
from vivit.linalg.eigh import EighComputation

# Set up testproblem
testproblem = cifar10_resnet32(batch_size=16)
torch.manual_seed(0)
testproblem.set_up()

# Extract model and set to evaluation mode (this is required by BackPACK)
model = testproblem.net.eval()

# Data
torch.manual_seed(0)
train_loader, _ = testproblem.data._make_train_and_valid_dataloader()
batch_data = next(iter(train_loader))
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
print("\n===== Test: Backpack extend =====")

extended_model = extend(deepcopy(model))
extended_lossfunc = extend(deepcopy(lossfunc))

torch.manual_seed(0)
outputs = model(inputs)
loss = lossfunc(outputs, labels)

torch.manual_seed(0)
extended_outputs = extended_model(inputs)
extended_loss = extended_lossfunc(extended_outputs, labels)

assert torch.allclose(outputs, extended_outputs)
assert torch.allclose(loss, extended_loss)
print("Done")


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


# ------------------------------------------------------------------------------
# Test ViViT
# ------------------------------------------------------------------------------
print("\n===== Test: ViViT's Eigh =====")

computation = EighComputation()


def keep_criterion(evals):
    """Filter criterion to keep the largest eigenvalue."""
    return [len(evals) - 1]


parameters = list(extended_model.parameters())
group = {"params": parameters, "criterion": keep_criterion}
param_groups = [group]

loss = extended_lossfunc(extended_model(inputs), labels)
with backpack(
    computation.get_extension(),
    extension_hook=computation.get_extension_hook(param_groups),
):
    loss.backward()
print("Done")


# ------------------------------------------------------------------------------
# Test DeepOBS
# ------------------------------------------------------------------------------
optimizer_class = SGD
hyperparams = {
    "lr": {"type": float, "default": 0.01},
    "momentum": {"type": float, "default": 0.90},
    "nesterov": {"type": bool, "default": False},
}

runner = pt.runners.StandardRunner(optimizer_class, hyperparams)
runner.run("cifar10_resnet32", batch_size=128, num_epochs=2)
print("Done")
