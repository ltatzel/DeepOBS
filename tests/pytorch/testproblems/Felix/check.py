"""Compare the ResNets of ``original.py`` and ``integrated.py``."""

import integrated
import numpy as np
import original
import torch
from backpack import backpack, extend, extensions
from backpack.utils.examples import autograd_diag_ggn_exact

# from vivit.linalg.eigh import EighComputation

NETS = [
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    # "resnet1202", # produces NaNs in forward pass with random input
]


def compare_forward(net_str: str):
    """Compare the net from ``original.py`` and ``integrated.py``."""
    print(f"Comparing forward pass for {net_str}")
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

    torch.manual_seed(0)
    net_original = getattr(integrated, net_str)()
    net_original.eval()

    torch.manual_seed(0)
    net_integrated = getattr(original, net_str)()
    net_integrated.eval()

    N = 8
    X, y = torch.rand(N, 3, 32, 32), torch.randint(0, 10, (N,))

    torch.manual_seed(1)
    output_original = net_original(X)
    loss_original = loss_func(output_original, y)
    loss_original.backward()

    torch.manual_seed(1)
    output_integrated = net_integrated(X)
    loss_integrated = loss_func(output_integrated, y)
    loss_integrated.backward()

    assert torch.allclose(output_original, output_integrated), "Outputs don't match"
    print("\tOutputs match")
    assert torch.allclose(loss_original, loss_integrated), "Losses don't match"
    print("\tLosses match")
    for p1, p2 in zip(net_original.parameters(), net_integrated.parameters()):
        assert torch.allclose(
            p1.grad, p2.grad
        ), f"Gradients of parameter with shape {p1.shape} does not match"
    print("\tGradients match")


def run_backpack_batch_grad(net_str: str):
    """Run BackPACK's BatchGrad extension with the specified net."""
    print(f"Running BatchGrad for {net_str}")
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

    torch.manual_seed(0)
    net = getattr(integrated, net_str)()
    net.eval()

    loss_func = extend(loss_func)
    net = extend(net)

    N = 8
    X, y = torch.rand(N, 3, 32, 32), torch.randint(0, 10, (N,))

    loss = loss_func(net(X), y)
    with backpack(extensions.BatchGrad()):
        loss.backward()

    print("\tPassing")


def run_backpack_sqrt_ggn_exact(net_str: str):
    """Run BackPACK's SqrtGGNExact extension with the specified net."""
    print(f"Running SqrtGGNExact for {net_str}")
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

    torch.manual_seed(0)
    net = getattr(integrated, net_str)()
    net.eval()

    loss_func = extend(loss_func)
    net = extend(net)

    N = 8
    X, y = torch.rand(N, 3, 32, 32), torch.randint(0, 10, (N,))

    loss = loss_func(net(X), y)
    with backpack(extensions.SqrtGGNExact()):
        loss.backward()

    print("\tPassing")


def compare_diag_ggn(net_str: str, num_elements: int = 10):
    """Compare elements of the GGN diagonal."""
    print(f"Comparing {num_elements} elements of the GGN diagonal for {net_str}")

    # input data
    torch.manual_seed(1)
    N = 8
    X, y = torch.rand(N, 3, 32, 32), torch.randint(0, 10, (N,))

    # autograd
    torch.manual_seed(0)
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    net = getattr(integrated, net_str)()
    net.eval()

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    idx = [i.item() for i in torch.linspace(0, num_params - 1, num_elements).int()]

    diag_autograd = autograd_diag_ggn_exact(X, y, net, loss_func, idx=idx)

    # BackPACK
    torch.manual_seed(0)
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    net = getattr(integrated, net_str)()
    net.eval()

    loss_func = extend(loss_func)
    net = extend(net)

    loss = loss_func(net(X), y)
    with backpack(extensions.DiagGGNExact()):
        loss.backward()

    diag_backpack = torch.cat(
        [p.diag_ggn_exact.flatten() for p in net.parameters() if p.requires_grad]
    )
    diag_backpack = diag_backpack[idx]

    # check
    assert torch.allclose(
        diag_autograd, diag_backpack
    ), "GGN diagonal elements don't match"

    print("\tPassing")


def run_vivit_largest_eigenpair(net_str):
    """Compute the largest eigenpair of the network using the ViViT extension."""
    print(f"Computing the largest eigenpair for {net_str}")

    # input data
    torch.manual_seed(1)
    N = 8
    X, y = torch.rand(N, 3, 32, 32), torch.randint(0, 10, (N,))

    torch.manual_seed(0)
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    net = getattr(integrated, net_str)()
    net.eval()

    loss_func = extend(loss_func)
    net = extend(net)
    computation = EighComputation()

    def keep_criterion(evals):
        """Filter criterion to keep the largest eigenvalue."""
        return [len(evals) - 1]

    parameters = list(net.parameters())
    group = {"params": parameters, "criterion": keep_criterion}
    param_groups = [group]

    loss = loss_func(net(X), y)
    with backpack(
        computation.get_extension(),
        extension_hook=computation.get_extension_hook(param_groups),
    ):
        loss.backward()

    print("\tPassing")


if __name__ == "__main__":
    for net_str in NETS:
        compare_forward(net_str)

    for net_str in NETS:
        run_backpack_batch_grad(net_str)

    for net_str in NETS:
        compare_diag_ggn(net_str)

    for net_str in NETS:
        run_backpack_sqrt_ggn_exact(net_str)

    # for net_str in NETS:
    #    run_vivit_largest_eigenpair(net_str)