"""This scripts tests if the loss, defined by the problem class, actually
corresponds to a quadratic loss. We do this by repeadedly defining problems
(determined by a random Hessian and an initial network parameter vector) and
checking if the model's loss corresponds to the manually computed quadratic
loss ``theta_init.T @ H @ theta_init``. It is assumed that the loss-function is 
used with ``reduction="mean"``.  
"""

import pytest
import torch
from deepobs.pytorch.testproblems import noise_free_quadratic

NOF_BATCHES = 2

DIMS = [1, 5, 50]
IDS_DIMS = [f"dimension={dim}" for dim in DIMS]

SEEDS = [0, 1, 42]
IDS_SEEDS = [f"seed_value={seed}" for seed in SEEDS]


@pytest.mark.parametrize("seed", SEEDS, ids=IDS_SEEDS)
@pytest.mark.parametrize("dim", DIMS, ids=IDS_DIMS)
def test_func(seed, dim):

    # Initialize testproblem
    nf_quadratic = noise_free_quadratic(batch_size=8)

    # Create random symmetric pos. definite Hessian and `theta_init`
    torch.manual_seed(seed)

    theta_init = torch.rand(dim)
    R = torch.rand(dim, dim)
    H = R @ R.T + 0.01 * torch.diag(torch.ones(dim))

    # Set up the problem
    nf_quadratic._dim = dim
    nf_quadratic._H = H
    nf_quadratic._theta_init = theta_init
    nf_quadratic.check_problem()
    nf_quadratic.set_up()

    # Extract dataset, net, loss function, device
    data = nf_quadratic.data
    train_loader, _ = data._make_train_and_valid_dataloader()
    net = nf_quadratic.net
    loss_function = nf_quadratic.loss_function(reduction="mean")
    device = torch.device(nf_quadratic._device)

    for batch_idx in range(NOF_BATCHES):

        # Get some data (the inputs shouldn't affect the loss at all)
        input, labels = list(train_loader)[batch_idx]
        input = input.to(device)
        labels = labels.to(device)

        # Compare the model's loss with the manually computed loss
        loss_model = loss_function(net(input), labels)
        loss_manually = theta_init.T @ H @ theta_init

        assert torch.allclose(
            loss_model, loss_manually
        ), "The model's loss and the manually computed quadratic loss deviate."


if __name__ == "__main__":

    # For debugging
    test_func(seed=0, dim=10)
