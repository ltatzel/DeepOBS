# -*- coding: utf-8 -*-
"""Noise-free quadratic problem (network with two linear layers and MSE-loss)"""

from contextlib import nullcontext
import scipy.linalg
import torch
from torch.utils import data

from ..datasets import dataset
from .testproblem import UnregularizedTestproblem


# The data set
class data_noise_free_quadratic(dataset.DataSet):
    """Dataset class for the noise-free quadratic

    The inputs and corresponding labels for training (and similarly for
    validation and testing) are basically both tensors of size
    ``train_size`` x ``dim``, where ```dim`` is the dimensionality of the
    problem. The input data is arbitraty (because it is later multiplied by
    zero), the labels are zero. The methods below yield the respective
    data loaders for all three sets.
    """

    def __init__(
        self, batch_size, dim, train_size=128, valid_size=128, test_size=128,
    ):
        self._dim = dim

        # Check batch size
        assert batch_size <= min(
            train_size, valid_size, test_size
        ), "Batch size exceeds size of training/validation/test set"

        self._train_size = train_size
        self._valid_size = valid_size
        self._test_size = test_size

        # This attribute is needed by _make_train_eval_dataloader
        self._train_eval_size = self._train_size

        super().__init__(batch_size)

    def _make_train_and_valid_dataloader(self):
        """Creates the training and validation data loader."""

        # Training data
        train_data = torch.rand(self._train_size, self._dim)
        train_labels = torch.zeros(self._train_size, self._dim)
        train_dataset = data.TensorDataset(train_data, train_labels)
        train_loader = self._make_dataloader(train_dataset, shuffle=True)

        # Validation data
        valid_data = torch.rand(self._valid_size, self._dim)
        valid_labels = torch.zeros(self._valid_size, self._dim)
        valid_dataset = data.TensorDataset(valid_data, valid_labels)
        valid_loader = self._make_dataloader(valid_dataset)

        return train_loader, valid_loader

    def _make_test_dataloader(self):
        """Creates the test data loader."""

        # Test data
        test_data = torch.rand(self._test_size, self._dim)
        test_labels = torch.zeros(self._test_size, self._dim)
        test_dataset = data.TensorDataset(test_data, test_labels)
        test_loader = self._make_dataloader(test_dataset)

        return test_loader


# Some helper functions
def set_param(linear_layer, param, param_str, req_grad):
    """Set weights (`param_str = weight`) or biases (`param_str = bias`) in 
    linear layer and choose if these parameters are trainable.
    """
    p = getattr(linear_layer, param_str)

    if param.shape != p.shape:
        raise ValueError("parameters don't have the right shape")

    p.data = param
    p.requires_grad = req_grad


def torch_to_numpy(tensor):
    """Convert a torch tensor to a numpy array"""
    return tensor.detach().cpu().numpy()


def numpy_to_torch(array):
    """Convert a numpy array to a torch float tensor"""
    return (torch.from_numpy(array)).to(torch.float32)


# The network
def get_noise_free_quadratic_net(H, theta):
    """Build the network for the noise-free quadratic

    The network is based on the Hessian ``H`` and the vector ``theta``. It
    is designed such that the MSE loss of the network (which is parameterized
    by ``theta``) is ``theta.T @ H @ theta`` for arbitrary inputs with labels
    that are zero.
    """
    dim = H.shape[0]

    # Use the matrix square root from scipy
    H_sqrt = numpy_to_torch(scipy.linalg.sqrtm(torch_to_numpy(H), disp=True))

    # First layer returns ``0 @ x + theta = theta``
    L1 = torch.nn.Linear(dim, dim, bias=True)
    set_param(L1, torch.zeros(dim, dim), "weight", req_grad=False)
    set_param(L1, theta.reshape(dim), "bias", req_grad=True)

    # Second layer returns ``H_sqrt @ theta``
    L2 = torch.nn.Linear(dim, dim, bias=False)
    set_param(L2, H_sqrt, "weight", req_grad=False)

    return torch.nn.Sequential(L1, L2)


# The problem class
class noise_free_quadratic(UnregularizedTestproblem):
    """Problem class for the noise-free quadratic

    The problem (determined by the Hessian and initial network parameters) is
    defined in the constructor. It is a quadratic problem of the form
    ``theta.T @ H @ theta``, where ``H`` is the Hessian and ``theta``
    corresponds to the trainable parameters of the network. They are initially
    set to ``theta_init``.
    """

    def __init__(self, batch_size, weight_decay=None):
        """Here, the quadratic problem is defined. Note that the batch size
        is arbitrary: since the problem is noise-free, the batch size has no
        impact on the resulting loss.
        """
        super().__init__(batch_size, weight_decay)

        # Define quadratic problem
        D = 20
        H_diag = torch.Tensor([i ** 2 for i in range(1, D + 1)])
        self._H = torch.diagflat(H_diag)
        self._theta_init = 100 * torch.ones(D)

        # Check problem
        self.check_problem()
        self._dim = self._H.shape[0]

    def check_problem(self):
        """Make sure that the attributes ``self._H`` and ``self._theta_init``
        "match" (dimensions) and that the Hessian is symmetric pos. definite.
        """
        H = self._H
        theta_init = self._theta_init

        # Check dimensions
        dim1, dim2 = H.shape
        assert dim1 == dim2, "Hessian has to be square"
        assert theta_init.shape == torch.Size(
            [dim1]
        ), "`theta_init` has to be 1D Tensor of the right size"

        # Check symmetric positive definite
        assert torch.allclose(H.T, H), "Hessian has to be symmetric"
        H_eigvals, _ = torch.symeig(H, eigenvectors=False)
        assert torch.all(H_eigvals > 0), "Hessian has to be positive definite"

    def set_up(self):
        """Initialize the global attributes ``net``, ``data`` and
        ``loss_function``.
        """
        # Network
        H_net = self._dim * self._H
        self.net = get_noise_free_quadratic_net(H_net, self._theta_init)
        self.net.to(self._device)

        # Data set
        self.data = data_noise_free_quadratic(self._batch_size, dim=self._dim)

        # Loss function
        self.loss_function = torch.nn.MSELoss

        # Create regularization groups (in our case no regularization is used)
        self.regularization_groups = self.get_regularization_groups()

    def get_batch_loss_and_accuracy_func(
        self, reduction="mean", add_regularization_if_available=True
    ):
        """The original method from the base class doesn't work here
        (especially for the accuracy), so we overwrite it. It is basically a
        copy of the original method, but we set the accuracy to zero instead
        of trying to computing it. Note that the accuracy does't make sense
        as a metric for our particular problem.
        """
        inputs, labels = self._get_next_batch()
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)

        loss_function = self.loss_function(reduction=reduction)

        def forward_func():

            # Evaluate loss: In evaluation phase no gradient is needed
            with torch.no_grad() if self.phase in [
                "train_eval",
                "test",
                "valid",
            ] else nullcontext():
                outputs = self.net(inputs)
                loss = loss_function(outputs, labels)

            # Evaluate regularizer loss
            if add_regularization_if_available:
                regularizer_loss = self.get_regularization_loss()
            else:
                regularizer_loss = torch.zeros(1).to(self._device)

            # Accuracy
            accuracy = 0.0

            return loss + regularizer_loss, accuracy

        return forward_func
