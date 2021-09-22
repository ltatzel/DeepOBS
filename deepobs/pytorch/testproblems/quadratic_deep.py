# -*- coding: utf-8 -*-
"""A simple N-Dimensional Noisy Quadratic Problem with Deep Learning eigenvalues."""

import numpy as np
import torch
from torch import Tensor

from ..datasets.quadratic import quadratic
from .testproblem import UnregularizedTestproblem
from .testproblems_modules import net_quadratic_deep


def random_rotation(D, rng=None):
    """Produce a rotation matrix R.

    The rotation matrix R is in SO(D) (the special orthogonal group SO(D), or
    orthogonal matrices with unit determinant, drawn uniformly from the Haar
    measure.

    The algorithm used is the subgroup algorithm as originally proposed by
    P. Diaconis & M. Shahshahani, "The subgroup algorithm for generating
    uniform random variables". Probability in the Engineering and
    Informational Sciences 1: 15?32 (1987)

    Args:
        D (int): Dimensionality of the matrix.
        rng (numpy.random.RandomState, optional): A random number generator.

    Returns:
        np.array: Random rotation matrix ``R``.

    """
    if rng is None:
        rng = np.random.RandomState(42)
    assert D >= 2
    D = int(D)  # make sure that the dimension is an integer

    # induction start: uniform draw from D=2 Haar measure
    t = 2 * np.pi * rng.uniform()
    R = [[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]

    for d in range(2, D):
        v = rng.normal(size=(d + 1, 1))
        # draw on S_d the unit sphere
        v = np.divide(v, np.sqrt(np.transpose(v).dot(v)))
        e = np.concatenate((np.array([[1.0]]), np.zeros((d, 1))), axis=0)
        # random coset location of SO(d-1) in SO(d)
        x = np.divide((e - v), (np.sqrt(np.transpose(e - v).dot(e - v))))

        D = np.vstack(
            [np.hstack([[[1.0]], np.zeros((1, d))]), np.hstack([np.zeros((d, 1)), R]),]
        )
        R = D - 2 * np.outer(x, np.transpose(x).dot(D))
    # return negative to fix determinant
    return np.negative(R)


class quadratic_deep(UnregularizedTestproblem):
    r"""DeepOBS test problem class for a stochastic quadratic test problem.

    The problem has ``100`` dimensions. 90 % of the eigenvalues of the Hessian
    are drawn from the interval :math:`(0.0, 1.0)` and the other 10 % are from
    :math:`(30.0, 60.0)` simulating an eigenspectrum which has been reported for
    Deep Learning https://arxiv.org/abs/1611.01838.

    This creates a loss functions of the form

    :math:`0.5* (\theta - x)^T * Q * (\theta - x)`

    with Hessian ``Q`` and "data" ``x`` coming from the quadratic data set, i.e.,
    zero-mean normal.

    Args:
      batch_size (int): Batch size to use.
      l2_reg (float): No L2-Regularization (weight decay) is used in this
          test problem. Defaults to ``None`` and any input here is ignored.

    Attributes:
        data: The DeepOBS data set class for the quadratic problem.
        loss_function: None. The output of the model is the loss.
        net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_quadratic_deep).
    """

    def __init__(self, batch_size, l2_reg=None):
        """Create a new quadratic deep test problem instance.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): No L2-Regularization (weight decay) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(quadratic_deep, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the quadratic test problem."""
        hessian = self._make_hessian()
        self._hessian = hessian
        self.net = net_quadratic_deep(hessian)
        self.data = quadratic(self._batch_size)
        self.net.to(self._device)
        self.loss_function = torch.nn.MSELoss
        self.regularization_groups = self.get_regularization_groups()

    @staticmethod
    def _make_hessian(eigvals_small=90, eigvals_large=10):
        rng = np.random.RandomState(42)
        eigenvalues = np.concatenate(
            (
                rng.uniform(0.0, 1.0, eigvals_small),
                rng.uniform(30.0, 60.0, eigvals_large),
            ),
            axis=0,
        )
        D = np.diag(eigenvalues)
        R = random_rotation(D.shape[0], rng=rng)
        Hessian = np.matmul(np.transpose(R), np.matmul(D, R))
        return torch.from_numpy(Hessian).to(torch.float32)

    @staticmethod
    def _compute_accuracy(outputs: Tensor, labels: Tensor) -> float:
        """Return zero as model accuracy (non-existent for this regression task).

        Args:
            outputs: Model predictions.
            labels: Ground truth.

        Returns:
            0
        """
        return 0.0
