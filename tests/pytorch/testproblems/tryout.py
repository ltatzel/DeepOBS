"""Try out different things"""

import torch
import torch.nn as nn

# ==============================================================================
# Make sure that ``ConstantPad1d`` and ``ZeroPad2d`` do the same thing
# ==============================================================================
input = torch.randn(1, 2, 3)
print("input.shape = ", input.shape)

padding = (0, 0, 0, 0, 1, 2)

m1 = nn.ConstantPad1d(padding, 0)
print("m1(input).shape = ", m1(input).shape)

m2 = nn.ZeroPad2d(padding)
print("m2(input).shape = ", m2(input).shape)

print("Same result? ", torch.allclose(m1(input), m2(input)))
