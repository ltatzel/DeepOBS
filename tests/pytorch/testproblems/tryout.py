"""Try out different things"""

import torch
import torch.nn as nn

# ==============================================================================
# Make sure that ``ConstantPad1d`` and ``ZeroPad2d`` do the same thing
# ==============================================================================
print("\n``ConstantPad1d`` vs. ``ZeroPad2d``")

input = torch.randn(1, 2, 3)
print("input.shape = ", input.shape)

padding1 = (0, 0, 0, 0, 1, 2)
m1 = nn.ConstantPad1d(padding1, 0)
print("m1(input).shape = ", m1(input).shape)

padding2 = (0, 0, 0, 0, 1, 2)
m2 = nn.ZeroPad2d(padding2)
print("m2(input).shape = ", m2(input).shape)

print("Same result? ", torch.allclose(m1(input), m2(input)))


# ==============================================================================
# Turn the ``view`` operation into ``Flatten``
# ==============================================================================
print("\n``view`` vs. ``Flatten``")

input = torch.randn(128, 64, 1, 1)

res1 = input.view(input.size(0), -1)
print("res1.shape = ", res1.shape)

m = nn.Flatten()
res2 = m(input)
print("res2.shape = ", res2.shape)

print("Same result? ", torch.allclose(res1, res2))