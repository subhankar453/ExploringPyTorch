import numpy as np
import torch
import torch.nn as nn

W = torch.randn(6)
x = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
b = torch.tensor(3)

y = W*x + b

W1 = torch.tensor(6)
W2 = torch.tensor(6)
W3 = torch.tensor(6)

x1 = torch.tensor([2, 2, 2])
x2 = torch.tensor([3, 3, 3])
x3 = torch.tensor([4, 4, 4])

b = torch.tensor(10)

intermediate_value = W1 * x1 + W2 * x2
final_value = W1 * x1 + W2 * x2 + W3 * x3 + b