import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
res = torch.autograd.gradcheck(mse, (yhat, y))


#  TODO:  Test du gradient de Linear

