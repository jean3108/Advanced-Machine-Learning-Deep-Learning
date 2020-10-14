import torch
from tp1 import mse, lin, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
res = torch.autograd.gradcheck(mse, (yhat, y))

print("True MSE grad")
print("-------------------")

# Test du gradient de Linear

X = torch.randn(10,20, requires_grad=True, dtype=torch.float64)
W = torch.randn(20,5, requires_grad=True, dtype=torch.float64)
b = torch.randn(5, requires_grad=True, dtype=torch.float64)
res = torch.autograd.gradcheck(lin, (X, W, b))

print("True Linear grad")
