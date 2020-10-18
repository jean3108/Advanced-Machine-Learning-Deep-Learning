import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
from tp1 import MSE, linear, Context


writer = SummaryWriter()

data=datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()

# Data tensors
x = torch.tensor(datax,dtype=torch.float64)
y = torch.tensor(datay,dtype=torch.float64).reshape(-1,1)
q, n = x.shape # batch_size, input features dimension
q, p = y.shape # batch_size, output dimension

# Data normalisation
x -= x.min(0, keepdim = True)[0]
x /= x.max(0, keepdim = True)[0] 

# Parameters
W = torch.randn((n, p), requires_grad = True, dtype=torch.float64)
B = torch.randn((p), requires_grad = True, dtype=torch.float64)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    # Pass Forward
    # TP2 -> with autograd

    yhat = linear.apply(x,W,B)
    loss = MSE.apply(yhat,y)

    # Back propagation TP2
    loss.backward()


    # Tensorboard visualization
    # To visualize type command : tensorboard --logdir=runs in current directory
    writer.add_scalar('Loss/train', loss, n_iter)
    print(f"Itérations {n_iter}: loss {loss}")

    # Updating parameters
    # inplace operation -> impossible with grad = True
    with torch.no_grad():
        W -= epsilon*W.grad
        B -= epsilon*B.grad

    # Reset gradient
    W.grad.data.zero_()
    B.grad.data.zero_()

writer.close()