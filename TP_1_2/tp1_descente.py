import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, linear, Context

# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 1)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 1)
b = torch.randn(1)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    # Pass Forward
    ctx1 = Context()
    ctx2 = Context()

    yhat = linear.forward(ctx1,x,w,b)
    loss = MSE.forward(ctx2,yhat,y)

    # Back propagation
    grad_yhat, grad_y = MSE.backward(ctx2, 1)
    grad_x, grad_w, grad_b = linear.backward(ctx1, grad_yhat)

    #import ipdb;ipdb.set_trace()

    # Tensorboard visualization
    # To visualize type command : tensorboard --logdir=runs in current directory
    writer.add_scalar('Loss/train', loss, n_iter)
    print(f"Itérations {n_iter}: loss {loss}")

    # Updating parameters
    w -= epsilon*grad_w
    b -= epsilon*grad_b

writer.close()

