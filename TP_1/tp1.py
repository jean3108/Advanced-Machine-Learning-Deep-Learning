# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """MSE(Y,Yhat) = 1/q * (Yhat-Y)**2"""
    @staticmethod
    def forward(ctx, yhat, y):
        
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)
        q, p = yhat.shape # batch_size, output_dim
        result = (1/q)*torch.norm(yhat - y)**2
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        """
        ctx : context with saved tensor, here Yhat and Y 
        grad_output : gradient of the output function (L) respect to the MSE ouputs (C)
                      here, C = L o mse(Yhat, Y) = Id o mse(Yhat, Y), so grad_ouput = grad(Id) = 1
        """

        yhat, y = ctx.saved_tensors
        q, _ = yhat.shape # batch_size, output_dim
        result_y = -(2/q)*grad_output*(yhat-y)
        result_yhat = (2/q)*grad_output*(yhat-y)
        
        return result_yhat, result_y


class linear(Function):
    """ f(X,W,b) = X @ W.T + b"""
    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X,W,b)
        result = torch.matmul(X,W) + b
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        
        """
        ctx : context with saved tensor, here X, W and b 
        grad_output : gradient of the output function (mse) respect to the Linear outputs (Yhat)
                      here, C = L o mse(f(X,W,b), Y) = mse(f(X,W,b), Y)
                      so grad_ouput = grad(mse) respect to Yhat = 2/q * (Yhat-Y)
        """
        X, W, B = ctx.saved_tensors
        p = B.shape # output_dim

        # compute gradients
        result_X = torch.matmul(grad_output,W.T)
        result_W = torch.matmul(X.T,grad_output)
        result_b = grad_output.sum(0)

        return result_X, result_W, result_b



mse = MSE.apply
lin = linear.apply


