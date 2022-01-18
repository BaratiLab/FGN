import numba as nb
import numpy as np
import torch
from torch.autograd import Function
from Constants import MPS_KERNEL as w
from Constants import BASE_RADIUS, ND_RAIUS, GRAD_RADIUS, LAP_RADIUS


class DivOp(Function):
    """Compute the divergence of a given physics value.
        Implement in terms of pytorch autograd function because we need to minimize the
        compressibility during training"""
    @staticmethod
    def forward(ctx, val, Adj_arr, N0):
        if not isinstance(val, torch.Tensor):
            val = torch.from_numpy(val)
        A = Adj_arr.clone() * (3. / N0)
        val.require_grad = True
        div_val = torch.zeros((val.size(0), 1), dtype=torch.float32)
        ctx.save_for_backward(A)
        for dim in range(3):
            sliced_val = val[:, dim].view(-1, 1)
            div_val += torch.sparse.mm(A[dim], sliced_val).view(-1, 1)
        return div_val

    @staticmethod
    def backward(ctx, grad_input):
        grad_input.double()
        A, = ctx.saved_tensors
        grad_output = []
        for dim in range(3):
            grad_output += [torch.sparse.mm(
                     A[dim], grad_input).view(-1, 1)]
        grad_output = torch.stack(grad_output).squeeze().view(-1, 3)
        return grad_output, None, None


class LapOp(Function):
    @staticmethod
    def forward(ctx, val, Adj_arr, N0, lam):
        if not isinstance(val, torch.Tensor):
            val = torch.from_numpy(val)
        A = Adj_arr * (2. * 3.)/(N0 * lam)
        out = torch.sparse.mm(A, val)
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, grad_input):
        grad_input.double()
        A, = ctx.saved_tensors
        grad_output = torch.sparse.mm(A, grad_input)
        return grad_output, None, None, None, None


Divergence = DivOp.apply
Laplacian = LapOp.apply


class GradientOp(object):
    @staticmethod
    def forward(val, val_min, A, A_diag, N0, to_numpy=True):
        if not isinstance(val, torch.Tensor):
            val = torch.from_numpy(val)
        # val.require_grad = True
        val = val.float().view(-1, 1)
        val_min = val_min.view(-1, 1)
        grad_val = torch.zeros((val.size(0), 3), dtype=torch.float32)
        # ctx.save_for_backward(A)

        for dim in range(3):
            grad_val[:, dim] = (3. / N0) * (torch.sparse.mm(A[dim], val) - torch.sparse.mm(A_diag[dim], val_min)).view(-1,)
        if to_numpy:
            return grad_val.detach().numpy()
        else:
            return grad_val


class CollisionOp(object):
    @staticmethod
    def forward(vel, Adj_arr, coef_rest):
        if not isinstance(vel, torch.Tensor):
            vel = torch.from_numpy(vel)
        fdt = torch.zeros_like(vel)
        fdt -= torch.sparse.mm(Adj_arr, vel)
        fdt *= (coef_rest + 1.0) / 2.0
        correction = torch.sparse.mm(Adj_arr, fdt)
        return correction


class SumOp(object):
    @staticmethod
    def forward(Adj_arr, device='cpu', to_numpy=True):
        A = Adj_arr.clone()
        I = torch.ones((A.size(0), 1), dtype=torch.float32).to(device)
        out = torch.sparse.mm(A, I)
        if to_numpy:
            return out.cpu().numpy()
        else:
            return out


