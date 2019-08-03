import torch
import torch.nn as nn
from torch.autograd import Function
import morphpool_cuda as morphpool_cuda
import numpy as np


class MorphPoolFunction(Function):
    @staticmethod
    def forward(ctx, input, mask, num_morph, kernel_size):
        with torch.no_grad():
            _kernel_size = torch.Tensor([kernel_size]).float().cuda()
            _num_morph = torch.Tensor([num_morph]).float().cuda()
        (outputs, outputs_indices) = morphpool_cuda.forward(input, mask, _num_morph, _kernel_size)
        ctx.save_for_backward(input, mask, outputs_indices, outputs, _num_morph, _kernel_size)

        return outputs

    @staticmethod
    def backward(ctx, grad):
        input, mask, outputs_indices, outputs, _num_morph, _kernel_size = ctx.saved_variables
        (grad_out,) = morphpool_cuda.backward(grad.contiguous(), input, mask, outputs_indices, outputs, _num_morph,
                                              _kernel_size)
        return (grad_out, None, None, None, None)


class MorphPool3D(nn.Module):
    def __init__(self):
        super(MorphPool3D, self).__init__()
        _P3 = [np.zeros((3, 3, 3)) for i in range(9)]
        _P3[0][:, :, 1] = 1
        _P3[1][:, 1, :] = 1
        _P3[2][1, :, :] = 1
        _P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
        _P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
        _P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
        _P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
        _P3[7][[0, 1, 2], [0, 1, 2], :] = 1
        _P3[8][[0, 1, 2], [2, 1, 0], :] = 1

        self.register_buffer('morph', torch.Tensor(_P3))

    def forward(self, input, erode=False):
        if not erode:
            x = MorphPoolFunction.apply(input, self.morph, self.morph.shape[0], 3)
            return x.min(2)[0]
        else:
            x = -MorphPoolFunction.apply(-input, self.morph, self.morph.shape[0], 3)
            return x.max(2)[0]
