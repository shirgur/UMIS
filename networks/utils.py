import torch
import torch.nn as nn
import torch.nn.functional as F


class GradXYZ(nn.Module):
    def __init__(self):
        super(GradXYZ, self).__init__()

        self.padding = 1
        self.register_buffer('dX', torch.Tensor([[[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [-1 / 2, 0, 1 / 2],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]]
                                                 ]
                                                ).unsqueeze(0).unsqueeze(0))
        self.register_buffer('dY', torch.Tensor([[[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, -1 / 2, 0],
                                                  [0, 0, 0],
                                                  [0, 1 / 2, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]]
                                                 ]
                                                ).unsqueeze(0).unsqueeze(0))
        self.register_buffer('dZ', torch.Tensor([[[0, 0, 0],
                                                  [0, -1 / 2, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 1 / 2, 0],
                                                  [0, 0, 0]]
                                                 ]
                                                ).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        dx = F.conv3d(x, self.dX, padding=self.padding).abs()
        dy = F.conv3d(x, self.dY, padding=self.padding).abs()
        dz = F.conv3d(x, self.dZ, padding=self.padding).abs()
        return dx + dy + dz


def norm_ip(img, min, max):
    out = torch.clamp(img, min=min, max=max)
    out = (out - min) / (max - min + 1e-5)
    return out


def norm_range(t, range=None):
    if range is not None:
        return norm_ip(t, range[0], range[1])
    else:
        return norm_ip(t, float(t.min()), float(t.max()))