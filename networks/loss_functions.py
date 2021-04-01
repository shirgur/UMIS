import torch
from torch.nn import functional as F


def euler_lagrange(data, seg, area, c0, c1, rec, grad_seg, grad_rec, args):
    dimsum = list(range(1, len(data.shape)))

    # Image force
    image_force = (args.lmd1 * (data - c0).pow(2) - args.lmd2 * (data - c1).pow(2)) * grad_seg

    # Rank loss
    rank_loss = torch.exp(c1 - c0).mean()

    # Entropy loss
    etropy_loss = (- seg * (seg + 1e-5).log()).mean()

    # Variance loss
    seg_size = seg.shape[1] * seg.shape[2] * seg.shape[3] * seg.shape[4]
    var_loss = (seg.pow(2).sum(dim=dimsum) / seg_size) - (seg.sum(dim=dimsum) / seg_size).pow(2)
    var_loss = torch.exp(var_loss).mean()

    # Reconstruction loss
    rec_loss = F.mse_loss(rec, data) + grad_rec.mean()

    # Image force loss
    one_opt = image_force[image_force < 0]
    one_opt_seg = seg[image_force < 0]
    zero_opt = image_force[image_force > 0]
    zero_opt_seg = seg[image_force > 0]
    image_force_loss = 0
    if len(one_opt) > 0:
        image_force_loss += torch.exp(one_opt * one_opt_seg).mean() * 0.5
    if len(zero_opt) > 0:
        image_force_loss += torch.exp(- zero_opt * (1 - zero_opt_seg)).mean() * 0.5

    # Compound loss
    loss = image_force_loss
    loss += 1e-2 * rank_loss
    loss += 1e-3 * etropy_loss
    loss += 1e-3 * var_loss
    loss += 1e-6 * rec_loss
    loss += args.lmd_area * area.mean()

    return loss


def level_set(data, seg, area, c0, c1, args):
    dimsum = list(range(1, len(data.shape)))

    # Level-set loss
    loss_ls = (data - c0).pow(2) * seg + (data - c1).pow(2) * (1 - seg)

    # Rank loss
    rank_loss = (c1 - c0).clamp(min=0).mean()

    # Compound loss
    loss = 1e-2 * loss_ls.sum(dim=dimsum).mean()
    loss += 1e3 * rank_loss
    loss += c0.mean()
    loss += args.lmd_area * area.mean()

    return loss
