import numpy as np
import torch
import cc3d
import math

def reconstruction_loss(x_pred, x, sum_mean):
    if sum_mean == "sum":
        rec_loss = torch.nn.functional.binary_cross_entropy(x_pred, x, reduction="sum")
    elif sum_mean == "mean":
        rec_loss = torch.nn.functional.binary_cross_entropy(x_pred, x, reduction="none").sum(dim=(1, 2, 3)).view(-1, 1).mean()
    return rec_loss

def latent_loss(mean, log_var, log_det, sum_mean):
    if sum_mean == "sum":
        # kld = 0.5 * torch.sum(-1 - log_var + mean.pow(2) + log_var.exp())
        kld = log_var.sum(1) + math.log(2*(math.pi*math.e)**.5)
        log_det = -log_det
        return torch.sum(kld - log_det)
    elif sum_mean == "mean":
        kld = 0.5 * torch.sum(-1 - log_var + mean.pow(2) + log_var.exp(), dim=1, keepdim=True)
        return (kld - log_det).mean()

def loss_fun(x_pred, x, mean, log_var, log_det, add_cc3d=False, sum_mean="sum"):
    rec_loss = reconstruction_loss(x_pred, x, sum_mean)
    lat_loss = latent_loss(mean, log_var, log_det, sum_mean)

    if add_cc3d:
        cc3d_loss = loss_cc3d(x_pred, x)
        return rec_loss - lat_loss + cc3d_loss, rec_loss, lat_loss, cc3d_loss

    return rec_loss - lat_loss, rec_loss, lat_loss

def loss_cc3d(pred_voxel, voxel):
    pred_batch = (pred_voxel.detach().cpu().numpy() >= 0.5).astype(np.uint8)
    gt_batch = voxel.detach().cpu().numpy().astype(np.uint8)
    cc3d_loss = 0
    for i in range(pred_batch.shape[0]):
        cc3d_loss += np.absolute(np.subtract(np.max(cc3d.connected_components(gt_batch[i])),
                                             np.max(cc3d.connected_components(pred_batch[i].reshape(32, 32, 32)))))
    return torch.tensor(cc3d_loss / pred_batch.shape[0], dtype=torch.float32)
