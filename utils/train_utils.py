import collections
import numbers
from torch_scatter import scatter_mean, scatter_sum
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from utils.mmcif import chi_pi_periodic_torch
from utils.residue_constants import blosum_numeric, blosum_62_cooccurrance_probs



def get_cooccur_score( res_pred, res_true, batch_idx):
    probs = blosum_62_cooccurrance_probs.to(res_pred.device)
    return scatter_sum((probs[res_pred, res_true] + 4).float(), batch_idx, dim=-1) / scatter_sum((probs[res_true,res_true] + 4).float(), batch_idx, dim=-1)

def get_blosum_score( res_pred, res_true, batch_idx):
    blosum = blosum_numeric.to(res_pred.device)
    return scatter_sum((blosum[res_pred, res_true] + 4).float(), batch_idx, dim=-1) / scatter_sum((blosum[res_true,res_true] + 4).float(), batch_idx, dim=-1)

def get_unnorm_blosum_score(res_pred, res_true, batch_idx):
    blosum = blosum_numeric.to(res_pred.device)
    return scatter_sum((blosum[res_pred, res_true]).float(), batch_idx, dim=-1) / scatter_sum((blosum[res_true,res_true]).float(), batch_idx, dim=-1)

def compute_rmsds(true_pos, x0, batch):
        rmsd = scatter_mean(torch.square(true_pos - x0).sum(-1), batch['ligand'].batch) ** 0.5
        centroid = scatter_mean(x0, batch['ligand'].batch, 0)
        true_cent = scatter_mean(true_pos, batch['ligand'].batch, 0)
        cent_rmsd = torch.square(centroid - true_cent).sum(-1) ** 0.5

        kabsch_rmsd = []
        for i in range(batch.num_graphs):
            x0_ = x0[batch['ligand'].batch == i].cpu().numpy()
            true_pos_ = true_pos[batch['ligand'].batch == i].cpu().numpy()
            try:
                kabsch_rmsd.append(
                    Rotation.align_vectors(x0_, true_pos_)[1] / np.sqrt(x0_.shape[0])
                )
            except:
                kabsch_rmsd.append(np.inf)
        return rmsd.cpu().numpy(), cent_rmsd.cpu().numpy(), np.array(kabsch_rmsd)

def squared_difference(x, y):
    """Computes Squared difference between two arrays."""
    return torch.square(x - y)


def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
    """Masked mean."""
    if drop_mask_channel:
        mask = mask[..., 0]

    mask_shape = mask.shape
    value_shape = value.shape

    assert len(mask_shape) == len(value_shape)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))
    assert isinstance(axis, collections.abc.Iterable), 'axis needs to be either an iterable, integer or "None"'

    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size

    return torch.sum(mask * value, dim=axis) / (torch.sum(mask, dim=axis) * broadcast_factor + eps)

def angle_unit_loss(preds, mask, eps=1e-6):
    # Aux loss to keep vectors in unit circle
    angle_norm = torch.sqrt(torch.sum(torch.square(preds), dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.)
    angle_norm_loss = mask_mean(mask=mask[..., None], value=norm_error)
    return angle_norm_loss

def l2_normalize(x, axis=-1, epsilon=1e-12):
    y = torch.sum(x**2, dim=axis, keepdim=True)
    return x / torch.sqrt(torch.maximum(y, torch.ones_like(y) * epsilon))

def supervised_chi_loss(batch, preds, angles_idx_s=0, angles_idx=11):
    chi_mask = batch['protein'].angle_mask[:, angles_idx_s:angles_idx]
    sin_cos_true_chi = batch["protein"].angles[:, angles_idx_s:angles_idx, :]  # 3 torsion + 3 angle + 5 side chain torsion = 11

    # Extend to backbone angle / torsion angles besides side chain chi angles
    # TODO move somewhere else, inefficient to redefine each time
    chi_pi_periodic = chi_pi_periodic_torch.to(preds.device)[:, angles_idx_s:angles_idx]

    # L2 normalized predicted angles
    angles_sin_cos = l2_normalize(preds, axis=-1)  # [:, :, angles_idx_s:angles_idx, :]

    # One-hot encode and apply periodic mask
    chi_pi_periodic = chi_pi_periodic[batch['protein'].aatype_num]

    # This is -1 if chi is pi-periodic and +1 if it's 2pi-periodic
    shifted_mask = (1 - 2 * chi_pi_periodic)[..., None]
    sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi  # Add + pi if rotation-symmetric

    # Main torsion loss
    sq_chi_error = torch.sum(squared_difference(sin_cos_true_chi, angles_sin_cos), -1)
    sq_chi_error_shifted = torch.sum(squared_difference(sin_cos_true_chi_shifted, angles_sin_cos), -1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)
    sq_chi_loss = mask_mean(mask=chi_mask, value=sq_chi_error)

    # Aux loss to keep vectors in unit circle
    angle_norm_loss = angle_unit_loss(preds, batch['protein'].is_canonical)

    # Final loss
    loss = sq_chi_loss + 0.02 * angle_norm_loss

    return loss

def get_recovered_aa_angle_loss(batch, angles, res_pred, angles_idx_s=0, angles_idx=11):
    if angles is None:
        return torch.tensor(0.0)
    correctly_predicted = torch.zeros_like(batch['protein'].designable_mask).bool()
    correctly_predicted[torch.where(torch.argmax(res_pred, dim=1) == batch['protein'].feat[:, 0].view(-1))[0]] = True
    batch["protein"].angles = batch["protein"].angles[correctly_predicted]
    batch['protein'].angle_mask = batch['protein'].angle_mask[correctly_predicted]
    batch['protein'].aatype_num = batch['protein'].aatype_num[correctly_predicted]
    batch['protein'].is_canonical = batch['protein'].is_canonical[correctly_predicted]
    angles = angles[correctly_predicted]
    if correctly_predicted.sum() == 0:
        return torch.tensor(0.0)
    return supervised_chi_loss(batch, angles, angles_idx_s=angles_idx_s, angles_idx=angles_idx)
