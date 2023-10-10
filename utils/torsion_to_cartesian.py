# Take backbone coordinates and torsion angles, output full atom coordinates bb + sc
from mp_nerf.massive_pnerf import *
from mp_nerf.utils import *
from mp_nerf.kb_proteins import *

from utils.angle import internal_coordinates_torch
from utils.residue_constants import restype_order_with_x_inverse


def scn_cloud_mask(seq, coords=None, strict=False):
    """ Gets the boolean mask atom positions (not all aas have same atoms).
        Inputs:
        * seqs: (length) iterable of 1-letter aa codes of a protein
        * coords: optional .(batch, lc, 3). sidechainnet coords.
                  returns the true mask (solves potential atoms that might not be provided)
        * strict: bool. whther to discard the next points after a missing one
        Outputs: (length, 14) boolean mask
    """
    if coords is not None:
        start = ((rearrange(coords, 'b (l c) d -> b l c d', c=14) != 0).sum(dim=-1) != 0).float()
        # if a point is 0, the following are 0s as well
        if strict:
            for b in range(start.shape[0]):
                for pos in range(start.shape[1]):
                    for chain in range(start.shape[2]):
                        if start[b, pos, chain].item() == 0:
                            start[b, pos, chain:] *= 0
        return start
    return torch.tensor(np.array([SUPREME_INFO[aa]['cloud_mask'] for aa in seq]))
def scn_bond_mask(seq):
    """ Inputs:
        * seqs: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 14) maps point to bond length
    """
    return torch.tensor(np.array([SUPREME_INFO[aa]['bond_mask'] for aa in seq]))
def scn_angle_mask(seq, angles=None, device=None):
    """ Inputs:
        * seq: (length). iterable of 1-letter aa codes of a protein
        * angles: (length, 12). [phi, psi, omega, b_angle(n_ca_c), b_angle(ca_c_n), b_angle(c_n_ca), 6_scn_torsions]
        Outputs: (L, 14) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """
    device = angles.device if angles is not None else torch.device("cpu")
    precise = angles.dtype if angles is not None else torch.get_default_dtype()
    torsion_mask_use = "torsion_mask" if angles is not None else "torsion_mask_filled"
    # get masks
    theta_mask = torch.tensor(np.array([SUPREME_INFO[aa]['theta_mask'] for aa in seq]), dtype=precise).to(device)
    torsion_mask = torch.tensor(np.array([SUPREME_INFO[aa][torsion_mask_use] for aa in seq]), dtype=precise).to(device)
    # adapt general to specific angles if passed
    if angles is not None:
        # fill masks with angle values
        theta_mask[:, 0] = angles[:, 4]  #  ca_c_n
        theta_mask[1:, 1] = angles[:-1, 5]  #  c_n_ca
        theta_mask[:, 2] = angles[:, 3]  # n_ca_c
        # backbone_torsions
        torsion_mask[:, 0] = angles[:, 1]  # n determined by psi of previous
        torsion_mask[1:, 1] = angles[:-1, 2]  # ca determined by omega of previous
        torsion_mask[:, 2] = angles[:, 0]  # c determined by phi
        # https://github.com/jonathanking/sidechainnet/blob/master/sidechainnet/structure/StructureBuilder.py#L313
        torsion_mask[:, 3] = angles[:, 1] - np.pi
        # add torsions to sidechains - no need to modify indexes due to torsion modification
        #  since extra rigid modies are in terminal positions in sidechain
        to_fill = torsion_mask != torsion_mask  # "p" fill with passed values
        to_pick = torsion_mask == 999  # "i" infer from previous one
        for i, aa in enumerate(seq):
            #  check if any is nan -> fill the holes
            number = to_fill[i].long().sum()
            torsion_mask[i, to_fill[i]] = angles[i, 6:6 + number]
            # pick previous value for inferred torsions
            for j, val in enumerate(to_pick[i]):
                if val:
                    torsion_mask[i, j] = torsion_mask[i, j - 1] - np.pi  # pick values from last one.
            # special rigid bodies anomalies:
            if aa == "I":  #  scn_torsion(CG1) - scn_torsion(CG2) = 2.13 (see KB)
                torsion_mask[i, 7] += torsion_mask[i, 5]
            elif aa == "L":
                torsion_mask[i, 7] += torsion_mask[i, 6]
    torsion_mask[-1, 3] += np.pi
    return torch.stack([theta_mask, torsion_mask], dim=0)
def scn_index_mask(seq):
    """ Inputs:
        * seq: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 11, 3) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """
    idxs = torch.tensor(np.array([SUPREME_INFO[aa]['idx_mask'] for aa in seq]))
    return rearrange(idxs, 'l s d -> d l s')
def build_scaffolds_from_scn_angles(seq, angles=None, coords=None, device="auto"):
    """ Builds scaffolds for fast access to data
        Inputs:
        * seq: string of aas (1 letter code)
        * angles: (L, 12) tensor containing the internal angles.
                  Distributed as follows (following sidechainnet convention):
                  * (L, 3) for torsion angles
                  * (L, 3) bond angles
                  * (L, 6) sidechain angles
        * coords: (L, 3) sidechainnet coords. builds the mask with those instead
                  (better accuracy if modified residues present).
        Outputs:
        * cloud_mask: (L, 14 ) mask of points that should be converted to coords
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, L, 14) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom
    """
    # auto infer device and precision
    precise = angles.dtype if angles is not None else torch.get_default_dtype()
    if device == "auto":
        device = angles.device if angles is not None else device
    if coords is not None:
        cloud_mask = scn_cloud_mask(seq, coords=coords)
    else:
        cloud_mask = scn_cloud_mask(seq)
    cloud_mask = cloud_mask.bool().to(device)
    point_ref_mask = scn_index_mask(seq).long().to(device)
    angles_mask = scn_angle_mask(seq, angles).to(device, precise)
    bond_mask = scn_bond_mask(seq).to(device, precise)
    # return all in a dict
    return {"cloud_mask": cloud_mask,
            "point_ref_mask": point_ref_mask,
            "angles_mask": angles_mask,
            "bond_mask": bond_mask}
def sidechain_fold(wrapper, cloud_mask, point_ref_mask, angles_mask, bond_mask, c_beta=False):
    """ Calcs coords of a protein given it's sequence and internal angles.
        Inputs:
        * wrapper: (L, 14, 3). coords container with backbone ([:, :3]) and optionally
                               c_beta ([:, 4])
        * cloud_mask: (L, 14) mask of points that should be converted to coords
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, 14, L) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom
        * c_beta: whether to place cbeta
        Output: (L, 14, 3) and (L, 14) coordinates and cloud_mask
    """
    # Parallel sidechain - do the oxygen, c-beta and side chain
    for i in range(3, 14):
        # Skip cbeta if arg is set
        if i == 4 and not c_beta:
            continue
        # Prepare inputs
        level_mask = cloud_mask[:, i]
        thetas, dihedrals = angles_mask[:, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i - 3]
        # To place C-beta, we need the carbons from prev res - not available for the 1st res
        if i == 4:
            # The c requested is from the previous residue - offset boolean mask by one
            # can't be done with slicing bc glycines are inside chain (dont have cb)
            coords_a = wrapper[(level_mask.nonzero().view(-1) - 1), idx_a]  # (L-1), 3
            # If first residue is not glycine,
            # for 1st residue, use position of the second residue's N (1,3)
            if level_mask[0].item():
                coords_a[0] = wrapper[1, 1]
        else:
            coords_a = wrapper[level_mask, idx_a]
        wrapper[level_mask, i] = mp_nerf_torch(coords_a,
                                               wrapper[level_mask, idx_b],
                                               wrapper[level_mask, idx_c],
                                               bond_mask[level_mask, i],
                                               thetas, dihedrals)
    return wrapper, cloud_mask
def build_sidechain(batch, pred_backbone_coords, pred_angles_sin_cos, pdb_path=None):

    # Predicted torsion angles from sin cos encoding
    pred_angles = torch.arctan2(pred_angles_sin_cos[..., 0], pred_angles_sin_cos[..., 1]).cpu()  # (B, L, 5)
    B, L = pred_angles.shape[:2]
    device = pred_angles_sin_cos.get_device()
    # Calculate internal angles (3 dihedral + 3 bond angles) from known backbone
    internals = internal_coordinates_torch(pred_backbone_coords, sin_cos=False).cpu()
    all_coords = torch.zeros(B, L, 14, 3)
    all_coords[:, :, 0:3, :] = pred_backbone_coords
    for i, angles in enumerate(pred_angles):
        l = batch["seq_len"][i]  # batch["mask"][i].sum()
        seq = "".join([restype_order_with_x_inverse[aa] for aa in sequence.cpu().numpy()[:l]])
        # Stack in side chainnet order 3 dihedral + 3 bond angles + C-N-CA-CB + chi1, ... chi4. Chi 5 is hardcoded as 0.
        angles = torch.concatenate((internals[i], angles, torch.zeros((internals.size(1), 1))), dim=1)[0:l]
        scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, device="cpu")
        coords, _ = sidechain_fold(wrapper=all_coords[i, :l], **scaffolds, c_beta=True)
        all_coords[i, 0:l] = coords
    """
    if pdb_path is not None:
        i = 0
        l = batch["mask"][i].sum()
        sequence = af_sequences[i]
        seq = "".join([restype_order_with_x_inverse[aa] for aa in sequence.cpu().numpy()[:l]])
        pdb_creator = PdbBuilder(seq, all_coords[i, :l].reshape(l * 14, 3).detach().cpu().numpy())
        pdb_creator.save_pdb(pdb_path + "/debug_3.pdb", "debug")
        np.save(pdb_path + "/pred.npy", all_coords[i, :l].reshape(l * 14, 3).detach().cpu().numpy())
        i = 0
        l = batch["mask"][i].sum()
        sequence = af_sequences[i]
        seq = "".join([restype_order_with_x_inverse[aa] for aa in sequence.cpu().numpy()[:l]])
        pdb_creator = PdbBuilder(seq, batch["full_coords"][i, :l].reshape(l * 14, 3).detach().cpu().numpy())
        pdb_creator.save_pdb(pdb_path + "/debug_gt_3.pdb", "debug")
        np.save(pdb_path + "/gt.npy", batch["full_coords"][i, :l].reshape(l * 14, 3).detach().cpu().numpy())
    """
    return all_coords.to(device)