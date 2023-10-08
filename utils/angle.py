import numpy as np
import prody as pr


import torch
import warnings

from utils.build_info import SC_BUILD_INFO, NUM_ANGLES, NUM_BB_TORSION_ANGLES, NUM_BB_OTHER_ANGLES
from utils.mmcif import ALLOWED_NONSTD_RESIDUES

GLOBAL_PAD_CHAR = 0


def get_bond_angles(res, next_res):
    """Given 2 residues, returns the ncac, cacn, and cnca bond angles between them.
    If any atoms are not present, the corresponding angles are set to the
    GLOBAL_PAD_CHAR. If next_res is None, then only NCAC is measured.
    """
    # First residue angles
    n1, ca1, c1 = tuple(res.select(f"name {a}") for a in ["N", "CA", "C"])

    if n1 and ca1 and c1:
        ncac = safecalcAngle(n1, ca1, c1, radian=True)
    else:
        ncac = GLOBAL_PAD_CHAR

    # Second residue angles
    if next_res is None:
        return ncac, GLOBAL_PAD_CHAR, GLOBAL_PAD_CHAR
    n2, ca2 = (next_res.select(f"name {a}") for a in ["N", "CA"])
    if ca1 and c1 and n2:
        cacn = safecalcAngle(ca1, c1, n2, radian=True)
    else:
        cacn = GLOBAL_PAD_CHAR
    if c1 and n2 and ca2:
        cnca = safecalcAngle(c1, n2, ca2, radian=True)
    else:
        cnca = GLOBAL_PAD_CHAR
    return ncac, cacn, cnca


def get_dihedral(coords1, coords2, coords3, coords4, radian=False):
    """Returns the dihedral angle between four coordinates in degrees.
    Modified from prody.measure.measure to use a numerically safe normalization
    method.
    """
    rad2deg = 180 / np.pi
    eps = 1e-6

    a1 = coords2 - coords1
    a2 = coords3 - coords2
    a3 = coords4 - coords3

    v1 = np.cross(a1, a2)
    v1 = v1 / (v1 * v1).sum(-1) ** 0.5
    v2 = np.cross(a2, a3)
    v2 = v2 / (v2 * v2).sum(-1) ** 0.5
    porm = np.sign((v1 * a3).sum(-1))
    arccos_input_raw = (v1 * v2).sum(-1) / ((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1)) ** 0.5
    if -1 <= arccos_input_raw <= 1:
        arccos_input = arccos_input_raw
    elif arccos_input_raw > 1 and arccos_input_raw - 1 < eps:
        arccos_input = 1
    elif arccos_input_raw < -1 and np.abs(arccos_input_raw) - 1 < eps:
        arccos_input = -1
    else:
        raise ArithmeticError(
            "Input to arccos is outside of acceptable [-1, 1] domain +/- 1e-6.")
    rad = np.arccos(arccos_input)
    if not porm == 0:
        rad = rad * porm
    if radian:
        return rad
    else:
        return rad * rad2deg


def compute_single_dihedral(atoms):
    """Given 4 Atoms, calculate the dihedral angle between them in radians."""
    if None in atoms:
        return GLOBAL_PAD_CHAR
    else:
        atoms = [a.getCoords()[0] for a in atoms]
        return get_dihedral(atoms[0], atoms[1], atoms[2], atoms[3], radian=True)


def measure_phi_psi_omega(residue, include_OXT=False, last_res=False):
    """Measures a residue's primary backbone torsional angles (phi, psi, omega).
    Args:
        residue: ProDy residue object.
        include_OXT: Boolean describing whether or not to measure an angle to
            place the terminal oxygen.
        last_res: Boolean describing if this residue is the last residue in a chain.
            If so, instead of measuring a psi angle, we measure the related torsional
            angle of (N, Ca, C, O).
    Returns:
        Python list of phi, psi, and omega angles for residue.
    """
    try:
        phi = pr.calcPhi(residue, radian=True)
    except ValueError:
        phi = GLOBAL_PAD_CHAR
    try:
        if last_res:
            psi = compute_single_dihedral(
                [residue.select("name " + an) for an in "N CA C O".split()])
        else:
            psi = pr.calcPsi(residue, radian=True)
    except (ValueError, IndexError):
        # For the last residue, we can measure a "psi" angle that is actually
        # the placement of the terminal oxygen. Currently, this is not utilized
        # in the building of structures, but it is included in case the need
        # arises in the future. Otherwise, this would simply become a pad
        # character.
        if include_OXT:
            try:
                psi = compute_single_dihedral(
                    [residue.select("name " + an) for an in "N CA C OXT".split()])
            except ValueError:
                psi = GLOBAL_PAD_CHAR
            except IndexError:
                psi = GLOBAL_PAD_CHAR
        else:
            psi = GLOBAL_PAD_CHAR
    try:
        omega = pr.calcOmega(residue, radian=True)
    except ValueError:
        omega = GLOBAL_PAD_CHAR
    return [phi, psi, omega]


def measure_bond_angles(residue, res_idx, all_res):
    """Given a residue, measure the ncac, cacn, and cnca bond angles."""
    if res_idx == len(all_res) - 1:
        next_res = None
    else:
        next_res = all_res[res_idx + 1]
    return list(get_bond_angles(residue, next_res))


def safecalcAngle(a, b, c, radian):
    """Calculates the angle between 3 coordinates.
    If any of them are missing, the function raises a MissingAtomsError.
    """
    try:
        angle = pr.calcAngle(a, b, c, radian=radian)[0]
    except ValueError as e:
        raise MissingAtomsError from e
    return angle


def compute_sidechain_dihedrals(residue, prev_residue, next_res):
    """Computes all angles to predict for a given residue.
    If the residue is the first in the protein chain, a fictitious C atom is
    placed before the first N. This is used to compute a [ C-1, N, CA, CB]
    dihedral angle. If it is not the first residue in the chain, the previous
    residue's C is used instead. Then, each group of 4 atoms in atom_names is
    used to generate a list of dihedral angles for this residue.
    """
    try:
        res_name = residue.getResname()
        if res_name in list(ALLOWED_NONSTD_RESIDUES.keys()):
            res_name = ALLOWED_NONSTD_RESIDUES[res_name]
        torsion_names = SC_BUILD_INFO[res_name]["torsion-names"]
    except KeyError as e:
        raise ValueError('Non canonical amino acid encountered for which we did not have torsion building information and for which there exists no replacement amino acid')
    if len(torsion_names) == 0:
        return (NUM_ANGLES -
                (NUM_BB_TORSION_ANGLES + NUM_BB_OTHER_ANGLES)) * [GLOBAL_PAD_CHAR]

    # Compute CB dihedral, which may depend on the previous or next residue for placement
    try:
        if prev_residue:
            cb_dihedral = compute_single_dihedral(
                (prev_residue.select("name C"),
                 *(residue.select(f"name {an}") for an in ["N", "CA", "CB"])))
        else:
            cb_dihedral = compute_single_dihedral(
                (next_res.select("name N"),
                 *(residue.select(f"name {an}") for an in ["C", "CA", "CB"])))
    except AttributeError:
        cb_dihedral = GLOBAL_PAD_CHAR

    res_dihedrals = [cb_dihedral]

    for t_name, t_val in zip(torsion_names[1:],
                             SC_BUILD_INFO[res_name]["torsion-vals"][1:]):
        # Only record torsional angles that are relevant (i.e. not planar).
        # All torsion values that vary are marked with 'p' in SC_BUILD_INFO
        if t_val != "p":
            break
        atom_names = t_name.split("-")
        res_dihedrals.append(
            compute_single_dihedral([residue.select("name " + an) for an in atom_names]))

    return res_dihedrals + (NUM_ANGLES - (NUM_BB_TORSION_ANGLES + NUM_BB_OTHER_ANGLES) -
                            len(res_dihedrals)) * [GLOBAL_PAD_CHAR]


def measure_res_coordinates(_res):
    # Given a ProDy residue, measure all relevant coordinates.
    sc_atom_names = determine_sidechain_atomnames(_res)
    bbcoords = get_atom_coords_by_names(_res, ["N", "CA", "C", "O"])
    sccoords = get_atom_coords_by_names(_res, sc_atom_names)
    coord_padding = np.zeros((NUM_COORDS_PER_RES - len(bbcoords) - len(sccoords), 3))
    coord_padding[:] = GLOBAL_PAD_CHAR
    return np.concatenate((np.stack(bbcoords + sccoords), coord_padding))


def determine_sidechain_atomnames(_res):
    # Given a residue from ProDy, returns a list of sidechain atom names that must be recorded.
    if _res.getResname() in SC_BUILD_INFO.keys():
        return SC_BUILD_INFO[_res.getResname()]["atom-names"]
    else:
        raise NonStandardAminoAcidError


def get_atom_coords_by_names(residue, atom_names):
    # Given a ProDy Residue and a list of atom names, this attempts to select and return
    # all the atoms.
    # If atoms are not present, it substitutes the pad character in lieu of their
    # coordinates.

    coords = []
    pad_coord = np.asarray([GLOBAL_PAD_CHAR] * 3)
    for an in atom_names:
        a = residue.select(f"name {an}")
        if a:
            coords.append(a.getCoords()[0])
        else:
            coords.append(pad_coord)
    return coords


def replace_nonstdaas(residues):
    """Replace the non-standard Amino Acids in a list with their equivalents.
    Args:
        residues: List of ProDy residues.
    Returns:
        A list of residues where any non-standard residues have been replaced
        with the canonical version (i.e. MSE -> MET, SEP -> SER, etc.)
        See http://prody.csb.pitt.edu/manual/reference/atomic/flags.html
        for a complete list of non-standard amino acids supported here.
        `XAA` is treated as missing.
    """
    replacements = ALLOWED_NONSTD_RESIDUES
    is_nonstd = []
    resnames = []

    for r in residues:
        rname = r.getResname()
        if rname in replacements.keys():
            r.setResname(replacements[rname])
            is_nonstd.append(1)
        else:
            is_nonstd.append(0)
        resnames.append(rname)

    return residues, resnames, np.asarray(is_nonstd)


def get_seq_coords_and_angles(chain, replace_nonstd=True):
    """Extracts protein sequence, coordinates, and angles from a ProDy chain.
    Args:
        chain: ProDy chain object
    Returns:
        Returns a tuple (angles, coords, sequence) for the protein chain.
        Returns None if the data fails to parse.
        Example angles returned:
            [[phi, psi, omega, ncac, cacn, cnca, chi1, chi2,...chi12],[...] ...]
    """
    chain = chain.select("protein")
    if chain is None:
        raise NoneStructureError
    chain = chain.copy()

    coords = []
    dihedrals = []
    observed_sequence = ""
    all_residues = list(chain.iterResidues())
    unmodified_sequence = [res.getResname() for res in all_residues]
    is_nonstd = np.asarray([0 for res in all_residues])
    if chain.nonstdaa and replace_nonstd:
        all_residues, unmodified_sequence, is_nonstd = replace_nonstdaas(all_residues)
    prev_res = None
    next_res = all_residues[1]

    for res_id, (res, is_modified) in enumerate(zip(all_residues, is_nonstd)):
        if res.getResname() == "XAA":  # Treat unknown amino acid as missing
            continue
        elif not res.stdaa:
            raise NonStandardAminoAcidError

        # Measure basic angles
        bb_angles = measure_phi_psi_omega(res, last_res=res_id == len(all_residues) - 1)
        bond_angles = measure_bond_angles(res, res_id, all_residues)

        # Measure sidechain angles
        all_res_angles = bb_angles + bond_angles + compute_sidechain_dihedrals(
            res, prev_res, next_res)

        # Measure coordinates
        rescoords = measure_res_coordinates(res)

        # Update records
        coords.append(rescoords)
        dihedrals.append(all_res_angles)
        prev_res = res
        observed_sequence += res.getSequence()[0]

    for res_id, (res, is_modified) in enumerate(zip(all_residues, is_nonstd)):
        # Standardized non-standard amino acids
        if is_modified:
            prev_coords = coords[res_id - 1] if res_id > 0 else None
            next_coords = coords[res_id + 1] if res_id + 1 < len(coords) else None
            prev_ang = dihedrals[res_id - 1] if res_id > 0 else None
            res = standardize_residue(res, all_res_angles, prev_coords, next_coords,
                                      prev_ang)

    dihedrals_np = np.asarray(dihedrals)
    coords_np = np.concatenate(coords)

    if coords_np.shape[0] != len(observed_sequence) * NUM_COORDS_PER_RES:
        print(
            f"Coords shape {coords_np.shape} does not match len(seq)*{NUM_COORDS_PER_RES} = "
            f"{len(observed_sequence) * NUM_COORDS_PER_RES},\nOBS: {observed_sequence}\n{chain}"
        )
        raise SequenceError

    return dihedrals_np, coords_np  # observed_sequence, unmodified_sequence, is_nonstd


def angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    p0 : _description_
    p1 : _description_
    p2 : _description_

    Returns
    -------
    _description_

    """
    # Compute vectors
    u1 = p1 - p2
    u2 = p3 - p2

    # Compute angle
    x = np.sum(u1 * u2, axis=1, keepdims=True)
    u1_norm = np.linalg.norm(u1, axis=1, keepdims=True)
    u2_norm = np.linalg.norm(u2, axis=1, keepdims=True)
    return np.arccos(x / (u1_norm * u2_norm)).squeeze(-1)

    # return torch.atan2( torch.norm(torch.cross(u1,u2, dim=-1), dim=-1),
    #                     -(u1*u2).sum(dim=-1) )


def dihedral_angle(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    p0 : _description_
    p1 : _description_
    p2 : _description_
    p3 : _description_

    Returns
    -------
    _description_

    """
    # Compute bond vectors
    u1 = p2 - p1
    u2 = p3 - p2
    u3 = p4 - p3

    # Compute dihedral angle
    b1 = np.cross(u1, u2)
    b2 = np.cross(u2, u3)
    x = np.sum(np.cross(b1, b2) * u2, axis=1, keepdims=True)
    u2_norm = np.linalg.norm(u2, axis=1, keepdims=True)
    y = u2_norm * np.sum(b1 * b2, axis=1, keepdims=True)
    return np.arctan2(x, y).squeeze(-1)


def internal_coordinates(x: np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    coords : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_

    """
    # Compute phi dihedral angle: C - N - CA - C
    phi = dihedral_angle(x[0:-1, 2], x[1:, 0], x[1:, 1], x[1:, 2])

    # Compute psi dihedral angle: N - CA - C - N
    psi = dihedral_angle(x[0:-1, 0], x[0:-1, 1], x[0:-1, 2], x[1:, 0])

    # Compute omega dihedral angle: CA - C - N - CA
    omega = dihedral_angle(x[0:-1, 1], x[0:-1, 2], x[1:, 0], x[1:, 1])

    # Compute bond angle: N - CA - C
    n_ca_c = angle(x[:, 0], x[:, 1], x[:, 2])

    # Compute bond angle: CA - C - N
    ca_c_n = angle(x[:-1, 1], x[:-1, 2], x[1:, 0])

    # Compute bond angle: C - N - CA
    c_n_ca = angle(x[:-1, 2], x[1:, 0], x[1:, 1])

    # Pad missing angles with zeros
    phi = np.concatenate([[0], phi])
    psi = np.concatenate([psi, [0]])
    omega = np.concatenate([omega, [0]])
    ca_c_n = np.concatenate([ca_c_n, [0]])
    c_n_ca = np.concatenate([c_n_ca, [0]])

    # Stack with sidechainnet ordering
    internals = np.stack([phi, psi, omega, n_ca_c, ca_c_n, c_n_ca], axis=1)
    return internals


def angle_torch(p1, p2, p3):
    """_summary_

    Parameters
    ----------
    p1 : (L, 3)
    p2 : _description_
    p3 : _description_
    seq_len :

    Returns
    -------
    _description_

    """

    # Compute vectors
    u1 = p1 - p2
    u2 = p3 - p2

    # Compute angle
    x = torch.sum(u1 * u2, dim=1, keepdim=True)
    u1_norm = torch.linalg.norm(u1, dim=1, keepdim=True)
    u2_norm = torch.linalg.norm(u2, dim=1, keepdim=True)

    with warnings.catch_warnings(record=True) as w:
        angle_r = torch.arccos(x / (u1_norm * u2_norm)).squeeze(-1)
        if len(w) > 0:
            pass

    return angle_r


def dihedral_angle_torch(p1, p2, p3, p4):
    """_summary_

    Parameters
    ----------
    p1 : _description_
    p2 : _description_
    p3 : _description_
    p4 : _description_

    Returns
    -------
    _description_

    """
    # Compute bond vectors
    u1 = p2 - p1
    u2 = p3 - p2
    u3 = p4 - p3

    # Compute dihedral angle
    b1 = torch.linalg.cross(u1, u2)
    b2 = torch.linalg.cross(u2, u3)
    x = torch.sum(torch.linalg.cross(b1, b2) * u2, dim=1, keepdim=True)
    u2_norm = torch.linalg.norm(u2, dim=1, keepdim=True)
    y = u2_norm * torch.sum(b1 * b2, dim=1, keepdim=True)
    return torch.arctan2(x, y).squeeze(-1)


def internal_coordinates_torch(coords, sin_cos=False):
    """_summary_
        coords: (B, L, 3, 3)
        sin_cos: returns as sin_cos encodings if True, as angles if False
    """
    device = coords.get_device()
    B, L = coords.shape[0:2]
    coords = coords.float()

    # We fold the batch into a single B*L sequence and replace the Lth element with zeros for non-existing angles
    # at N and C termini
    x = coords.reshape(B * L, 3, 3)

    # Compute masks
    mask_left = torch.zeros(B * L).bool().reshape(B, L)
    mask_left[:, 0] = 1  # L+1 th element set to 0
    mask_left = mask_left.reshape(B * L)

    mask_right = torch.zeros(B * L).bool().reshape(B, L)
    mask_right[:, -1] = 1  # L th element set to 0
    mask_right = mask_right.reshape(B * L)

    # Compute phi dihedral angle: C - N - CA - C
    phi = dihedral_angle_torch(x[0:-1, 2], x[1:, 0], x[1:, 1], x[1:, 2])
    phi = torch.concatenate([torch.Tensor([0]).to(device), phi])
    phi[mask_left] = 0
    phi = phi.reshape(B, L)[..., None]

    # Compute psi dihedral angle: N - CA - C - N
    psi = dihedral_angle_torch(x[0:-1, 0], x[0:-1, 1], x[0:-1, 2], x[1:, 0])
    psi = torch.concatenate([psi, torch.Tensor([0]).to(device)])
    psi[mask_right] = 0
    psi = psi.reshape(B, L)[..., None]

    # Compute omega dihedral angle: CA - C - N - CA
    omega = dihedral_angle_torch(x[0:-1, 1], x[0:-1, 2], x[1:, 0], x[1:, 1])
    omega = torch.concatenate([omega, torch.Tensor([0]).to(device)])
    omega[mask_right] = 0
    omega = omega.reshape(B, L)[..., None]

    # Compute bond angle: N - CA - C
    n_ca_c = angle_torch(x[:, 0], x[:, 1], x[:, 2]).reshape(B, L)[..., None]

    # Compute bond angle: CA - C - N
    ca_c_n = angle_torch(x[:-1, 1], x[:-1, 2], x[1:, 0])
    ca_c_n = torch.concatenate([ca_c_n, torch.Tensor([0]).to(device)])
    ca_c_n[mask_right] = 0
    ca_c_n = ca_c_n.reshape(B, L)[..., None]

    # Compute bond angle: C - N - CA
    c_n_ca = angle_torch(x[:-1, 2], x[1:, 0], x[1:, 1])
    c_n_ca = torch.concatenate([c_n_ca, torch.Tensor([0]).to(device)])
    c_n_ca[mask_right] = 0
    c_n_ca = c_n_ca.reshape(B, L)[..., None]

    # Stack with sidechainnet ordering
    internals = torch.cat([phi, psi, omega, n_ca_c, ca_c_n, c_n_ca], dim=2)

    if sin_cos:
        angles = torch.zeros(B, L, 6, 2).to(device)
        angles[:, :, :, 0] = torch.sin(internals)  # sin([phi, psi, omega, ncac, cacn, cnca])
        angles[:, :, :, 1] = torch.cos(internals)  # cos([phi, psi, omega, ncac, cacn, cnca])
        return angles  # B, L, 6, 2

    return internals  # (B, L, 6)
