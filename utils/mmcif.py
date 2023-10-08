import numpy as np

import prody as pr
import torch

# Order matters! Do not change indeces
RESTYPES = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
        'S', 'T', 'W', 'Y', 'V'
    ]

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
CHI_ANGLES_MASK = [
    [1.0, 0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 1.0, 0.0],  # GLU
    [1.0, 0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 1.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0, 0.0],  # UNK
]

CHI_PI_PERIODIC = [
    [0.0, 0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0, 0.0],  # UNK
]
chi_pi_periodic_torch = torch.cat((torch.zeros(21, 6), torch.tensor(CHI_PI_PERIODIC)), dim=1)

ALLOWED_NONSTD_RESIDUES = {
    "ASX": "ASP",
    "GLX": "GLU",
    "CSO": "CYS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "MSE": "MET",
    "SEC": "CYS",
    "SEP": "SER",
    "TPO": "THR",
    "PTR": "TYR",
    "XLE": "LEU",
    "4FB": "PRO",
    "MLY": "LYS",  # N-dimethyl-lysine
    "AIB": "ALA",  # alpha-methyl-alanine, not included during generation on 1/23/22
    "MK8": "MET"   # 2-methyl-L-norleucine, added 3/16/21
}

def read_angles_coords(path: str, mmcif=True):
    if mmcif:
        ag = pr.parseMMCIF(path)
    else:
        ag = pr.parsePDB(path)
    for chain in ag.iterChains():
        angles_tmp, full_coords = get_seq_coords_and_angles(chain)
        L = len(angles_tmp)
        angles = np.zeros((L, 12, 2))

        angles[:, :, 0] = np.sin(angles_tmp)  # [phi, psi, omega, ncac, cacn, cnca, (chi1, chi2, chi3, chi4), ...]
        angles[:, :, 1] = np.cos(angles_tmp)

        full_coords = full_coords.reshape(L, 14, 3)

        return angles, full_coords


def load_mmcif(path, chain=None):
    block = cif.read(path).sole_block()
    atoms = block.find(
        "_atom_site.",
        [
            "label_atom_id",
            "label_comp_id",
            "label_seq_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "B_iso_or_equiv",
            "auth_asym_id",
        ],
    )

    # Filter by chain
    if chain is not None:
        atoms = [atom for atom in atoms if atom[7] == chain]

    # Filter to protein residues
    atoms = [atom for atom in atoms if atom[1] in AA_CODE]

    # Get coords
    a_map = {(int(a[2]), a[0]): a for a in atoms}
    c_map = {k: [float(v[3]), float(v[4]), float(v[5])] for k, v in a_map.items()}
    res_indices = [int(atom[2]) for atom in atoms]
    res_indices = sorted(set(res_indices))
    coords = np.array(
        [
            [c_map[(idx, "N")], c_map[(idx, "CA")], c_map[(idx, "C")]]
            for idx in res_indices
        ]
    )

    # Get sequence
    sequence = "".join([AA_CODE[a_map[(i, "CA")][1]] for i in res_indices])
    plddt = np.array([float(a_map[(i, "CA")][6]) / 100 for i in res_indices])
    num_atoms_mask = [num_atoms[x] for x in sequence]

    # Compute the chi angle mask
    a_to_num = {restype: i for i, restype in enumerate(RESTYPES)}
    aatype = np.array([a_to_num[x] for x in sequence])

    chi_angles_mask_modified = np.concatenate((np.ones((21, 6)), CHI_ANGLES_MASK), axis=1)
    chi_angles_mask_modified[-1] = 0

    chi_mask = chi_angles_mask_modified[aatype]

    return coords, sequence, plddt, num_atoms_mask, chi_mask, aatype, res_indices

