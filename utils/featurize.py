import warnings

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import spatial
from torch_cluster import radius_graph
from torch_geometric.utils import remove_isolated_nodes, subgraph
from torch_scatter import scatter_min, scatter_sum

from utils.angle import measure_phi_psi_omega, measure_bond_angles, compute_sidechain_dihedrals
from utils.logging import lg
from utils.mmcif import RESTYPES, CHI_ANGLES_MASK, ALLOWED_NONSTD_RESIDUES



biopython_parser = PDBParser()
bond_features_list = {
    'bond_type' : ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc'],
    'bond_stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'],
    'is_conjugated': [False, True]
}

atom_features_list = {
    'atomic_num': list(range(1, 75)) + ['misc'], # stop at tungsten
    'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'numring': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'numring': [0, 1, 2, 'misc'],
    'implicit_valence': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'numH': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'numH': [0, 1, 2, 3, 4, 'misc'],
    'number_radical_e': [0, 1, 2, 3, 4, 'misc'],
    'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
    'is_aromatic': [False, True],
    'is_in_ring3': [False, True],
    'is_in_ring4': [False, True],
    'is_in_ring5': [False, True],
    'is_in_ring6': [False, True],
    'is_in_ring7': [False, True],
    'is_in_ring8': [False, True],
    'residues': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                          'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                          'A', 'C', 'G', 'I', 'U', 'DA', 'DC', 'DG', 'DI', 'DU', 'N',
                          'SEC', 'PYL', 'ASX', 'GLX', 'UNK',
                          'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                          'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'residues_canonical': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                          'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'misc'],
    'residues_canonical_1letter': ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
    'atom_types': ['C', 'C*', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1',
                            'CG2', 'CH', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'N*', 'ND', 'ND1', 'ND2', 'NE', 'NE1',
                            'NE2', 'NH', 'NH1', 'NH2', 'NZ', 'O', 'O*', 'OD', 'OD1', 'OD2', 'OE', 'OE1', 'OE2',
                            'OG', 'OG1', 'OH', 'OX', 'OXT', 'S*', 'SD', 'SG', 'misc']
}

def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1
def parse_pdb_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', path)
        rec = structure[0]
    return rec

def featurize_prody(args, prody_struct, lig_pos_list, data, inference=False):
    alpha_pos = []
    n_pos = []
    c_pos = []
    o_pos = []
    res_feats = []
    min_lig_dist_lists = []
    seq = []
    pdb_res_ids = []
    pdb_chain_ids = []
    atom_pos = []
    atom_res_idx = []
    atom_names = []
    atom_elements = []
    angles_list = []
    aatype_num = []
    is_canonical = []
    a_to_num = {restype: i for i, restype in enumerate(RESTYPES)}

    res_count = 0
    total_count = 0
    for chain in prody_struct.iterChains():
        all_residues = list(chain.iterResidues())
        prev_res = None
        for idx, residue in enumerate(all_residues):
            if residue.getResname() not in atom_features_list['residues_canonical'] and residue.getResname() not in list(ALLOWED_NONSTD_RESIDUES.keys()):
                if residue.getResname() != 'HOH':
                    print("WARNING: encountered a non-canonical residue without replacement residue and we are skipping it: {}".format(residue.getResname()))
                continue
            total_count += 1
            alpha_carbon = residue.select('name CA')
            nitrogen = residue.select('name N')
            carbon = residue.select('name C')
            oxygen = residue.select('name O')
            residue_atoms = residue.select('not hydrogen')
            residue_pos = residue_atoms.getCoords()
            residue_atom_names = residue_atoms.getNames()
            residue_elements = residue_atoms.getElements()
            if alpha_carbon is not None and nitrogen is not None and carbon is not None:
                chain_id = ord(residue.getChid()) - ord('A')
                assert chain_id < 26, "ERROR: encountered a chain id that is not a capital letter: {}".format(
                    residue.getChid())
                pdb_chain_ids.append(chain_id)
                atom_names.extend(residue_atom_names)
                atom_elements.extend(residue_elements)
                atom_pos.extend(residue_pos)
                atom_res_idx.extend([res_count] * len(residue_pos))
                alpha_pos.extend(alpha_carbon.getCoords())
                n_pos.extend(nitrogen.getCoords())
                c_pos.extend(carbon.getCoords())
                if oxygen is not None:  # I think sometimes the terminal residue does not have the oxygen
                    o_pos.extend(oxygen.getCoords())
                else:
                    o_pos.extend((nitrogen.getCoords() + alpha_carbon.getCoords() + carbon.getCoords()) / 3)
                res_feats.append([safe_index(atom_features_list['residues_canonical'], residue.getResname())])
                seq.append(residue.getResname())
                min_lig_dist_list = []
                for lig_pos in lig_pos_list:
                    distances = spatial.distance.cdist(lig_pos, residue_pos)
                    min_lig_dist_list.append(distances.min())
                min_lig_dist_lists.append(np.array(min_lig_dist_list))
                pdb_res_ids.append(residue.getResnum())
                if residue.getResname() in list(ALLOWED_NONSTD_RESIDUES.keys()):
                    is_canonical.append(0)
                    aatype_num.append(a_to_num[seq1(ALLOWED_NONSTD_RESIDUES[residue.getResname()])])
                else:
                    is_canonical.append(1)
                    aatype_num.append(a_to_num[seq1(residue.getResname())])
                # get angles
                bb_angles = measure_phi_psi_omega(residue, last_res=idx == len(all_residues) - 1)
                bond_angles = measure_bond_angles(residue, idx, all_residues)
                # Measure sidechain angles
                all_res_angles = bb_angles + bond_angles + compute_sidechain_dihedrals(residue, prev_res, all_residues[1])
                angles_list.append(all_res_angles)

                res_count += 1

    res_feats = np.array(res_feats)  # [n_residues, 1]
    alpha_pos = np.array(alpha_pos)  # [n_residues, 3]
    n_pos = np.array(n_pos)  # [n_residues, 3]
    c_pos = np.array(c_pos)  # [n_residues, 3]
    o_pos = np.array(o_pos)  # [n_residues, 3]
    min_lig_dist_lists = np.array(min_lig_dist_lists)
    pdb_res_ids = np.array(pdb_res_ids)
    pdb_chain_ids = np.array(pdb_chain_ids)
    atom_pos = np.array(atom_pos)
    atom_res_idx = np.array(atom_res_idx)
    atom_names = np.array(atom_names)
    atom_elements = np.array(atom_elements)
    aatype_num = np.array(aatype_num)
    is_canonical = np.array(is_canonical)

    # get angles
    angles = np.zeros((len(angles_list), 12, 2))
    angles[:, :, 0] = np.sin(np.asarray(angles_list))  # [phi, psi, omega, ncac, cacn, cnca, (chi1, chi2, chi3, chi4), ...]
    angles[:, :, 1] = np.cos(np.asarray(angles_list))
    # get angle mask

    chi_angles_mask_modified = np.concatenate((np.ones((21, 6)), CHI_ANGLES_MASK), axis=1)
    chi_angles_mask_modified[-1] = 0
    chi_mask = chi_angles_mask_modified[aatype_num]


    for chain_id in np.unique(pdb_chain_ids):
        chain_res_ids = pdb_res_ids[pdb_chain_ids == chain_id]
        assert np.all(np.diff(chain_res_ids) >= 0), 'ERROR: pdb_res_ids are not sorted within the chain. We need them to be sorted for the get_inter_res_distances calculation and in '
    assert len(alpha_pos) == len(n_pos) == len(angles) == len(c_pos) == len(o_pos) == len(res_feats) == len(min_lig_dist_lists) == len(seq) == len(pdb_res_ids), "SEVERE ERROR: Lengths of protein features are not equal in featurize_protein_model"

    data['protein'].is_canonical = torch.from_numpy(is_canonical).float()
    data['protein'].aatype_num = torch.from_numpy(aatype_num).long()
    data['protein'].angle_mask = torch.from_numpy(chi_mask).float()
    data['protein'].angles = torch.from_numpy(angles).float()
    data['protein'].feat = torch.from_numpy(res_feats).long()
    data['protein'].pos = torch.from_numpy(alpha_pos).float()
    data['protein'].min_lig_dist = torch.from_numpy(min_lig_dist_lists).float()
    data['protein'].pdb_res_id = torch.from_numpy(pdb_res_ids).long()
    data['protein'].atom_names = atom_names
    data['protein'].atom_elements = atom_elements
    data['protein'].pdb_chain_id = torch.from_numpy(pdb_chain_ids).long()
    data['protein'].atom_res_idx = torch.from_numpy(atom_res_idx).long()
    protein_center = torch.mean(data['protein'].pos, dim=0, keepdim=True)
    data['protein'].atom_pos = torch.from_numpy(atom_pos).float() - protein_center
    data['protein'].pos_C = torch.from_numpy(c_pos).float() - protein_center
    data['protein'].pos_O = torch.from_numpy(o_pos).float() - protein_center
    data['protein'].pos_N = torch.from_numpy(n_pos).float() - protein_center
    data['protein'].pos -= protein_center

    data.original_center = protein_center
    data['protein', 'radius_graph', 'protein'].edge_index = radius_graph(data['protein'].pos, r=args.protein_radius, max_num_neighbors=1000)

    b = data['protein'].pos - data['protein'].pos_N
    c = data['protein'].pos_C - data['protein'].pos
    a = torch.cross(b, c, dim=-1)
    data['protein'].pos_Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + data['protein'].pos

    # this might take quite some time
    if not inference:
        get_inter_res_distances(data)

    num_residues = len(c_pos)
    if num_residues <= 1: raise ValueError(f"rec contains only 1 residue!")

def get_protein_subgraph(data, mask, has_designable_mask=True):
    try:
        subset = torch.where(mask)[0]
        atom_mask = mask[data['protein'].atom_res_idx]
        data['protein'].atom_pos = data['protein'].atom_pos[atom_mask]
        if hasattr(data['protein'], 'atom_names'): data['protein'].atom_names = data['protein'].atom_names[atom_mask]
        if hasattr(data['protein'], 'atome_elements'): data['protein'].atom_elements = data['protein'].atom_elements[atom_mask]

        residue_sizes = scatter_sum(torch.ones_like(data['protein'].atom_res_idx), data['protein'].atom_res_idx)
        residue_sizes = residue_sizes[subset]
        debug = data['protein'].atom_res_idx[atom_mask]
        data['protein'].atom_res_idx = torch.arange(len(residue_sizes)).repeat_interleave(residue_sizes)
        assert torch.all(torch.bincount(debug)[torch.nonzero(torch.bincount(debug))].squeeze() == torch.bincount(data['protein'].atom_res_idx)), 'you made a mistake in the construction of the atom to residue index mapping when taking a subgraph'

        data['protein'].is_canonical = data['protein'].is_canonical[subset]
        if hasattr(data['protein'], 'inter_res_dist'): data['protein'].inter_res_dist = data['protein'].inter_res_dist[subset][:, subset]
        data['protein'].pdb_res_id = data['protein'].pdb_res_id[subset]
        data['protein'].pdb_chain_id = data['protein'].pdb_chain_id[subset]
        if has_designable_mask:  data['protein'].designable_mask = data['protein'].designable_mask[subset]
        data['protein'].min_lig_dist = data['protein'].min_lig_dist[subset]
        data['protein'].aatype_num = data['protein'].aatype_num[subset]
        data['protein'].angle_mask = data['protein'].angle_mask[subset]
        data['protein'].angles = data['protein'].angles[subset]
        data['protein'].pos = data['protein'].pos[subset]
        data['protein'].pos_Cb = data['protein'].pos_Cb[subset]
        data['protein'].pos_C = data['protein'].pos_C[subset]
        data['protein'].pos_O = data['protein'].pos_O[subset]
        data['protein'].pos_N = data['protein'].pos_N[subset]
        data['protein'].feat = data['protein'].feat[subset]
        data['protein', 'radius_graph', 'protein'].edge_index, _, mask = remove_isolated_nodes(subgraph(subset, data['protein', 'radius_graph', 'protein'].edge_index, num_nodes=len(mask))[0])
    except Exception as e:
        lg(f'Error in get_protein_subgraph for: {data.pdb_id}')
        try:
            lg(f'For {data.pdb_id} the fake_lig_id was {data["protein"].fake_lig_id} and the pdb_res_id was {data["protein"].pdb_res_id[data["protein"].fake_lig_id]} and the pdb_chain_id was {data["protein"].pdb_chain_id[data["protein"].fake_lig_id]}')
        except:
            pass
        lg(str(e))
        raise e
    return data

def get_inter_res_distances(data):
    atom_res_idx = data['protein'].atom_res_idx
    atom_pos = data['protein'].atom_pos

    inter_res_dist = []
    for i in range(len(data['protein'].pos)):
        atom_mask = atom_res_idx == i
        res_pos = atom_pos[atom_mask][1:] # exclude the first atom, which is the nitrogen
        atom_min_dist = torch.cdist(res_pos, atom_pos).min(dim=0).values
        res_min_dist = scatter_min(atom_min_dist, atom_res_idx)[0]
        inter_res_dist.append(res_min_dist)
    data['protein'].inter_res_dist = torch.stack(inter_res_dist)


def get_feature_dims():
    node_feature_dims = [
        len(atom_features_list['atomic_num']),
        len(atom_features_list['chirality']),
        len(atom_features_list['degree']),
        len(atom_features_list['numring']),
        len(atom_features_list['implicit_valence']),
        len(atom_features_list['formal_charge']),
        len(atom_features_list['numH']),
        len(atom_features_list['hybridization']),
        len(atom_features_list['is_aromatic']),
        len(atom_features_list['is_in_ring5']),
        len(atom_features_list['is_in_ring6'])
    ]
    edge_attribute_dims = [
        len(bond_features_list['bond_type']),
        len(bond_features_list['bond_stereo']),
        len(bond_features_list['is_conjugated']),
        1 # for the distance edge attributes after we combine the distance and bond edges
    ]
    bond_attribute_dims = [
        len(bond_features_list['bond_type']),
        len(bond_features_list['bond_stereo']),
        len(bond_features_list['is_conjugated']),
    ]
    rec_feature_dims = [
        len(atom_features_list['residues_canonical'])
    ]
    return node_feature_dims, edge_attribute_dims,bond_attribute_dims, rec_feature_dims


def featurize_atoms(mol):
    atom_features = []
    ringinfo = mol.GetRingInfo()
    def safe_index_(key, val):
        return safe_index(atom_features_list[key], val)
    for idx, atom in enumerate(mol.GetAtoms()):
        features = [
            safe_index_('atomic_num', atom.GetAtomicNum()),
            safe_index_('chirality', str(atom.GetChiralTag())),
            safe_index_('degree', atom.GetTotalDegree()),
            safe_index_('numring', ringinfo.NumAtomRings(idx)),
            safe_index_('implicit_valence', atom.GetImplicitValence()),
            safe_index_('formal_charge', atom.GetFormalCharge()),
            safe_index_('numH', atom.GetTotalNumHs()),
            safe_index_('hybridization', str(atom.GetHybridization())),
            safe_index_('is_aromatic', atom.GetIsAromatic()),
            safe_index_('is_in_ring5', ringinfo.IsAtomInRingOfSize(idx, 5)),
            safe_index_('is_in_ring6', ringinfo.IsAtomInRingOfSize(idx, 6))        ]
        atom_features.append(features)

    return torch.tensor(atom_features)

def featurize_bond(bond):
    bond_feature = [
        safe_index(bond_features_list['bond_type'], str(bond.GetBondType())),
        safe_index(bond_features_list['bond_stereo'], str(bond.GetStereo())),
        safe_index(bond_features_list['is_conjugated'], bond.GetIsConjugated())
    ]
    return bond_feature


def init_lig_graph(args,lig, data):
    lig_pos = torch.from_numpy(lig.GetConformer().GetPositions()).float()
    atom_feats = featurize_atoms(lig)

    bond_idx, bond_attr = get_bond_edges(lig)
    data['ligand'].feat = atom_feats
    data['ligand'].pos = lig_pos
    data['ligand'].size = len(lig_pos)
    data['ligand', 'bond_edge', 'ligand'].edge_index = bond_idx
    data['ligand', 'bond_edge', 'ligand'].edge_attr = bond_attr


def get_ligand_distance_edges(args, complex_graph):
    distance_idx = radius_graph(complex_graph['ligand'].pos, r=args.lig_radius, max_num_neighbors=1000)
    distance_attr = torch.zeros((distance_idx.shape[1], 1), device=complex_graph['ligand'].pos.device)
    bond_idx = complex_graph['ligand', 'bond_edge', 'ligand'].edge_index
    bond_attr = complex_graph['ligand', 'bond_edge', 'ligand'].edge_attr

    edge_index, edge_attr = combine_edges((bond_idx, bond_attr), (distance_idx, distance_attr))
    complex_graph['ligand', 'combined_edge', 'ligand'].edge_index = edge_index
    complex_graph['ligand', 'combined_edge', 'ligand'].edge_attr = edge_attr

def get_bond_edges(mol):
    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_attr += [featurize_bond(bond)]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr)
    edge_attr = torch.concat([edge_attr, edge_attr], 0)

    return edge_index, edge_attr.type(torch.uint8)
def combine_edges(*edges):
    idxs, feats = zip(*edges)
    idx = torch.cat(idxs, -1)
    feat_dim = sum(f.shape[-1] for f in feats)
    feat = torch.zeros((idx.shape[-1], feat_dim), dtype=torch.float32, device=idx.device)
    r, c = 0, 0
    for f in feats:
        rr, cc = f.shape
        feat[r:r+rr,c:c+cc] = f
        r+=rr; c+=cc;
    return idx, feat

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol

def generate_conformer(mol):
    ps = AllChem.ETKDGv2()
    failures, id = 0, -1
    while failures < 5 and id == -1:
        if failures > 0:
            print(f'rdkit coords could not be generated. trying again {failures}.')
        id = AllChem.EmbedMolecule(mol, ps)
        failures += 1
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        id = AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
        return id
    return id
