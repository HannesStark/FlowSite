import os
import copy
import re
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import prody
import scipy
import torch
from Bio.SeqUtils import seq1

from rdkit import Chem
from rdkit.Chem import RemoveHs
from rdkit.Geometry import Point3D
from scipy.spatial.transform import Rotation
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm

from datasets.complex_dataset import ComplexDataset
from utils.featurize import read_molecule, featurize_prody, init_lig_graph, atom_features_list, bond_features_list, \
    get_protein_subgraph, generate_conformer
from utils.logging import lg
from utils.mmcif import RESTYPES
from utils.residue_constants import amino_acid_atom_names
from utils.visualize import plot_point_cloud


class EmptyPocketException(Exception):
    pass

def expand_range(r):
    start, end = map(int, r.split('-'))
    return list(range(start, end+1))

class InferenceDataset(Dataset):
    def __init__(self, args, device='cpu'):
        super(InferenceDataset, self).__init__()
        self.args = args
        self.device = device
        if self.args.csv_file is not None:
            self.df = pd.read_csv(self.args.csv_file)
        else:
            self.df = pd.DataFrame({'ligand': self.args.ligand,
                                    'smiles': self.args.smiles,
                                    'protein': self.args.protein,
                                    'design_residues': self.args.design_residues,
                                    'pocket_def_ligand': self.args.pocket_def_ligand,
                                    'pocket_def_residues':self.args.pocket_def_residues,
                                    'pocket_def_center': self.args.pocket_def_center}, index=[0])

        self.data_dict = {}
        self.pocket_type = [] # 0 for distance pocket and 1 for radius pocket
        for idx, row in self.df.iterrows():
            assert os.path.exists(row['protein'])
            assert (pd.isna(row['pocket_def_ligand']) + pd.isna(row['pocket_def_residues']) + pd.isna(row['pocket_def_center'])) == 2, f'Exactly one of pocket_def_ligand, pocket_def_residues, pocket_def_center must be specified. This was not the case for the {idx} row in your csv or your command line input.'
            assert (pd.isna(row['ligand']) + pd.isna(row['smiles'])) == 1, f'Ligand must be either defined via a path to a ligand file or a smiles string. This was not the case for the {idx} row in your csv or your command line input.'
            if not pd.isna(row['pocket_def_ligand']):
                assert os.path.exists(row['pocket_def_ligand'])
                self.pocket_type.append(0)
            else:
                self.pocket_type.append(1)

    def len(self):
        return len(self.df)
    def get(self, idx):
        if idx in list(self.data_dict.keys()):
            return copy.deepcopy(self.data_dict[idx])
        row = self.df.iloc[idx]
        prody_struct = prody.parsePDB(row['protein'])
        data = HeteroData()
        data['protein'].name = os.path.basename(row['protein']).split('.')[0]

        if not pd.isna(row['smiles']):
            lig = Chem.MolFromSmiles(row['smiles'])
            data['ligand'].name = row['smiles']
            signal = generate_conformer(lig)
            if signal == -1: raise Exception(f'Could not generate conformer for {row["smiles"]} for the {idx} row in your csv or your command line input.')
        else:
            lig = read_molecule(row['ligand'])
            data['ligand'].name = os.path.basename(row['ligand']).split('.')[0]
        data['pdb_id'] = f'complexid{idx}'
        lig = RemoveHs(lig)
        data['ligand'].mol = copy.deepcopy(lig)
        init_lig_graph(self.args, lig, data)


        if not pd.isna(row['pocket_def_ligand']):
            pocket_def_lig = read_molecule(row['pocket_def_ligand'])
            pocket_def_lig = RemoveHs(pocket_def_lig)
            pocket_def_pos = torch.from_numpy(pocket_def_lig.GetConformer().GetPositions()).float()


        featurize_prody(self.args, prody_struct, [data['ligand'].pos.numpy()], data, inference=True)
        data['ligand'].pos -= data.original_center
        if not pd.isna(row['pocket_def_ligand']): pocket_def_pos -= data.original_center

        data['protein'].designable_mask = self.string_to_mask(row['design_residues'], data)


        for key, value in data['protein'].items():
            data['full_protein'][key] = copy.deepcopy(value)
        if not pd.isna(row['pocket_def_ligand']):
            assert self.args.pocket_residue_cutoff is not None, 'distance pocket requires a pocket_resiudue_cutoff'
            ca_distances = torch.cdist(data['protein'].pos, pocket_def_pos).min(dim=1)[0]
            pocket_mask = ca_distances < self.args.pocket_residue_cutoff
            assert pocket_mask.sum() > 0, f'No pocket residues found for {data.pdb_id} with pocket_residue_cutoff {self.args.pocket_residue_cutoff}'
            data = get_protein_subgraph(data, pocket_mask)
            pocket_center_mask = ca_distances[pocket_mask] < 8
            if pocket_center_mask.sum() == 0:
                print(f'warning : No residues found for {data.pdb_id} with a CA within 8A of the pocket_def_ligand for constructing the pocket center. Using the closest')
                pocket_center_mask = torch.zeros_like(ca_distances[pocket_mask]).bool()
                pocket_center_mask[ca_distances[pocket_mask].argmin()] = True
            assert pocket_center_mask.sum() > 0, f'No residues found for {data.pdb_id} with a CA within   8A for constructing the pocket center'
            pocket_center = torch.mean(data['protein'].pos[pocket_center_mask], dim=0, keepdim=True)
        else:
            if not pd.isna(row['pocket_def_residues']):
                pocket_center = data['protein'].pos[self.string_to_mask(row['pocket_def_residues'], data)].mean(dim=0, keepdim=True)
            elif not pd.isna(row['pocket_def_center']):
                pocket_center = torch.tensor([float(s) for s in re.findall(r'[-+]?\d*\.\d+|\d+', row['pocket_def_center'])]).float()[None, :]
            radius = max(5, 0.5 * (torch.cdist(data['ligand'].pos, data['ligand'].pos).max())) + self.args.radius_pocket_buffer
            center_distances = ((data['protein'].pos - pocket_center) ** 2).sum(-1).sqrt()
            pocket_mask = center_distances < radius
            if pocket_mask.sum() == 0: raise EmptyPocketException(f'No pocket residues found for {data.pdb_id} with radius {radius} from radius_pocket_buffer {self.args.radius_pocket_buffer}')
            assert pocket_mask.sum() > 0, f'No pocket residues found for {data.pdb_id} with radius {radius} from radius_pocket_buffer {self.args.radius_pocket_buffer}'
            data = get_protein_subgraph(data, pocket_mask)
        data['full_protein'].pocket_mask = pocket_mask

        # Center positions around pocket center
        data.pocket_center = pocket_center
        data['protein'].atom_pos -= data.pocket_center
        data['protein'].pos -= data.pocket_center
        data['protein'].pos_Cb -= data.pocket_center
        data['protein'].pos_C -= data.pocket_center
        data['protein'].pos_O -= data.pocket_center
        data['protein'].pos_N -= data.pocket_center
        data['ligand'].pos -= data.pocket_center

        if self.args.self_condition_inv:
            data['protein'].input_feat = torch.zeros_like(data['protein'].feat) + len(atom_features_list['residues_canonical']) # mask out all the residues
        # plot_point_cloud([data['protein'].pos, data['ligand'].pos, data['full_protein'].pos, data['full_protein'].pos[data['full_protein'].designable_mask], pocket_center])
        data.protein_sigma = (torch.square(data["protein"].pos).mean() ** 0.5)
        self.data_dict[idx] = copy.deepcopy(data)
        return data

    def string_to_mask(self, residue_specification, data):
        chain_matches = re.findall(r'[A-Za-z]+', residue_specification)
        matches = re.findall(r'(\d+-\d+|\d+)', residue_specification)
        chain_ids = []
        res_ids = []
        for match, chain_id in zip(matches, chain_matches):
            if '-' in match:
                expanded = expand_range(match)
                res_ids.extend(expanded)
                chain_ids.extend([ord(chain_id) - ord('A')] * len(expanded))
            else:
                res_ids.append(int(match))
                chain_ids.append(ord(chain_id) - ord('A'))
        chain_mask = torch.isin(data['protein'].pdb_chain_id, torch.tensor(chain_ids))
        res_mask = torch.isin(data['protein'].pdb_res_id, torch.tensor(res_ids))
        return torch.logical_and(chain_mask, res_mask)
