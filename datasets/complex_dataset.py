import os
import copy
import time
import traceback
from datetime import datetime

import numpy as np
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

from utils.featurize import read_molecule, featurize_prody, init_lig_graph, atom_features_list, bond_features_list, \
    get_protein_subgraph
from utils.logging import lg
from utils.residue_constants import amino_acid_atom_names

class ComplexDataset(Dataset):
    def __init__(self, args, split_path, multiplicity = 1):
        super(ComplexDataset, self).__init__()
        self.args = args
        self.cache_file_name = f'idxFile{os.path.splitext(os.path.basename(split_path))[0]}--protFile{args.protein_file_name}--ligConRad{args.lig_connection_radius}{"--newData"}{"--Angles"}{"--correctLigSel" if args.correct_moad_lig_selection else ""}'
        self.full_cache_path = os.path.join(args.cache_path, self.cache_file_name)
        os.makedirs(self.full_cache_path, exist_ok=True)
        pdb_ids = np.loadtxt(split_path, dtype=str)
        shuffled_pdb_ids = pdb_ids[np.random.permutation(len(pdb_ids))]

        valid_complex_paths_path = os.path.join(self.full_cache_path,'valid_complex_paths.txt')
        lig_meta_data_path = os.path.join(self.full_cache_path,'lig_meta_data.npz')
        if args.await_preprocessing:
            while not os.path.exists(valid_complex_paths_path):
                lg(f'Waiting for {valid_complex_paths_path} to be created')
                time.sleep(10)
        if args.delte_unreadable_cache:
            for path in tqdm(np.loadtxt(valid_complex_paths_path, dtype=str), desc="Checking for corrupted files"):
                try:
                    data = torch.load(path)
                except Exception as e:
                    if "StreamReader failed reading f" in str(e):
                        print('removing corrupted file:', path)
                        os.remove(path)
                    elif "No such file or directory:":
                        print('found corrupted file but it was already taken care of')
                    else:
                        raise e
            if os.path.exists(valid_complex_paths_path):
                os.remove(valid_complex_paths_path)
        if not os.path.exists(valid_complex_paths_path):
            for pdb_id in tqdm(shuffled_pdb_ids, desc="Preprocessing"):
                if os.path.exists(os.path.join(self.full_cache_path, pdb_id + "none.pt")):
                    continue
                if not os.path.exists(os.path.join(self.full_cache_path, pdb_id + ".pt")):
                    self.preprocess(pdb_id)
                else:
                    lg(f'The file for {pdb_id} already exists: {os.path.join(self.full_cache_path, pdb_id + ".pt")}' )
            valid_paths = []
            lig_sizes = []
            num_contacts = []
            for pdb_id in tqdm(shuffled_pdb_ids, desc="checking for valid complexes"):
                if os.path.exists(os.path.join(self.full_cache_path, pdb_id + ".pt")):
                    valid_paths.append(os.path.join(self.full_cache_path, pdb_id + ".pt"))
                    npz_file = np.load(os.path.join(self.full_cache_path, pdb_id + "lig_meta_data.npz"))
                    lig_sizes.append(npz_file['arr_0'])
                    num_contacts.append(npz_file['arr_1'])
            np.savez(os.path.join(lig_meta_data_path), np.array(lig_sizes,dtype=object), np.array(num_contacts,dtype=object), allow_pickle=True)
            np.savetxt(os.path.join(valid_complex_paths_path), valid_paths, fmt="%s")

        lg('Loading valid complex path names')
        valid_complex_paths = np.loadtxt(valid_complex_paths_path, dtype=str)
        npz_file = np.load(lig_meta_data_path, allow_pickle=True)
        lig_sizes, num_contacts = npz_file['arr_0'], npz_file['arr_1']
        assert len(valid_complex_paths) == len(lig_sizes)
        assert len(valid_complex_paths) > 0
        lg(f'Finished loading combined data of length: {len(valid_complex_paths)}')
        if args.biounit1_only and args.data_source == 'moad':
            filter_mask = [idx for idx, (path, size) in enumerate(zip(valid_complex_paths, lig_sizes)) if 'unit1' in path]
            filtered_paths, filtered_sizes, filtered_contacts = valid_complex_paths[filter_mask], lig_sizes[filter_mask], num_contacts[filter_mask]
            lg(f'Finished filtering for biounit1 to remain with: {len(filtered_paths)}')
        else:
            filtered_paths, filtered_sizes, filtered_contacts = valid_complex_paths, lig_sizes, num_contacts

        valid_ids = []
        for idx in range(len(filtered_paths)):
            if np.any((filtered_contacts[idx] >= args.min_num_contacts) & (filtered_sizes[idx] <= args.max_lig_size) & (filtered_sizes[idx] >= args.min_lig_size)):
                valid_ids.append(idx)
        filtered_paths, filtered_sizes, filtered_contacts = filtered_paths[valid_ids], filtered_sizes[valid_ids], filtered_contacts[valid_ids]
        lg(f'Finished filtering combined data for ligands of min size {args.min_lig_size} and max size {args.max_lig_size} and min_num_contacts {args.min_num_contacts} to end up with this many: {len(filtered_paths)}')
        self.data_paths = [path for path, size in zip(filtered_paths, filtered_sizes) if any(size <= args.max_lig_size)]
        lg(f'Finished filtering combined data for ligands of max size {args.max_lig_size} to end up with this many: {len(self.data_paths)}')
        if multiplicity > 1:
            self.data_paths = self.data_paths * multiplicity

        self.fake_lig_ratio = 0
        self.data_dict = {}

    def len(self):
        return len(self.data_paths)

    def get_data(self,idx):
        if idx in list(self.data_dict.keys()) and not self.args.dont_cache_dataset:
            return self.data_dict[idx]
        else:
            try:
                data = torch.load(self.data_paths[idx])
            except RuntimeError as e:
                if "ytorchStreamReader failed reading f" in str(e):
                    print('AAaa')
                    raise e
                else:
                    raise e
            if not self.args.dont_cache_dataset: self.data_dict[idx] = data
            return data

    def get(self, idx):
        if self.args.overfit_lig_idx is not None:
            idx = 0
        data = copy.deepcopy(self.get_data(idx))

        if self.fake_lig_ratio > 0 and np.random.rand() < self.fake_lig_ratio:
            del data['ligand']
            try:
                fake_lig_id = self.get_fake_lig_id(data, idx)
                if fake_lig_id is None: # no fake ligand found
                    lg(f'No valid sidechains for fake ligands in {data.pdb_id}. Trying a new random complex instead now.')
                    return self.get(torch.randint(low=0, high=self.len(), size=(1,)).item())
            except Exception as e:
                lg(f'ERROR: Failed to get fake_lig_id for {data.pdb_id} due to an error in get_fake_lig_id. Trying a new random complex instead now.')
                lg(e)
                traceback.logger.info_exc()
                return self.get(torch.randint(low=0, high=self.len(), size=(1,)).item())

            try:
                data['protein'].fake_lig_id = fake_lig_id
                data['protein'].min_lig_dist = data['protein'].inter_res_dist[fake_lig_id]
                init_success = self.init_fake_lig(data, fake_lig_id)
                data['ligand'].ccd_ids = np.array(['XXX'], dtype='<U3')
                data['ligand'].num_components = np.array([1])
                data['ligand'].name = 'fake_lig_' + data['ligand'].fake_lig_type
                if not init_success:
                    lg(f'Could not initialize fake ligand with fake_lig_id {fake_lig_id} for {data.pdb_id}. Trying a new random complex instead now.')
                    return self.get(torch.randint(low=0, high=self.len(), size=(1,)).item())
                # remove residue of fake ligand and residues that should be masked from protein graph
                arange = torch.arange(len(data['protein'].pos))
                valid_mask = (fake_lig_id - self.args.num_chain_masks > arange) | (arange > fake_lig_id + self.args.num_chain_masks) | (data['protein'].pdb_chain_id != data['protein'].pdb_chain_id[fake_lig_id]) | (data['protein'].inter_res_dist[fake_lig_id] > self.args.min_chain_mask_dist)
                data = get_protein_subgraph(data, valid_mask, has_designable_mask=False)
                data.num_ligs = valid_mask.sum()
                data['ligand'].lig_choice_id = torch.tensor(0)
            except Exception as e:
                lg(f'ERROR: when initializing fake ligand for {data.pdb_id} with fake_lig_id {fake_lig_id} and pdb_chain_id {data["protein"].pdb_chain_id[fake_lig_id]} and pdb_res_id {data["protein"].pdb_res_id[fake_lig_id]}. Trying a new random complex instead now.')
                lg(e)
                traceback.logger.info_exc()
                return self.get(torch.randint(low=0, high=self.len(), size=(1,)).item())
        else:
            # right now data['ligand'] is a list of multiple ligands that bind to the protein. We randomly choose one. When using PDBBind there is only one option. The option --use_largest_lig will always choose the largest ligand satisfying the num_contacts and ligand size requriements.
            # also, the data['protein'].min_lig_dist is a len_protein x num_ligands matrix. One column for each ligand and we choose the corresponding one.
            lig_sizes = np.array([lig_data['ligand'].size for lig_data in data['ligand']])
            res_in_contact = data['protein'].min_lig_dist < 4
            num_contacts = res_in_contact.sum(dim=0)
            valid_ids = np.where((num_contacts >= self.args.min_num_contacts) & (lig_sizes <= self.args.max_lig_size) & (lig_sizes >= self.args.min_lig_size))[0]

            if len(valid_ids) == 0: # no ligands that are small enough and close enough to the protein
                lg(f'Warning, no ligands of maximum size {self.args.max_lig_size} had min_num_contacts {self.args.min_num_contacts}. The number of contacts were {num_contacts} and sizes {[lig_data["ligand"].size for lig_data in data["ligand"]]}. This is in {data.pdb_id}. Sampling a new complex.')
                with open(f"{os.environ['MODEL_DIR']}/anomalous_complexes.txt", "a") as f:
                    f.write(f'Warning, no ligands of maximum size {self.args.max_lig_size} had min_num_contacts {self.args.min_num_contacts}. The number of contacts were {num_contacts} and sizes {[lig_data["ligand"].size for lig_data in data["ligand"]]}. This is in {data.pdb_id}. Sampling a new complex.\n')
                return self.get(torch.randint(low=0, high=self.len(), size=(1,)).item())
            if self.args.overfit_lig_idx is not None:
                valid_ids = [self.args.overfit_lig_idx]
            lig_choice_id = torch.tensor(np.random.choice(valid_ids))
            if self.args.use_largest_lig:
                lig_choice_id = torch.tensor(valid_ids[np.argmax(lig_sizes[valid_ids])])
            assert not torch.isnan(data['protein'].pos).any()
            lig_choice = copy.deepcopy(data['ligand'][lig_choice_id])
            del data['ligand']
            data = self.update_data(data, lig_choice)
            data['protein'].min_lig_dist = data['protein'].min_lig_dist[:, lig_choice_id]
            data['protein'].fake_lig_id = -1
            data['ligand'].lig_choice_id = lig_choice_id
            data['ligand'].fake_lig_type = 'None'
            data['ligand'].rdkit_lig = "None"
            data.num_ligs = torch.tensor(len(valid_ids))

        # get designable mask on designable residues
        data['protein'].designable_mask = torch.zeros_like(data['protein'].min_lig_dist).bool()
        data['protein'].designable_mask[data['protein'].min_lig_dist < self.args.design_residue_cutoff] = True


        if self.args.diffdock_pocket:
            data = self.get_diffdock_pocket(data)
        self.center_data(data)


        if self.args.pocket_residue_cutoff is not None:
            data = get_protein_subgraph(data, data['protein'].min_lig_dist < self.args.pocket_residue_cutoff)
        if self.args.self_condition_inv:
            data['protein'].input_feat = torch.zeros_like(data['protein'].feat) + len(atom_features_list['residues_canonical']) # mask out all the residues
        if self.args.tfn_use_aa_identities:
            data['protein'].input_feat = data['protein'].feat.clone()
        data.protein_sigma = (torch.square(data["protein"].pos).mean() ** 0.5)
        if self.args.mask_lig_translation:
            data['ligand'].pos -= data['ligand'].pos.mean(dim=0, keepdim=True)
            data['ligand'].pos += data['protein'].pos.mean(dim=0, keepdim=True)
        if self.args.mask_lig_rotation:
            temp_lig_pos = data['ligand'].pos - data['ligand'].pos.mean(dim=0, keepdim=True)
            temp_lig_pos = temp_lig_pos @ torch.from_numpy(Rotation.random().as_matrix()).float()
            temp_lig_pos += data['ligand'].pos.mean(dim=0, keepdim=True)
            data['ligand'].pos = temp_lig_pos
        if self.args.mask_lig_pos:
            center = data['ligand'].pos.mean(dim=0)
            data['ligand'].pos = torch.randn_like(data['ligand'].pos)
            data['ligand'].pos += center
        if self.args.backbone_noise > 0:
            data['protein'].pos += torch.randn_like(data['protein'].pos) * self.args.backbone_noise
            data['protein'].pos_N += torch.randn_like(data['protein'].pos_N) * self.args.backbone_noise
            data['protein'].pos_O += torch.randn_like(data['protein'].pos_O) * self.args.backbone_noise
            data['protein'].pos_C += torch.randn_like(data['protein'].pos_C) * self.args.backbone_noise
            data['protein'].pos_Cb += torch.randn_like(data['protein'].pos_Cb) * self.args.backbone_noise

        data.protein_size = data['protein'].pos.shape[0]
        data['protein'].inter_res_dist = None
        data['ligand', 'bond_edge', 'ligand'].edge_attr = data['ligand', 'bond_edge', 'ligand'].edge_attr.resize(0, len(bond_features_list)) if data['ligand', 'bond_edge', 'ligand'].edge_attr.nelement() == 0 else data['ligand', 'bond_edge', 'ligand'].edge_attr # we do this because we have no edge attributes if we have onlye a single atom
        assert not torch.isnan(data['protein'].pos).any()
        return data

    def get_diffdock_pocket(self, data):
        diffdock_contact_res = data['protein'].min_lig_dist < 5
        pocket_center = data['protein'].pos[diffdock_contact_res].mean(dim=0)
        dist = ((data['ligand'].pos - pocket_center) ** 2).sum(-1).sqrt()
        radius = dist.max() + 10
        pocket_mask =  ((data['protein'].pos - pocket_center) ** 2).sum(-1).sqrt() < radius
        data.diffdock_contact_res = diffdock_contact_res[pocket_mask]
        return get_protein_subgraph(data, pocket_mask)

    def get_fake_lig_id(self, data, idx):
        # remove residues that are close in the chain and less than 12A away
        arange = torch.arange(len(data['protein'].pos))
        valid_fake_lig_ids = []
        for i in range(len(data['protein'].pos)):
            valid_mask = (i - self.args.num_chain_masks > arange) | (arange > i + self.args.num_chain_masks) | (data['protein'].pdb_chain_id != data['protein'].pdb_chain_id[i]) | (data['protein'].inter_res_dist[i] > self.args.min_chain_mask_dist)
            contact_mask = (data['protein'].inter_res_dist[i] < self.args.design_residue_cutoff)
            num_valid_contacts = (contact_mask & valid_mask).sum()
            if num_valid_contacts > self.args.fake_min_num_contacts:
                valid_fake_lig_ids.append(i)
        if len(valid_fake_lig_ids) == 0:
            lg(f'Warning, no valid fake residues for {data.pdb_id} with idx {idx}, returning None now such that a new complex can be chosen.')
            return None
        else:
            fake_lig_id = valid_fake_lig_ids[torch.randint(low=0, high=len(valid_fake_lig_ids), size=(1,))]
            data['protein'].fake_lig_id = fake_lig_id
            return fake_lig_id

    def update_data(self, data1: 'HeteroData', data2: 'HeteroData') -> 'HeteroData':
        for store in data2.stores:
            for key, value in store.items():
                data1[store._key][key] = value
        return data1
    def init_fake_lig(self, data, fake_lig_id):
        # construct the ligand
        fake_lig_atom_names = data['protein'].atom_names[data['protein'].atom_res_idx == fake_lig_id]
        data['ligand'].fake_lig_type = seq1(atom_features_list['residues'][data['protein'].feat[fake_lig_id][0]])
        if data['protein'].feat[fake_lig_id][0].item() == len(atom_features_list['residues_canonical']) - 1:
            lg(f'Warning, {data.pdb_id} has non canonical amino acid at fake_lig_id {fake_lig_id} and pdb_res_id {data["protein"].pdb_res_id[fake_lig_id]} and pdb_chain_id {data["protein"].pdb_chain_id[fake_lig_id]}. Trying a new random complex instead now.')
            return False
        if not len(fake_lig_atom_names) == len(amino_acid_atom_names[data['ligand'].fake_lig_type]) or not np.all(np.array(fake_lig_atom_names) == amino_acid_atom_names[data['ligand'].fake_lig_type]):
            lg(f'Warning, {data.pdb_id} with amino acid {data["ligand"].fake_lig_type} has atom names {fake_lig_atom_names} instead of {amino_acid_atom_names[data["ligand"].fake_lig_type]}. Trying a new random complex instead now. The fake_lig_id is {fake_lig_id} and pdb_res_id {data["protein"].pdb_res_id[fake_lig_id]} and pdb_chain_id {data["protein"].pdb_chain_id[fake_lig_id]}. Trying a new random complex instead now.')
            return False

        res_pos = data['protein'].atom_pos[data['protein'].atom_res_idx == fake_lig_id]

        fake_lig_pos = res_pos[1:] # drop the nitrogen position. If you remove more coordinates here than just the N, then you also have to change the min distance calculation in the preprocessing in get_inter_res_distances
        rdkit_res = RemoveHs(Chem.MolFromFASTA(data['ligand'].fake_lig_type))
        edit_mol = Chem.EditableMol(rdkit_res)

        # remove the Nitrogen at the beginning and the Oxygen at the end which is the OH of the acid group (maybe one should instead replace it with the C of the next residue)
        remove_atoms = [0, rdkit_res.GetNumAtoms() - 1]
        for idx in sorted(remove_atoms, reverse=True):
            edit_mol.RemoveAtom(idx)
        lig = edit_mol.GetMol()

        # add positions to rdkit ligand
        assert len(fake_lig_pos) == lig.GetNumAtoms()
        conformer = Chem.Conformer(lig.GetNumAtoms())
        for atom_idx, coord in enumerate(fake_lig_pos):
            conformer.SetAtomPosition(atom_idx, Point3D(*(coord.tolist())))
        lig.AddConformer(conformer)

        # sanitize lig and
        Chem.SanitizeMol(lig)
        init_lig_graph(self.args, lig, data)
        data['ligand'].rdkit_lig = lig

        '''
        pdb_id = data.pdb_id
        os.makedirs(f"data/{pdb_id}_sidechain_vis", exist_ok=True)
        shutil.copy(os.path.join(self.args.data_dir, pdb_id, f'{pdb_id}_ligand.mol2'), os.path.join(f"data/{pdb_id}_sidechain_vis", f'{pdb_id}_ligand.mol2'))
        shutil.copy(os.path.join(self.args.data_dir, pdb_id, f'{pdb_id}_{self.args.protein_file_name}.pdb'), os.path.join(f"data/{pdb_id}_sidechain_vis", f'{pdb_id}_{self.args.protein_file_name}.pdb'))
        file = PDBFile(lig)
        file.add(fake_lig_pos + data.original_center , order=0, part=0)
        file.write(path=f'data/{pdb_id}_sidechain_vis/debug_sidechain_lig{fake_lig_id}_{data["ligand"].fake_lig_type}.pdb')
        '''
        return True

    def center_data(self, data):
        if self.args.diffusion_center_radius is not None:
            mask = torch.zeros_like(data['protein'].min_lig_dist).bool()
            mask[data['protein'].min_lig_dist < self.args.diffusion_center_radius] = True
        elif self.args.full_prot_diffusion_center:
            mask = torch.ones_like(data['protein'].min_lig_dist).bool()
        elif self.args.diffdock_pocket:
            mask = data.diffdock_contact_res
        else:
            mask = data['protein'].designable_mask
        data.diffusion_center = torch.mean(data['protein'].pos[mask], dim=0, keepdim=True)
        data['protein'].atom_pos -= data.diffusion_center
        data['protein'].pos -= data.diffusion_center
        data['protein'].pos_Cb -= data.diffusion_center
        data['protein'].pos_C -= data.diffusion_center
        data['protein'].pos_O -= data.diffusion_center
        data['protein'].pos_N -= data.diffusion_center
        data['ligand'].pos -= data.diffusion_center

    def preprocess(self, pdb_id):
        try:
            args = self.args
            if args.data_source == 'moad':
                protein_name = f'{pdb_id}.pdb'
                prody_struct = prody.parsePDB(os.path.join(args.data_dir, pdb_id, protein_name))
                if not args.old_data:
                    ligs_sel = prody_struct.select('not protein').select("not water").select('not hydrogen')
                else:
                    ligs_sel = prody_struct.select('not water and hetero').select('not hydrogen')
                ligs_pos = []
                res_names = []
                res_ids = []
                chain_ids = []
                for res in ligs_sel.getHierView().iterResidues():
                    ligs_pos.append(res.getCoords())
                    res_names.append(res.getResname())
                    res_ids.append(res.getResnum())
                    chain_ids.append(res.getChid())
                dist = np.zeros((len(ligs_pos), len(ligs_pos)))
                for i in range(len(ligs_pos)):
                    for j in range(len(ligs_pos)):
                        dist[i, j] = scipy.spatial.distance.cdist(ligs_pos[i], ligs_pos[j]).min()
                adjacency_matrix = dist < self.args.lig_connection_radius

                n_components, labels = scipy.sparse.csgraph.connected_components(adjacency_matrix, directed=False)
                component_indices = [np.where(labels == i)[0] for i in range(n_components)]

                lig_names = []
                lig_num_components = []
                lig_ccd_ids = []
                for connected_idx in component_indices:
                    connected_resnums = np.array(res_ids)[connected_idx]
                    connected_resnames = np.array(res_names)[connected_idx]
                    connected_chain_ids = np.array(chain_ids)[connected_idx]
                    connected_atoms = ligs_sel.select('resnum ' + ' '.join(connected_resnums.astype(str)))
                    if self.args.correct_moad_lig_selection:
                        connected_atoms = ligs_sel.select('resnum ' + ' '.join(connected_resnums.astype(str))).select('resname ' + ' '.join(connected_resnames)).select('chain ' + ' '.join(connected_chain_ids))
                    lig_name = pdb_id +f'_lig{self.args.lig_connection_radius}'+ ''.join([f'-{res_name}_resID{res_id}_chID{chain_id}' for res_name, res_id, chain_id in zip(connected_resnames, connected_resnums, connected_chain_ids)]) + '.pdb'
                    prody.writePDB(os.path.join(args.data_dir, pdb_id, lig_name), connected_atoms)
                    lig_names.append(lig_name)
                    lig_num_components.append(len(connected_idx))
                    lig_ccd_ids.append(connected_resnames)
            else:
                lig_num_components = [1]
                lig_ccd_ids = ['XXX']
                lig_names = [f'{pdb_id}_ligand.mol2']
                protein_name = f'{pdb_id}_{args.protein_file_name}.pdb'
                prody_struct = prody.parsePDB(os.path.join(args.data_dir, pdb_id, protein_name))

            data = HeteroData()
            data['ligand'] = []
            data['pdb_id'] = pdb_id
            data['protein'].name = protein_name
            success_count = 0
            for idx, lig_name in enumerate(lig_names):
                try:
                    lig = read_molecule(os.path.join(args.data_dir, pdb_id, lig_name))
                    lig = RemoveHs(lig)
                    lig_data = HeteroData()
                    init_lig_graph(args, lig, lig_data)
                    lig_data['ligand'].ccd_ids = np.array(lig_ccd_ids[idx])
                    lig_data['ligand'].num_components = np.array(lig_num_components[idx])
                    lig_data['ligand'].name = lig_name
                    data['ligand'].append(lig_data)
                    success_count += 1
                except Exception as e:
                    if idx == len(lig_names) - 1 and success_count == 0:
                        lg(f"ERROR: RDKit could not initialize any ligand in {pdb_id}")
                        raise e
                    else:
                        lg(f'Warning: RDKit could not initialize ligand {lig_names[idx]} in {pdb_id} which is number {idx} out of {len(lig_names)} ligands. Continuing with the next ligand.')
                        lg(str(e))

            featurize_prody(args, prody_struct, [lig_data['ligand'].pos.numpy() for lig_data in data['ligand']], data)

            lig_sizes = [lig_data['ligand'].size for lig_data in data['ligand']]
            lig_contacts = (data['protein'].min_lig_dist < 4).sum(dim=0)
            np.savez(os.path.join(self.full_cache_path, pdb_id + "lig_meta_data.npz"), lig_sizes, lig_contacts)
            torch.save(data, os.path.join(self.full_cache_path, pdb_id + ".pt"))
        except Exception as e:
            lg(f"Error in {pdb_id}: {e}")
            torch.save(None, os.path.join(self.full_cache_path, pdb_id + "none.pt"))
            os.makedirs('data/preprocess_errors', exist_ok=True)
            with open(f'data/preprocess_errors/{self.cache_file_name}', 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} Error in {pdb_id}: {e} \n")



