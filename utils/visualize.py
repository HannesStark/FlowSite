import copy
import os
import shutil
from collections import defaultdict
import matplotlib
import rdkit
from rdkit import Geometry
from rdkit.Chem import MolToPDBBlock, RemoveHs
from utils.featurize import read_molecule
matplotlib.use('Agg')
import numpy as np
import plotly.graph_objects as go
import torch

class PDBFile:
    # this class is from Bowen Jing
    def __init__(self, mol):
        self.parts = defaultdict(dict)
        self.mol = copy.deepcopy(mol)
        [self.mol.RemoveConformer(j) for j in range(mol.GetNumConformers()) if j]
        self.num_blocks = 0

    def add(self, pos, order=None, part=0, repeat=1):
        if order is not None:
            order = self.num_blocks
            self.num_blocks += 1
        if type(pos) in [rdkit.Chem.Mol, rdkit.Chem.RWMol]:
            block = MolToPDBBlock(pos).split("\n")[:-2]
            self.parts[part][order] = {"block": block, "repeat": repeat}
            return
        elif type(pos) is np.ndarray:
            pos = pos.astype(np.float64)
        elif type(pos) is torch.Tensor:
            pos = pos.double().numpy()
        for i in range(pos.shape[0]):
            self.mol.GetConformer(0).SetAtomPosition(
                i, Geometry.Point3D(pos[i, 0], pos[i, 1], pos[i, 2])
            )
        block = MolToPDBBlock(self.mol).split("\n")[:-2]
        self.parts[part][order] = {"block": block, "repeat": repeat}

    def write(self, path=None, limit_parts=None):
        is_first = True
        str_ = ""
        for part in sorted(self.parts.keys()):
            if limit_parts and part >= limit_parts:
                break
            part = self.parts[part]
            keys_positive = sorted(filter(lambda x: x >= 0, part.keys()))
            keys_negative = sorted(filter(lambda x: x < 0, part.keys()))
            keys = list(keys_positive) + list(keys_negative)
            for key in keys:
                block = part[key]["block"]
                times = part[key]["repeat"]
                for _ in range(times):
                    if not is_first:
                        block = [line for line in block if "CONECT" not in line]
                    is_first = False
                    str_ += "MODEL\n"
                    str_ += "\n".join(block)
                    str_ += "\nENDMDL\n"
        if not path:
            return str_
        with open(path, "w") as f:
            f.write(str_)

def save_trajectory_pdb(args, batch, trajectory, model_pred, extra_string = '', production_mode= False, out_dir=None):
    bid = batch["ligand"].batch
    if out_dir is None:
        out_dir = f"{os.environ['MODEL_DIR']}/inference_output"
    pdb_files_xt = []
    pdb_files_x1 = []
    for i, pdb_id in enumerate(batch.pdb_id):
        os.makedirs(f"{out_dir}/{pdb_id}", exist_ok=True)
        if not os.path.exists(os.path.join(f"{out_dir}/{pdb_id}", batch['ligand'].name[i])) and not production_mode:
            shutil.copy(os.path.join(args.data_dir, pdb_id, batch['ligand'].name[i]), os.path.join(f"{out_dir}/{pdb_id}", batch['ligand'].name[i]))
        if not os.path.exists(os.path.join(f"{out_dir}/{pdb_id}", batch['protein'].name[i])) and not production_mode:
            shutil.copy(os.path.join(args.data_dir, pdb_id, batch['protein'].name[i]), os.path.join(f"{out_dir}/{pdb_id}", batch['protein'].name[i]))
        rdkit_mol = RemoveHs(batch['ligand'].mol[i] if production_mode else read_molecule(os.path.join(args.data_dir, pdb_id, batch['ligand'].name[i]), sanitize=True))
        pdb_files_xt.append(PDBFile(rdkit_mol))
        pdb_files_x1.append(PDBFile(rdkit_mol))

    for idx, (xt, model_pred) in enumerate(zip(trajectory, model_pred)):
        for i, (f_xt, f_preds) in enumerate(zip(pdb_files_xt, pdb_files_x1)):
                center_offset = (batch.original_center[i].detach().cpu() + batch.pocket_center[i].detach().cpu())
                f_xt.add((xt[torch.where(bid == i)[0]]).detach().cpu() + center_offset, order=idx + 1, part=0)
                f_preds.add((model_pred[torch.where(bid == i)[0]]).detach().cpu() + center_offset, order=idx + 1, part=0)

    for i, (f_xt, f_preds) in enumerate(zip(pdb_files_xt, pdb_files_x1)):
            os.makedirs(f"{out_dir}/{batch.pdb_id[i]}", exist_ok=True)
            f_xt.write(path=f"{out_dir}/{batch.pdb_id[i]}/{batch.pdb_id[i]}_{extra_string}_xt.pdb")
            f_preds.write(path=f"{out_dir}/{batch.pdb_id[i]}/{batch.pdb_id[i]}_{extra_string}_x1.pdb")

def plot_point_cloud(point_clouds):
    # Takes a list of point cloud tensors and plots them
    if not isinstance(point_clouds, list):
        point_clouds = [point_clouds]

    colors = ['red', 'blue', 'green', 'yellow', 'orange']  # List of colors for each point cloud
    traces = []  # List to store individual traces for each point cloud

    for i, point_cloud in enumerate(point_clouds):
        if isinstance(point_cloud, np.ndarray):
            pass
        elif isinstance(point_cloud, torch.Tensor):
            point_cloud = point_cloud.numpy()

        x_data = point_cloud[:, 0]
        y_data = point_cloud[:, 1]
        z_data = point_cloud[:, 2]

        # Create a trace for each point cloud with a different color
        trace = go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=5,
                opacity=0.8,
                color=colors[i % len(colors)]  # Assign color based on the index of the point cloud
            ),
            name=f"Point Cloud {i + 1}"
        )
        traces.append(trace)

    # Create the layout
    layout = go.Layout(
        scene=dict(
            aspectmode='data'
        )
    )

    # Create the figure and add the traces to it
    fig = go.Figure(data=traces, layout=layout)

    # Show the figure
    fig.show()


