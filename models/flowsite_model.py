import time
from functools import partial

import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch_cluster
from e3nn import o3
from torch.nn import ModuleList
from torch_geometric.nn import PNAConv, BatchNorm
from torch_geometric.utils import unbatch
from torch_scatter import scatter_mean

from models.invariant_layers import InvariantLayer
from models.pytorch_modules import Linear, Encoder
from models.tfn_layers import RefinementTFNLayer, build_cg, GaussianSmearing
from utils.diffusion import get_time_mapping

from utils.featurize import get_feature_dims, atom_features_list
from utils.simdesign_utils import gather_nodes, _dihedrals, _orientations_coarse_gl_tuple, _get_rbf



class FlowSiteModel(nn.Module):
    def __init__(self, args, device):
        super(FlowSiteModel, self).__init__()
        self.args = args
        self.device = device
        fold_dim = args.fold_dim
        num_inv_layers = args.num_inv_layers
        atom_feature_dims, edge_feature_dims, lig_bond_feature_dims, rec_feature_dims = get_feature_dims()

        assert args.use_tfn or args.use_inv, "Must use at least one of tfn or inv, otherwise this model will do nothing"
        assert not (args.use_tfn and args.ignore_lig), "Tensorfield always uses lig so ignore_lig does not work with it"
        if not args.ignore_lig or args.lig2d_mpnn:
            self.lig_node_embedder = nn.Sequential(
                Encoder(emb_dim=fold_dim, feature_dims=atom_feature_dims),
                nn.ReLU(),
                Linear(fold_dim, fold_dim, init="final" if args.fancy_init else "default"),
            )

            self.lig_edge_embedder = nn.Sequential(
                Encoder(emb_dim=fold_dim, feature_dims=lig_bond_feature_dims),
                nn.ReLU(),
                Linear(fold_dim, fold_dim, init="final" if args.fancy_init else "default"),
            )
        if not args.ignore_lig:
            self.lig_edge_builder = LigEdgeBuilder(args, device)
            self.cross_edge_builder = CrossEdgeBuilder(args, device)
        if self.args.lig2d_mpnn:
            if self.args.lig2d_batch_norm: self.mpnn_batch_norms = ModuleList([BatchNorm(args.fold_dim) for i in range(args.lig_mpnn_layers)])
            self.mpnn_convs = ModuleList([PNAConv(in_channels=args.fold_dim , out_channels=args.fold_dim , aggregators=['mean', 'min', 'max', 'sum'], scalers=['identity'], deg=torch.tensor([0, 225, 21, 135, 65]).to(device), edge_dim=args.fold_dim) for i in range(args.lig_mpnn_layers)])

        if self.args.use_inv:
            if self.args.self_condition_inv:
                if self.args.self_condition_bit:
                    self.self_condition_bit_encoder = Encoder(emb_dim=args.fold_dim, feature_dims=[2])
                rec_feature_dims[0] += 1 # now we have 22 different values. 20 for the amino acids, 1 for unknown, and one for a mask token.
                self.inv_rec_node_embedder = nn.Sequential(
                    Encoder(emb_dim=args.fold_dim, feature_dims=rec_feature_dims),
                    nn.ReLU(),
                    Linear(args.fold_dim, args.fold_dim, init="final" if args.self_fancy_init else "default"),
                )
                if self.args.self_condition_inv_logits:
                    self.inv_self_logit_embedder = nn.Sequential(
                        Linear(len(atom_features_list['residues_canonical']), args.fold_dim, init="final" if args.self_fancy_init else "default"),
                        nn.ReLU(),
                        Linear(args.fold_dim, args.fold_dim, init="final" if args.self_fancy_init else "default"),
                    )
                if self.args.standard_style_self_condition_inv:
                    self.standard_self_condition_inv_embedder = nn.Sequential(
                        Encoder(emb_dim=args.fold_dim, feature_dims=rec_feature_dims),
                        nn.ReLU(),
                        Linear(args.fold_dim, args.fold_dim, init="final" if args.self_fancy_init else "default"),
                    )
                    self.standard_self_condition_inv_combiner = nn.Sequential(
                        Linear(args.fold_dim * 3, args.fold_dim),
                        nn.ReLU(),
                        Linear(args.fold_dim, args.fold_dim, init="final" if args.self_fancy_init else "default"),
                    )

            self.inv_embedder = PiFoldEmbedder(args, device)
            self.inv_layers = nn.Sequential(*[InvariantLayer(args, device, update_edges=True if i + 1 < num_inv_layers else False) for i in range(num_inv_layers)])

            if self.args.time_condition_inv:
                time_mapping = get_time_mapping(args.time_emb_type, args.time_emb_dim)
                self.time_encoder_inv = nn.Sequential(
                    time_mapping,
                    Linear(args.time_emb_dim, args.fold_dim),
                    nn.ReLU(),
                    Linear(args.fold_dim, args.fold_dim)
                )

        self.decoder = nn.Linear(fold_dim, len(atom_features_list['residues_canonical']))
        if self.args.num_angle_pred > 0:
            self.angle_linear = Linear(fold_dim, fold_dim)
            self.angle_linear_skip = Linear(fold_dim, fold_dim)
            self.angle_decoder1 = nn.Sequential(Linear(fold_dim, fold_dim), nn.ReLU(), Linear(fold_dim, fold_dim), nn.ReLU())
            self.angle_decoder2 = nn.Sequential(Linear(fold_dim, fold_dim), nn.ReLU(), Linear(fold_dim, fold_dim, nn.ReLU()))
            self.angle_predictor = nn.Linear(fold_dim, self.args.num_angle_pred * 2)

        if self.args.use_tfn:
            self.tfn = TensorFieldNet(args, device)
            if not args.drop_tfn_feat:
                self.lig_tfn2inv = nn.Sequential(Linear(args.ns, args.ns), nn.ReLU(), Linear(args.ns, fold_dim))
                self.rec_tfn2inv = nn.Sequential(Linear(args.ns, args.ns), nn.ReLU(), Linear(args.ns, fold_dim))
        assert not ((args.time_condition_inv or args.time_condition_tfn) and args.ignore_lig), "It does not make sense to use time conditioning without the ligand and therefore without diffusion."

    def forward(self, data, x_self=None, x_prior= None):
        if self.args.use_tfn:
            lig_na_tfn, rec_na_tfn, lig_pos_stack = self.tfn(data, x_self, x_prior)
            if self.args.tfn_detach:
                lig_na_tfn, rec_na_tfn = lig_na_tfn.detach(), rec_na_tfn.detach()
            data['ligand'].pos = lig_pos_stack[-1].detach()

        if self.args.use_inv:
            if self.args.ignore_lig:
                lig_na, lig_ea, lig_idx, cross_idx, cross_ea = None, None, None, None, None
            else:
                lig_na = self.lig_node_embedder(data["ligand"].feat)
                lig_ea = self.lig_edge_embedder(data['ligand', 'bond_edge', 'ligand'].edge_attr)
                lig_idx, lig_ea = self.lig_edge_builder(data, lig_ea)
                cross_idx, cross_ea = self.cross_edge_builder(data)

            rec_na, rec_ea, rec_idx = self.inv_embedder(data)
            if self.args.self_condition_inv and self.args.residue_loss_weight > 0:
                if self.args.standard_style_self_condition_inv:
                    self_conditioned_rec_na = self.inv_rec_node_embedder(data['protein'].input_feat) if isinstance(data['protein'].input_feat, torch.LongTensor) or isinstance(data['protein'].input_feat, torch.cuda.LongTensor) else self.inv_self_logit_embedder(data['protein'].input_feat)
                    original_feat = self.standard_self_condition_inv_embedder(data['protein'].original_feat)
                    rec_na = rec_na + self.standard_self_condition_inv_combiner(torch.cat([self_conditioned_rec_na, original_feat, rec_na], dim=1))
                else:
                    rec_na = rec_na + self.inv_rec_node_embedder(data['protein'].input_feat) if isinstance(data['protein'].input_feat, torch.LongTensor) or isinstance(data['protein'].input_feat, torch.cuda.LongTensor) else self.inv_self_logit_embedder(data['protein'].input_feat)
                if self.args.self_condition_bit:
                    rec_na = rec_na + self.self_condition_bit_encoder(data.self_condition_bit[data['protein'].batch.long()])

            if self.args.lig2d_mpnn:
                mpnn_lig_ea = self.lig_edge_embedder(data['ligand', 'bond_edge', 'ligand'].edge_attr)
                mpnn_lig_na = self.lig_node_embedder(data["ligand"].feat)
                for i, conv in enumerate(self.mpnn_convs):
                    mpnn_lig_na = conv(mpnn_lig_na, data['ligand', 'bond_edge', 'ligand'].edge_index, mpnn_lig_ea)
                    if self.args.lig2d_batch_norm and i < self.args.lig_mpnn_layers-1: mpnn_lig_na = self.mpnn_batch_norms[i](mpnn_lig_na)
                    if self.args.lig2d_additional_relu and i < self.args.lig_mpnn_layers-1: mpnn_lig_na = F.relu(mpnn_lig_na)
                mpnn_lig_na_pooled = scatter_mean(mpnn_lig_na, data['ligand'].batch, dim=0)
                rec_na = rec_na + mpnn_lig_na_pooled[data['protein'].batch.long()]

            if self.args.time_condition_inv:
                temb_inv = self.time_encoder_inv(data.t01 if self.args.correct_time_condition else data.normalized_t)
                rec_na = rec_na + temb_inv[data['protein'].batch.long()]
                lig_na = lig_na + temb_inv[data['ligand'].batch.long()]
                lig_ea = lig_ea + temb_inv[data['ligand'].batch.long()[lig_idx[0].long()]]
                rec_ea = rec_ea + temb_inv[data['protein'].batch.long()[rec_idx[0].long()]]
                cross_ea = cross_ea + temb_inv[data['ligand'].batch.long()[cross_idx[0].long()]]
            else:
                temb_inv = None


            if self.args.use_tfn and not self.args.drop_tfn_feat:
                lig_na = lig_na + self.lig_tfn2inv(lig_na_tfn[:, :self.args.ns])
                rec_na = rec_na + self.rec_tfn2inv(rec_na_tfn[:, :self.args.ns])

            for inv_layer in self.inv_layers:
                rec_na, rec_ea, lig_na, lig_ea, cross_ea = inv_layer(data, rec_idx, rec_na, rec_ea, lig_idx, lig_na, lig_ea, cross_idx, cross_ea, temb_inv)
        else:
            rec_na = self.rec_tfn2inv(rec_na_tfn[:, :self.args.ns])

        logits = self.decoder(rec_na)
        if self.args.num_angle_pred > 0:
            angle_na = self.angle_linear(rec_na) + self.angle_linear_skip(rec_na)
            angle_na = angle_na + self.angle_decoder1(angle_na)
            angle_na = angle_na + self.angle_decoder2(angle_na)
            angles = self.angle_predictor(angle_na).reshape(-1, self.args.num_angle_pred, 2)
        else:
            angles = None

        return logits, lig_pos_stack if self.args.use_tfn else torch.stack([data['ligand'].pos]), angles

class TensorFieldNet(nn.Module):
    def __init__(self, args, device):
        super(TensorFieldNet, self).__init__()
        self.args = args
        self.device = device
        atom_feature_dims, edge_feature_dims, lig_bond_feature_dims, rec_feature_dims = get_feature_dims()

        assert args.nv >= 3, "nv must be at least 3 to accommodate N, C, O vector features"
        self.feature_irreps = o3.Irreps([(args.ns if l == 0 else args.nv, (l, 1)) for l in range(args.order + 1)])
        self.sh_irreps = o3.Irreps([(1, (l, 1)) for l in range(args.sh_lmax + 1)])
        self.build_cg = partial(build_cg, sh_irreps=self.sh_irreps)
        self.rec_radius_embedding = nn.Sequential(
                    GaussianSmearing(0.0, args.protein_radius, args.radius_emb_dim),
                    Linear(args.radius_emb_dim, args.ns, init="relu" if args.fancy_init else "default",),
                    nn.ReLU(),
                    Linear(args.ns, args.ns, init="final" if args.fancy_init else "default"),
                )
        if self.args.tfn_pifold_feat:
            self.inv_embedder = PiFoldEmbedder(args, device, dim=args.ns)
        else:
            self.rec_node_init = nn.Parameter(torch.randn(args.ns))

        if self.args.self_condition_inv and self.args.residue_loss_weight > 0 or self.args.tfn_use_aa_identities:
            if self.args.self_condition_inv and self.args.residue_loss_weight > 0 and not self.args.no_tfn_self_condition_inv:
                rec_feature_dims[0] += 1  # now we have 22 different values. 20 for the amino acids, 1 for unknown, and one for a mask token.
            self.rec_node_embedder = nn.Sequential(
                Encoder(emb_dim=args.ns, feature_dims=rec_feature_dims),
                nn.ReLU(),
                Linear(args.ns, args.ns, init="final" if args.fancy_init else "default"),
            )
            if self.args.self_condition_inv_logits:
                self.inv_self_logit_embedder = nn.Sequential(
                    Linear(len(atom_features_list['residues_canonical']), args.ns),
                    nn.ReLU(),
                    Linear(args.ns, args.ns, init="final" if args.self_fancy_init else "default"),
                )

        self.lig_node_embedder = nn.Sequential(
            Encoder(emb_dim=args.ns, feature_dims=atom_feature_dims),
            nn.ReLU(),
            Linear(args.ns, args.ns, init="final" if args.fancy_init else "default"),
        )

        self.lig_edge_embedder = nn.Sequential(
            Encoder(emb_dim=args.ns, feature_dims=lig_bond_feature_dims),
            nn.ReLU(),
            Linear(args.ns, args.ns, init="final" if args.fancy_init else "default"),
        )
        if self.args.time_condition_tfn:
            time_mapping = get_time_mapping(args.time_emb_type, args.time_emb_dim)
            self.time_encoder_tfn = nn.Sequential(
                time_mapping,
                Linear(args.time_emb_dim, args.ns),
                nn.ReLU(),
                Linear(args.ns, args.ns)
            )

        self.tfn_layers = nn.ModuleList([RefinementTFNLayer(args, last_layer=i == (args.num_tfn_layers - 1)) for i in range(args.num_tfn_layers)])

    def forward(self, data, x_self=None, x_prior=None):
        lig_pos = data["ligand"].pos
        rec_cg = self.build_cg(
            pos=data["protein"].pos,
            edge_attr=None,
            edge_index=data['protein', 'radius_graph', 'protein'].edge_index,
            radius_emb_func=self.rec_radius_embedding,
            batch=None,
            radius=None,
        )
        if self.args.tfn_pifold_feat:
            init_scalar_rec_na, _, _ = self.inv_embedder(data)
        else:
            init_scalar_rec_na = self.rec_node_init.expand(data['protein'].num_nodes, -1)
        if self.args.self_condition_inv and self.args.residue_loss_weight > 0 and not self.args.no_tfn_self_condition_inv or self.args.tfn_use_aa_identities:
            init_scalar_rec_na = init_scalar_rec_na + self.rec_node_embedder(data['protein'].input_feat) if isinstance(data['protein'].input_feat, torch.LongTensor) or isinstance(data['protein'].input_feat, torch.cuda.LongTensor) else self.inv_self_logit_embedder(data['protein'].input_feat)
        lig_ea = self.lig_edge_embedder(data['ligand', 'bond_edge', 'ligand'].edge_attr)
        lig_na = self.lig_node_embedder(data["ligand"].feat)

        if self.args.time_condition_tfn:
            temb = self.time_encoder_tfn(data.t01 if self.args.correct_time_condition else data.normalized_t)
            lig_na = lig_na + temb[data['ligand'].batch.long()]
            lig_ea = lig_ea + temb[data['ligand'].batch.long()[data['ligand', 'bond_edge', 'ligand'].edge_index.long()[0]]]
            init_scalar_rec_na = init_scalar_rec_na + temb[data['protein'].batch.long()]
            rec_cg.attr = rec_cg.attr + temb[data['protein'].batch.long()[rec_cg.idx[1].long()]]
        else:
            temb = None

        if self.args.no_tfn_vector_inputs:
            rec_na = F.pad(init_scalar_rec_na, (0, self.feature_irreps.dim - self.args.ns))
        else:
            rec_na = torch.cat([init_scalar_rec_na, data['protein'].pos_N - data['protein'].pos, data['protein'].pos_C - data['protein'].pos, data['protein'].pos_O - data['protein'].pos], dim=1)
            rec_na = F.pad(rec_na, (0, self.feature_irreps.dim - self.args.ns - 9))
        lig_na = F.pad(lig_na, (0, self.feature_irreps.dim - self.args.ns))

        data.logs['radius_graph_time'] = 0
        lig_pos_list = []
        for tfn_layer in self.tfn_layers:
            lig_pos, lig_na, rec_na = tfn_layer(data, rec_cg, lig_pos.detach(), lig_na, lig_ea, rec_na, temb, x_self, x_prior)
            lig_pos_list.append(lig_pos)
        return lig_na, rec_na, torch.stack(lig_pos_list)

class LigEdgeBuilder(nn.Module):
    def __init__(self, args, device):
        super(LigEdgeBuilder, self).__init__()
        self.args = args
        self.device = device
        self.lig_radius_embedder = nn.Sequential(
            GaussianSmearing(0.0, args.protein_radius, args.radius_emb_dim),
            Linear(
                args.radius_emb_dim,
                args.fold_dim,
                init="relu" if args.fancy_init else "default",
            ),
            nn.ReLU(),
            Linear(args.fold_dim, args.fold_dim, init="final" if args.fancy_init else "default"),
        )
    def forward(self, data, lig_ea):
        edge_index, edge_attr = build_cg_general(
            pos=data['ligand'].pos,
            edge_attr=lig_ea,
            edge_index=data["ligand", 'bond_edge', "ligand"].edge_index,
            batch=data["ligand"].batch,
            radius=self.args.lig_radius,
        )
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_attr = edge_attr + self.lig_radius_embedder(edge_vec.norm(dim=-1))
        return edge_index, edge_attr

class CrossEdgeBuilder(nn.Module):
    def __init__(self, args, device):
        super(CrossEdgeBuilder, self).__init__()
        self.args = args
        self.device = device
        self.cross_rbf = GaussianSmearing(0.0, args.protein_radius, args.radius_emb_dim)
        self.cross_attr_embedder = nn.Sequential(
            Linear(
                args.radius_emb_dim * 5,
                args.fold_dim,
                init="relu" if args.fancy_init else "default",
            ),
            nn.ReLU(),
            Linear(args.fold_dim, args.fold_dim, init="final" if args.fancy_init else "default"),
        )

    def forward(self, data):
        cross_idx, _ = build_cg_general(
            pos=(data["protein"].pos, data['ligand'].pos),
            edge_attr=None,
            edge_index=None,
            batch=(data["protein"].batch, data["ligand"].batch),
            radius=self.args.cross_radius,
        )
        # src is the indices for ligand and dst is the indices for protein
        src, dst = cross_idx
        edge_vec = data['ligand'].pos[src.long()] - data['protein'].pos[dst.long()]
        edge_vec_cb = data['ligand'].pos[src.long()] - data['protein'].pos_Cb[dst.long()]
        edge_vec_c = data['ligand'].pos[src.long()] - data['protein'].pos_C[dst.long()]
        edge_vec_o = data['ligand'].pos[src.long()] - data['protein'].pos_O[dst.long()]
        edge_vec_n = data['ligand'].pos[src.long()] - data['protein'].pos_N[dst.long()]
        edge_attr = torch.cat([self.cross_rbf(edge_vec.norm(dim=-1)),
                               self.cross_rbf(edge_vec_cb.norm(dim=-1)),
                               self.cross_rbf(edge_vec_c.norm(dim=-1)),
                               self.cross_rbf(edge_vec_o.norm(dim=-1)),
                               self.cross_rbf(edge_vec_n.norm(dim=-1))], dim=-1)
        return cross_idx, self.cross_attr_embedder(edge_attr)

def build_cg_general(
    pos,
    edge_attr,
    edge_index,
    batch=None,
    radius=None,
):
    if isinstance(pos, tuple):
        pos1, pos2 = pos
        batch1, batch2 = batch or (None, None)
    else:
        pos1 = pos2 = pos
        batch1 = batch2 = batch

    if radius is not None:
        if isinstance(radius, tuple):
            radius1, radius2 = radius
        else:
            radius1 = radius2 = radius

        # 0 element is the indices for pos2 and 1 element is the indices for pos1
        radius_edge_idx = torch_cluster.radius(pos1 / radius1, pos2 / radius2, 1.0, batch1, batch2, max_num_neighbors=10000)

        if edge_index is not None:
            edge_index = torch.cat([edge_index, radius_edge_idx], 1).long()
        else:
            edge_index = radius_edge_idx

        if edge_attr is not None:
            edge_attr = F.pad(edge_attr, (0, 0, 0, radius_edge_idx.shape[-1]))

    return edge_index, edge_attr
def get_edge_attr(conv_graph, node_attr1, node_attr2=None, ns=None):
    if node_attr2 is None:
        node_attr2 = node_attr1
    src, dst = conv_graph.idx
    return torch.cat([conv_graph.attr, node_attr1[src, :ns], node_attr2[dst, :ns]], -1)

class PiFoldEmbedder(nn.Module):
    def __init__(self, args, device, dim=None):
        super(PiFoldEmbedder, self).__init__()
        self.args = args
        self.device = device
        if dim is None:
            fold_dim = args.fold_dim
        else:
            fold_dim = dim

        self.top_k = args.k_neighbors
        self.num_rbf = 16

        self.virtual_atoms = nn.Parameter(torch.rand(self.args.virtual_num, 3, device=self.device))

        node_in = 0
        if self.args.node_dist:
            pair_num = 6
            if self.args.virtual_num > 0:
                pair_num += self.args.virtual_num * (self.args.virtual_num - 1)
            node_in += pair_num * self.num_rbf
        if self.args.node_angle:
            node_in += 12
        if self.args.node_direct:
            node_in += 9

        edge_in = 0
        if self.args.edge_dist:
            pair_num = 16
            if self.args.virtual_num > 0:
                pair_num += self.args.virtual_num
                pair_num += self.args.virtual_num * (self.args.virtual_num - 1)
            edge_in += pair_num * self.num_rbf
        if self.args.edge_angle:
            edge_in += 4
        if self.args.edge_direct:
            edge_in += 12

        self.node_embedding = nn.Linear(node_in, fold_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in, fold_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(fold_dim)
        self.norm_edges = nn.BatchNorm1d(fold_dim)

        self.W_v = nn.Sequential(
            nn.Linear(fold_dim, fold_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fold_dim),
            nn.Linear(fold_dim, fold_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fold_dim),
            nn.Linear(fold_dim, fold_dim, bias=True)
        )
        self.W_e = nn.Linear(fold_dim, fold_dim, bias=True)
        self._init_params()
    def forward(self,data):
        start = time.time()
        unbatched_pos = unbatch(data['protein'].pos, data['protein'].batch)
        pos_N = data['protein'].pos_N
        pos_C = data['protein'].pos_C
        pos_O = data['protein'].pos_O
        unbatched_pos_N = unbatch(pos_N, data['protein'].batch)
        unbatched_pos_C = unbatch(pos_C, data['protein'].batch)
        unbatched_pos_O = unbatch(pos_O, data['protein'].batch)

        lengths = np.array([len(b) for b in unbatched_pos], dtype=np.int32)
        L_max = max(lengths)
        B = len(unbatched_pos)
        X = torch.zeros([B, L_max, 4, 3], device=self.device)
        for i, (pos_Ca, pos_C, pos_N, pos_O) in enumerate(
                zip(unbatched_pos, unbatched_pos_C, unbatched_pos_N, unbatched_pos_O)):
            x = torch.stack([pos_N, pos_Ca, pos_C, pos_O], 1)  # [#atom, 3, 3]
            l = len(pos_Ca)
            x_pad = torch.from_numpy(np.pad(x.detach().cpu().numpy(), [[0, L_max - l], [0, 0], [0, 0]], 'constant',
                                            constant_values=(np.nan,))).to(
                self.device)  # [#atom, 4, 3]  # [#atom, 4, 3]
            X[i, :, :, :] = x_pad

        mask = torch.isfinite(torch.sum(X, (2, 3))).float()  # atom mask
        numbers = torch.sum(mask, axis=1).long()
        pos_new = torch.zeros_like(X) + torch.nan
        for i, n in enumerate(numbers):
            pos_new[i, :n, ::] = X[i][mask[i] == 1]
        pos = pos_new
        isnan = torch.isnan(X)
        mask = torch.isfinite(torch.sum(X, (2, 3))).float()
        pos[isnan] = 0.
        data.logs['padding_time'] = time.time() - start

        mask_bool = (mask == 1)
        B, N, _, _ = pos.shape
        X_ca = pos[:, :, 1, :]
        D_neighbors, rec_idx = self._full_dist(X_ca, mask, self.top_k)

        mask_attend = gather_nodes(mask.unsqueeze(-1), rec_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x: torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1, x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

        # angle & direction
        V_angles = _dihedrals(pos, 0)
        V_angles = node_mask_select(V_angles)

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(pos, rec_idx)
        V_direct = node_mask_select(V_direct)
        E_direct = edge_mask_select(E_direct)
        E_angles = edge_mask_select(E_angles)

        # distance
        atom_N = pos[:, :, 0, :]
        atom_Ca = pos[:, :, 1, :]
        atom_C = pos[:, :, 2, :]
        atom_O = pos[:, :, 3, :]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)

        if self.args.virtual_num > 0:
            virtual_atoms = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
            for i in range(self.virtual_atoms.shape[0]):
                vars()['atom_v' + str(i)] = virtual_atoms[i][0] * a \
                                            + virtual_atoms[i][1] * b \
                                            + virtual_atoms[i][2] * c \
                                            + 1 * atom_Ca

        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            node_dist.append(node_mask_select(
                _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))

        if self.args.virtual_num > 0:
            for i in range(self.virtual_atoms.shape[0]):
                # # true atoms
                for j in range(0, i):
                    node_dist.append(node_mask_select(
                        _get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], None,
                                 self.num_rbf).squeeze()))
                    node_dist.append(node_mask_select(
                        _get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], None,
                                 self.num_rbf).squeeze()))
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()

        pair_lst = ['Ca-Ca', 'Ca-C', 'C-Ca', 'Ca-N', 'N-Ca', 'Ca-O', 'O-Ca', 'C-C', 'C-N', 'N-C', 'C-O', 'O-C',
                    'N-N', 'N-O', 'O-N', 'O-O']

        edge_dist = []  # Ca-Ca
        for pair in pair_lst:
            atom1, atom2 = pair.split('-')
            rbf = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], rec_idx, self.num_rbf)
            edge_dist.append(edge_mask_select(rbf))

        if self.args.virtual_num > 0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(edge_mask_select(
                    _get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(i)], rec_idx, self.num_rbf)))
                for j in range(0, i):
                    edge_dist.append(edge_mask_select(
                        _get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], rec_idx, self.num_rbf)))
                    edge_dist.append(edge_mask_select(
                        _get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], rec_idx, self.num_rbf)))

        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        rec_na = []
        if self.args.node_dist:
            rec_na.append(V_dist)
        if self.args.node_angle:
            rec_na.append(V_angles)
        if self.args.node_direct:
            rec_na.append(V_direct)

        rec_ea = []
        if self.args.edge_dist:
            rec_ea.append(E_dist)
        if self.args.edge_angle:
            rec_ea.append(E_angles)
        if self.args.edge_direct:
            rec_ea.append(E_direct)

        rec_na = torch.cat(rec_na, dim=-1)
        rec_ea = torch.cat(rec_ea, dim=-1)

        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B, 1, 1) + rec_idx
        src = torch.masked_select(src, mask_attend).view(1, -1)
        dst = shift.view(B, 1, 1) + torch.arange(0, N, device=src.device).view(1, -1, 1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1, -1)
        rec_idx = torch.cat((dst, src), dim=0).long()

        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        batch_id = sparse_idx[:, 0]

        rec_na = self.W_v(self.norm_nodes(self.node_embedding(rec_na)))
        rec_ea = self.W_e(self.norm_edges(self.edge_embedding(rec_ea)))
        assert all(batch_id == data['protein'].batch)
        return rec_na, rec_ea, rec_idx

    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = (1. - mask_2D) * 10000 + mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max + 1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx

