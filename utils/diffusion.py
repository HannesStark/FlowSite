import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from models.tfn_layers import GaussianSmearing


class SinusoidalEmbedding(nn.Module):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    def __init__(self, embedding_dim, embedding_scale, max_positions=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        self.embedding_scale = embedding_scale

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1
        timesteps = timesteps * self.embedding_scale
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], self.embedding_dim)
        return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb

def get_time_mapping(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = SinusoidalEmbedding(embedding_dim=embedding_dim, embedding_scale=embedding_scale)
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    elif embedding_type == 'gaussian':
        emb_func = GaussianSmearing(0.0, 1, embedding_dim)
    else:
        raise NotImplemented
    return emb_func

def pyg_add_vp_noise(args, batch):
    batch["ligand"].shadow_pos = batch["ligand"].pos.clone()
    sde = DiffusionSDE(batch.protein_sigma)

    # quadratic time sampling schedule
    batch.normalized_t = torch.square(torch.rand_like(sde.lamb))
    batch.t = sde.max_t() * batch.normalized_t + 1  # temporary solution

    batch["ligand"].pos *= sde.mu_factor(batch.t)[batch["ligand"].batch, None]
    noise = torch.randn_like(batch["ligand"].pos)
    batch.std = torch.sqrt(sde.var(batch.t))
    batch["ligand"].pos += noise * batch.std[batch["ligand"].batch, None]
def pyg_add_harmonic_noise(args, batch):
    bid = batch['ligand'].batch
    sde = DiffusionSDE(batch.protein_sigma * args.prior_scale)

    if args.highest_noise_only:
        batch.t01 = torch.ones_like(sde.lamb)
    else:
        # quadratic time sampling schedule
        batch.t01 = torch.rand_like(sde.lamb)
    batch.normalized_t = torch.square(batch.t01)
    batch.t = sde.max_t() * batch.normalized_t + 1  # temporary solution

    edges = batch['ligand', 'bond_edge', 'ligand'].edge_index
    edges = edges[:, edges[0] < edges[1]]  # de-duplicate

    D, P = HarmonicSDE.diagonalize(batch['ligand'].num_nodes, edges=edges.T, lamb=sde.lamb[bid],ptr=batch['ligand'].ptr)
    batch.D = D
    batch.P = P
    sde.lamb = D
    pos = P.T @ batch['ligand'].pos
    pos = pos * sde.mu_factor(batch.t[bid])[:, None]

    noise = torch.randn_like(batch["ligand"].pos)
    std = torch.sqrt(sde.var(batch.t[bid]))
    pos += noise * std[:, None]
    batch['ligand'].pos = P @ pos
    batch.std = scatter_mean(std ** 2, bid) ** 0.5

def sample_prior(batch, sigma, harmonic=True):
    if harmonic:
        bid = batch['ligand'].batch
        sde = DiffusionSDE(batch.protein_sigma * sigma)

        edges = batch['ligand', 'bond_edge', 'ligand'].edge_index
        edges = edges[:, edges[0] < edges[1]]  # de-duplicate
        try:
            D, P = HarmonicSDE.diagonalize(batch['ligand'].num_nodes, edges=edges.T, lamb=sde.lamb[bid], ptr=batch['ligand'].ptr)
        except Exception as e:
            print('batch["ligand"].num_nodes', batch['ligand'].num_nodes)
            print("batch['ligand'].size", batch['ligand'].size)
            print("batch['protein'].size", batch['protein'].batch.bincount())
            print(batch.pdb_id)
            raise e
        noise = torch.randn_like(batch["ligand"].pos)
        prior = P @ (noise / torch.sqrt(D)[:, None])
        return prior
    else:
        prior = torch.randn_like(batch["ligand"].pos)
        return prior * sigma

class DiffusionSDE:
    def __init__(self, sigma: torch.Tensor, tau_factor=5.0):
        self.lamb = 1 / sigma**2
        self.tau_factor = tau_factor

    def var(self, t):
        return (1 - torch.exp(-self.lamb * t)) / self.lamb

    def max_t(self):
        return self.tau_factor / self.lamb

    def mu_factor(self, t):
        return torch.exp(-self.lamb * t / 2)
class HarmonicSDE:
    def __init__(self, N=None, edges=[], antiedges=[], a=0.5, b=0.3,
                 J=None, diagonalize=True):
        self.use_cuda = False
        self.l = 1
        if not diagonalize: return
        if J is not None:
            J = J
            self.D, P = np.linalg.eigh(J)
            self.P = P
            self.N = self.D.size
            return


    @staticmethod
    def diagonalize(N, edges=[], antiedges=[], a=1, b=0.3, lamb=0., ptr=None):
        J = torch.zeros((N, N), device=edges.device)  # temporary fix
        for i, j in edges:
            J[i, i] += a
            J[j, j] += a
            J[i, j] = J[j, i] = -a
        for i, j in antiedges:
            J[i, i] -= b
            J[j, j] -= b
            J[i, j] = J[j, i] = b
        J += torch.diag(lamb)
        if ptr is None:
            return torch.linalg.eigh(J)

        Ds, Ps = [], []
        for start, end in zip(ptr[:-1], ptr[1:]):
            D, P = torch.linalg.eigh(J[start:end, start:end])
            Ds.append(D)
            Ps.append(P)
        return torch.cat(Ds), torch.block_diag(*Ps)

    def eigens(self, t):  # eigenvalues of sigma_t
        np_ = torch if self.use_cuda else np
        D = 1 / self.D * (1 - np_.exp(-t * self.D))
        t = torch.tensor(t, device='cuda').float() if self.use_cuda else t
        return np_.where(D != 0, D, t)

    def conditional(self, mask, x2):
        J_11 = self.J[~mask][:, ~mask]
        J_12 = self.J[~mask][:, mask]
        h = -J_12 @ x2
        mu = np.linalg.inv(J_11) @ h
        D, P = np.linalg.eigh(J_11)
        z = np.random.randn(*mu.shape)
        return (P / D ** 0.5) @ z + mu

    def A(self, t, invT=False):
        D = self.eigens(t)
        A = self.P * (D ** 0.5)
        if not invT: return A
        AinvT = self.P / (D ** 0.5)
        return A, AinvT

    def Sigma_inv(self, t):
        D = 1 / self.eigens(t)
        return (self.P * D) @ self.P.T

    def Sigma(self, t):
        D = self.eigens(t)
        return (self.P * D) @ self.P.T

    @property
    def J(self):
        return (self.P * self.D) @ self.P.T

    def rmsd(self, t):
        l = self.l
        D = 1 / self.D * (1 - np.exp(-t * self.D))
        return np.sqrt(3 * D[l:].mean())

    def sample(self, t, x=None, score=False, k=None, center=True, adj=False):
        l = self.l
        np_ = torch if self.use_cuda else np
        if x is None:
            if self.use_cuda:
                x = torch.zeros((self.N, 3), device='cuda').float()
            else:
                x = np.zeros((self.N, 3))
        if t == 0: return x
        z = np.random.randn(self.N, 3) if not self.use_cuda else torch.randn(self.N, 3, device='cuda').float()
        D = self.eigens(t)
        xx = self.P.T @ x
        if center: z[0] = 0; xx[0] = 0
        if k: z[k + l:] = 0; xx[k + l:] = 0

        out = np_.exp(-t * self.D / 2)[:, None] * xx + np_.sqrt(D)[:, None] * z

        if score:
            score = -(1 / np_.sqrt(D))[:, None] * z
            if adj: score = score + self.D[:, None] * out
            return self.P @ out, self.P @ score
        return self.P @ out

    def score_norm(self, t, k=None, adj=False):
        if k == 0: return 0
        l = self.l
        np_ = torch if self.use_cuda else np
        k = k or self.N - 1
        D = 1 / self.eigens(t)
        if adj: D = D * np_.exp(-self.D * t)
        return (D[l:k + l].sum() / self.N) ** 0.5

    def inject(self, t, modes):
        # Returns noise along the given modes
        z = np.random.randn(self.N, 3) if not self.use_cuda else torch.randn(self.N, 3, device='cuda').float()
        z[~modes] = 0
        A = self.A(t, invT=False)
        return A @ z

    def score(self, x0, xt, t):
        # Score of the diffusion kernel
        Sigma_inv = self.Sigma_inv(t)
        mu_t = (self.P * np.exp(-t * self.D / 2)) @ (self.P.T @ x0)
        return Sigma_inv @ (mu_t - xt)

    def project(self, X, k, center=False):
        l = self.l
        # Projects onto the first k nonzero modes (and optionally centers)
        D = self.P.T @ X
        D[k + l:] = 0
        if center: D[0] = 0
        return self.P @ D

    def unproject(self, X, mask, k, return_Pinv=False):
        # Finds the vector along the first k nonzero modes whose mask is closest to X
        l = self.l
        PP = self.P[mask, :k + l]
        Pinv = np.linalg.pinv(PP)
        out = self.P[:, :k + l] @ Pinv @ X
        if return_Pinv: return out, Pinv
        return out

    def energy(self, X):
        l = self.l
        return (self.D[:, None] * (self.P.T @ X) ** 2).sum(-1)[l:] / 2

    @property
    def free_energy(self):
        l = self.l
        return 3 * np.log(self.D[l:]).sum() / 2

    def KL_H(self, t):
        l = self.l
        D = self.D[l:]
        return -3 * 0.5 * (np.log(1 - np.exp(-D * t)) + np.exp(-D * t)).sum(0)

