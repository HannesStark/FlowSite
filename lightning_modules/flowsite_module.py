import csv
import time

import torch, wandb, os, copy
import numpy as np
import pandas as pd
from lightning_modules.general_module import GeneralModule
from utils.diffusion import DiffusionSDE, pyg_add_harmonic_noise, sample_prior

from collections import defaultdict
from datetime import datetime
from torch_scatter import scatter_mean, scatter_add

from utils.featurize import atom_features_list
from utils.logging import lg
from utils.mmcif import RESTYPES

from utils.train_utils import compute_rmsds, supervised_chi_loss, get_blosum_score, get_cooccur_score, \
    get_recovered_aa_angle_loss, get_unnorm_blosum_score
from utils.visualize import  save_trajectory_pdb


def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log

class FlowSiteModule(GeneralModule):
    def __init__(self, args, device, model, train_data=None):
        super().__init__(args, device, model)
        os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
        self.args = args
        self.train_data = train_data

        self.model = model

        self._log = defaultdict(list)
        self.inference_counter = 0

        self.fake_ratio_storage = torch.optim.Adam([torch.tensor([1],dtype=torch.float)],lr=1)
        self.fake_ratio_scheduler = torch.optim.lr_scheduler.ConstantLR(self.fake_ratio_storage,factor=args.fake_ratio_start, total_iters=args.fake_constant_dur)
        if args.fake_ratio_start > 0:
            decay = torch.optim.lr_scheduler.LinearLR(self.fake_ratio_storage, start_factor=args.fake_ratio_start, end_factor=args.fake_ratio_end, total_iters=args.fake_decay_dur)
            self.fake_ratio_scheduler = torch.optim.lr_scheduler.SequentialLR(self.fake_ratio_storage,schedulers=[self.fake_ratio_scheduler, decay], milestones=[args.fake_constant_dur])
    def on_save_checkpoint(self, checkpoint):
        checkpoint['fake_ratio_scheduler'] = self.fake_ratio_scheduler
        checkpoint['fake_ratio_storage'] = self.fake_ratio_storage
    def on_load_checkpoint(self, checkpoint):
        if 'fake_ratio_scheduler' in checkpoint:
            self.fake_ratio_scheduler = checkpoint['fake_ratio_scheduler']
            self.fake_ratio_storage = checkpoint['fake_ratio_storage']
            if self.train_data is not None:
                self.train_data.fake_lig_ratio = self.fake_ratio_storage.param_groups[0]['lr']

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        out = self.general_step_oom_wrapper(batch, batch_idx)
        self.print_iter_log()
        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.stage = "val"
        out = self.general_step_oom_wrapper(batch, batch_idx)
        self.print_iter_log()
        return out

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.stage = "pred"
        out = []
        for i in range(self.args.num_inference):
            batch_copy = copy.deepcopy(batch)
            out.append(self.general_step(batch_copy, batch_idx))
        return torch.tensor(out).mean()

    def print_iter_log(self):
        if (self.trainer.global_step + 1) % self.args.print_freq == 0:
            print('Run name:', self.args.run_name)
            log = self._log
            log = {key: log[key] for key in log if "iter_" in key}

            log = gather_log(log, self.trainer.world_size)
            if self.trainer.is_global_zero:
                lg(str(self.get_log_mean(log)))
                self.log_dict(self.get_log_mean(log), batch_size=1)
                if self.args.wandb:
                    wandb.log(self.get_log_mean(log))
            for key in list(log.keys()):
                if "iter_" in key:
                    del self._log[key]

    def lg(self, key, data):
        log = self._log
        log["iter_" + key].extend(data)
        log[self.stage + "_" + key].extend(data)



    def general_step(self, batch, batch_idx):
        if self.args.use_true_pos: assert self.args.residue_loss_weight == 1, 'You probably do not want the true positions as input if you are predicting them'
        if self.args.residue_loss_weight == 1: assert self.args.use_true_pos, 'If you are not predicting the positions, you probably want to use the true positions as input'
        if self.args.tfn_use_aa_identities == 1: assert self.args.residue_loss_weight == 0, 'If you are training to predict the residues you probably do not want the ground truth residues as input'
        logs = {}
        batch.logs = logs
        start_time = time.time()
        if self.args.debug:
            lg(f'PDBIDs: {batch.pdb_id}')
            lg(f'Id of ligands lig_choice_id: {batch["ligand"].lig_choice_id}')
            lg(f'Lig name: {batch["ligand"].name}')
            lg(f'fake_lig_id: {batch["protein"].fake_lig_id}')
        if batch['ligand'].size.sum() == len(batch['ligand'].size):
            lg(f'All ligands had size 1 in the batch. Skipping the batch with {batch.pdb_id} with ligand {batch["ligand"].name}')
            return None
        #with open('batch.txt', 'w') as f:
        #    f.write(str(sorted(batch.__dict__.items(), key=lambda x: x[0])))
        batch['ligand'].shadow_pos = batch['ligand'].pos.clone()

        if self.args.flow_matching:
            x0 = sample_prior(batch, self.args.prior_scale, harmonic=not self.args.gaussian_prior)
            x1 = batch['ligand'].shadow_pos
            eps = torch.randn_like(batch["ligand"].pos)
            batch.t01 = torch.rand(batch.num_graphs, device=self.device)
            batch.normalized_t = torch.square(batch.t01)
            batch.t = batch.t01
            batch.std = batch.protein_sigma
            t01 = batch.t01[batch["ligand"].batch, None]
            mu_t = t01 * x1 + (1 - t01) * x0
            x_t = mu_t + eps * self.args.flow_matching_sigma
            batch['ligand'].pos = x_t
        else:
            try:
                start_time2 = time.time()
                pyg_add_harmonic_noise(self.args, batch)
                batch.logs['add_harmonic_noise_time'] = time.time() - start_time2
            except Exception as e:
                lg("Error adding noise")
                lg(batch.pdb_id)
                raise e
        bid = batch["ligand"].batch
        if self.args.use_true_pos:
            batch['ligand'].pos = batch['ligand'].shadow_pos.clone()

        # forward pass
        try:
            if (self.args.self_condition_inv or self.args.self_condition_x) and np.random.rand() < self.args.self_condition_ratio:

                with torch.no_grad():
                    if self.args.self_condition_bit:
                        batch.self_condition_bit = torch.zeros((len(batch.pdb_id),1), dtype=torch.float32, device=self.device)
                    if self.args.standard_style_self_condition_inv:
                        original_input_feat = batch['protein'].input_feat.clone()
                        batch['protein'].original_feat = original_input_feat.clone()
                    res_pred, pos_list, angles = self.model(copy.deepcopy(batch), x_self=sample_prior(batch, self.args.prior_scale , harmonic=not self.args.gaussian_prior) if self.args.self_condition_x else None)
                if self.args.self_condition_inv:
                    del batch['protein'].input_feat
                    batch['protein'].input_feat = res_pred if self.args.self_condition_inv_logits else torch.argmax(res_pred, dim=1)[:, None]
                    if self.args.self_condition_bit:
                        batch.self_condition_bit = torch.ones((len(batch.pdb_id),1), dtype=torch.float32, device=self.device)
                    if self.args.standard_style_self_condition_inv:
                        del batch['protein'].original_feat
                        batch['protein'].original_feat = original_input_feat.clone()
                res_pred, pos_list, angles = self.model(batch, x_self =copy.deepcopy(pos_list[-1].detach()) if self.args.self_condition_x else None)
            else:
                if self.args.standard_style_self_condition_inv:
                    batch['protein'].original_feat = batch['protein'].input_feat.clone()
                if self.args.self_condition_bit:
                    batch.self_condition_bit = torch.zeros((len(batch.pdb_id),1), dtype=torch.float32, device=self.device)
                res_pred, pos_list, angles = self.model(copy.deepcopy(batch), x_self=sample_prior(batch, self.args.prior_scale , harmonic=not self.args.gaussian_prior) if self.args.self_condition_x else None)

        except Exception as e:
            lg("Error forward pass")
            lg(batch.pdb_id)
            raise e

        start_time2 = time.time()
        training_target = batch["ligand"].shadow_pos if not self.args.velocity_prediction else batch["ligand"].shadow_pos - x0
        pos_losses = torch.nn.functional.l1_loss(training_target[None,:,:].expand(len(pos_list),-1,-1), pos_list, reduction='none') if self.args.l1_loss else torch.square(training_target - pos_list)
        if not self.args.flow_matching: pos_losses = pos_losses / (batch.std[bid, None] ** 2)
        pos_losses = scatter_mean(pos_losses.mean(-1), bid, -1)
        if self.args.clamp_loss is not None:
            pos_losses = torch.clamp(pos_losses, max=self.args.clamp_loss)
        aux_loss = pos_losses[1:].mean(0)
        pos_loss = pos_losses[-1]
        pos_loss =  (1 - self.args.aux_weight) * pos_loss
        if not self.args.use_true_pos:
            pos_loss = pos_loss + self.args.aux_weight * aux_loss

        discrete_loss, designable_loss, all_res_loss, accuracy, all_res_accuracy, allmean_accuracy, allmean_all_res_accuracy, blosum_score, all_res_blosum_score, cooccur_score, all_res_cooccur_score, unnorm_blosum_score, all_res_unnorm_blosum_score = self.get_discrete_metrics(batch, res_pred)

        loss = torch.zeros_like(pos_loss, device=self.device, dtype=torch.float)
        if self.args.residue_loss_weight > 0 and self.args.pos_only_epochs <= self.current_epoch:
            loss += discrete_loss * self.args.residue_loss_weight
        if not self.args.use_true_pos:
            loss += pos_loss * (1 - self.args.aux_weight)
        if self.args.num_angle_pred > 0:
            angle_loss = supervised_chi_loss(batch, angles, angles_idx_s=11-self.args.num_angle_pred)
            loss += angle_loss * self.args.angle_loss_weight
        else:
            angle_loss = torch.tensor(0.0)




        with torch.no_grad():
            self.lg("num_res", (batch.protein_size.cpu().numpy()))
            self.lg("batch_idx", [batch_idx]*len(batch.pdb_id))
            self.lg("num_ligs", (batch.num_ligs.cpu().numpy()))
            self.lg("num_designable", ((scatter_add(batch['protein'].designable_mask.int(), batch['protein'].batch)).cpu().numpy()))
            self.lg("lig_size", (batch['ligand'].size).cpu().numpy())
            self.lg("name", batch.pdb_id)
            self.lg("rec_sigma", batch.protein_sigma.cpu().numpy())
            self.lg("norm_t", batch.normalized_t.cpu().numpy())
            self.lg("t", batch.t.cpu().numpy())
            self.lg("sigma", batch.std.cpu().numpy())
            self.lg("aux_loss", aux_loss.cpu().numpy())
            self.lg("pos_loss", pos_loss.cpu().numpy())
            self.lg("angle_loss", [angle_loss.cpu().numpy()])
            self.lg("loss", loss.cpu().numpy())
            lowT = torch.where(batch.normalized_t <= 2 / 20)[0]
            self.lg('lowT_pos_loss', pos_loss[lowT].cpu().numpy())
            self.lg('lowT_aux_loss', aux_loss[lowT].cpu().numpy())
            self.lg('lowT_loss', loss[lowT].cpu().numpy())
            self.lg('lowT_t', batch.t[lowT].cpu().numpy())
            if self.args.residue_loss_weight > 0:
                self.lg('allT_designable_loss', designable_loss.cpu().numpy())
                self.lg('allT_all_res_loss', all_res_loss.cpu().numpy())
                self.lg('allT_accuracy', accuracy.cpu().numpy())
                self.lg('allT_all_res_accuracy', all_res_accuracy.cpu().numpy())
                self.lg('allT_blosum_score', blosum_score.cpu().numpy())
                self.lg('allT_all_res_blosum_score', all_res_blosum_score.cpu().numpy())
                self.lg('allT_unnorm_blosum_score', unnorm_blosum_score.cpu().numpy())
                self.lg('allT_all_res_unnorm_blosum_score', all_res_unnorm_blosum_score.cpu().numpy())
                self.lg('allT_cooccur_score', cooccur_score.cpu().numpy())
                self.lg('allT_all_res_cooccur_score', all_res_cooccur_score.cpu().numpy())
                self.lg('lowT_designable_loss', designable_loss[lowT].cpu().numpy())
                self.lg('lowT_all_res_loss', all_res_loss[lowT].cpu().numpy())
                self.lg('lowT_accuracy', accuracy[lowT].cpu().numpy())
                self.lg('lowT_all_res_accuracy', all_res_accuracy[lowT].cpu().numpy())
                self.lg('lowT_blosum_score', blosum_score[lowT].cpu().numpy())
                self.lg('lowT_all_res_blosum_score', all_res_blosum_score[lowT].cpu().numpy())
                self.lg('lowT_cooccur_score', cooccur_score[lowT].cpu().numpy())
                self.lg('lowT_all_res_cooccur_score', all_res_cooccur_score[lowT].cpu().numpy())

            if (self.stage == "val" and (self.trainer.current_epoch + 1) % self.args.check_val_every_n_epoch == 0) or self.stage == "pred":
                if self.args.use_true_pos:
                    res_pred, pos_list, angles = self.model(batch)
                    x1 = pos_list[-1]
                    x1_out = pos_list[-1]
                else:
                    if self.args.flow_matching:
                        x1_out, x1, res_pred = self.flow_match_inference(batch, batch_idx)
                    else:
                        x1_out, x1, res_pred = self.harmonic_inference(batch, batch_idx)

                angle_loss = supervised_chi_loss(batch, angles, angles_idx_s=11 - self.args.num_angle_pred) if self.args.num_angle_pred > 0 else torch.tensor(0.0)
                recovered_aa_angle_loss = get_recovered_aa_angle_loss(copy.deepcopy(batch), angles, res_pred, angles_idx_s=11-self.args.num_angle_pred) if self.args.num_angle_pred > 0 else torch.tensor(0.0)

                rmsd, centroid_rmsd, kabsch_rmsd = compute_rmsds(batch["ligand"].shadow_pos, x1, batch )
                rmsd_out, centroid_rmsd_out, kabsch_rmsd_out = compute_rmsds(batch["ligand"].shadow_pos, x1_out, batch)

                self.log_3D_metrics(rmsd, centroid_rmsd, kabsch_rmsd, suffix="")
                self.log_3D_metrics(rmsd_out, centroid_rmsd_out, kabsch_rmsd_out, suffix="_out")

                if self.args.residue_loss_weight > 0:
                    discrete_loss, designable_loss, all_res_loss, accuracy, all_res_accuracy, allmean_accuracy, allmean_all_res_accuracy, blosum_score, all_res_blosum_score, cooccur_score, all_res_cooccur_score, unnorm_blosum_score, all_res_unnorm_blosum_score = self.get_discrete_metrics(batch, res_pred)
                    self.lg('designable_loss', designable_loss.cpu().numpy())
                    self.lg('all_res_loss', all_res_loss.cpu().numpy())
                    self.lg('accuracy', accuracy.cpu().numpy())
                    self.lg('inf_angle_loss', [angle_loss.cpu().numpy()])
                    self.lg('inf_recovered_aa_angle_loss', [recovered_aa_angle_loss.cpu().numpy()])
                    self.lg('all_res_accuracy', all_res_accuracy.cpu().numpy())
                    self.lg('blosum_score', blosum_score.cpu().numpy())
                    self.lg('all_res_blosum_score', all_res_blosum_score.cpu().numpy())
                    self.lg('unnorm_blosum_score', unnorm_blosum_score.cpu().numpy())
                    self.lg('all_res_unnorm_blosum_score', all_res_unnorm_blosum_score.cpu().numpy())
                    self.lg('cooccur_score', cooccur_score.cpu().numpy())
                    self.lg('all_res_cooccur_score', all_res_cooccur_score.cpu().numpy())
                    self.lg('allmean_accuracy', allmean_accuracy.cpu().numpy())
                    self.lg('allmean_all_res_accuracy', allmean_all_res_accuracy.cpu().numpy())

            batch.logs['metric_calculation_time'] = time.time() - start_time2
            batch.logs['general_step_time'] = time.time() - start_time
            for k, v in batch.logs.items():
                self.lg(k, np.array([v]))
            batch.logs = {}
        return loss.mean()

    def log_3D_metrics(self, rmsd, centroid_rmsd, kabsch_rmsd, suffix=""):
        self.lg(f"rmsd{suffix}", rmsd)
        self.lg(f"rmsd<1{suffix}", (rmsd < 1))
        self.lg(f"rmsd<2{suffix}", (rmsd < 2))
        self.lg(f"rmsd<5{suffix}", (rmsd < 5))

        self.lg(f"centroid_rmsd{suffix}", centroid_rmsd)
        self.lg(f"centroid_rmsd<1{suffix}", (centroid_rmsd < 1))
        self.lg(f"centroid_rmsd<2{suffix}", (centroid_rmsd < 2))
        self.lg(f"centroid_rmsd<5{suffix}", (centroid_rmsd < 5))

        self.lg(f"kabsch_rmsd{suffix}", kabsch_rmsd)
        self.lg(f"kabsch_rmsd<1{suffix}", (kabsch_rmsd < 1))
        self.lg(f"kabsch_rmsd<2{suffix}", (kabsch_rmsd < 2))
        self.lg(f"kabsch_rmsd<5{suffix}", (kabsch_rmsd < 5))


    @torch.no_grad()
    def flow_match_inference(self, batch, batch_idx=None, production_mode = False):
        # be careful, the meaning of x0 and x1 is reversed here in flow matching compared to diffusion

        x0 = sample_prior(batch, self.args.prior_scale , harmonic=not self.args.gaussian_prior)
        if self.args.self_condition_inv:
            batch['protein'].input_feat = batch['protein'].feat * 0 + len(atom_features_list['residues_canonical'])
        x_self = sample_prior(batch, self.args.prior_scale , harmonic=not self.args.gaussian_prior) if self.args.self_condition_x else None
        if self.args.self_condition_bit:
            batch.self_condition_bit = torch.zeros((len(batch.pdb_id), 1), dtype=torch.float32, device=self.device)

        # at t=0, we are at x0
        t_span = torch.linspace(0, 1, self.args.num_integration_steps, device=self.device)
        t, dt = t_span[0], t_span[1] - t_span[0]

        sol = [x0]
        model_pred = [x0]
        xt = x0
        steps = 1
        while steps <= len(t_span) - 1:
            batch["ligand"].pos = xt
            batch.t01 = t.expand(len(batch.pdb_id)).to(self.device)
            res_pred, pos_list, angles = self.model(batch, x_self=x_self)
            x1_pred = pos_list[-1]
            vt = x1_pred - x0 if not self.args.velocity_prediction else x1_pred
            xt = xt + dt * vt
            t = t + dt

            if self.args.self_condition_inv:
                if self.args.self_condition_inv_logits:
                    batch['protein'].input_feat = res_pred
                else:
                    batch['protein'].input_feat = torch.argmax(res_pred, dim=1)[:,None]
            if self.args.self_condition_x:
                x_self = x1_pred
            if self.args.self_condition_bit:
                batch.self_condition_bit = torch.ones((len(batch.pdb_id), 1), dtype=torch.float32, device=self.device)

            sol.append(xt)
            model_pred.append(x1_pred)
            if steps < len(t_span) - 1: dt = t_span[steps + 1] - t
            steps += 1

        if self.args.save_inference and (batch_idx == 0 or self.args.save_all_batches) and self.inference_counter % self.args.inference_save_freq == 0 or self.stage == "pred" and self.args.save_inference and self.args.save_all_batches:
            self.inference_counter = 0
            save_trajectory_pdb(self.args, batch, sol, model_pred, extra_string=f'{self.stage}_{self.trainer.global_step}globalStep', production_mode=production_mode, out_dir=os.path.join(self.args.out_dir, 'structures') if production_mode else None)
        return x1_pred, xt, res_pred

    @torch.no_grad()
    def harmonic_inference(self, batch, batch_idx=None):
        sde = DiffusionSDE(batch.protein_sigma * self.args.prior_scale)
        device = batch.protein_sigma.device

        times01 = torch.linspace(1, 0, self.args.num_integration_steps + 1, device=device)[:, None]
        ts = times01 ** 2 * sde.max_t()
        bid = batch["ligand"].batch

        noise = torch.randn_like(batch["ligand"].pos)
        xt = batch.P @ (noise / torch.sqrt(batch.D)[:,None])
        if self.args.self_condition_inv:
            batch['protein'].input_feat = batch['protein'].feat * 0 + len(atom_features_list['residues_canonical'])
        
        lamb = batch.D
        sol = [xt]
        model_pred = [xt]
        very_first_x0 = None
        for idx, (t, s, t01) in enumerate(zip(ts[:-1], ts[1:], times01[1:])):
            batch["ligand"].pos = xt
            if self.args.correct_time_condition:
                batch.t01 = copy.deepcopy(t01).to(device).expand(len(batch.pdb_id))
                batch.normalized_t = s.expand(len(batch.pdb_id))
            try:
                res_pred, pos_list, angles = self.model(batch)
            except Exception as e:
                lg("Error in inference")
                lg(batch.pdb_id)
                raise e

            x0 = pos_list[-1]
            if very_first_x0 is None:
                x0 = pos_list[-1]
            very_first_x0 = x0.clone()

            t, s = t[bid], s[bid]
            x0_coeff = ( (1 - torch.exp(-lamb * (t - s))) / (1 - torch.exp(-lamb * t)) * torch.exp(-lamb * s / 2) )
            xt_coeff = ((1 - torch.exp(-lamb * s)) / (1 - torch.exp(-lamb * t)) * torch.exp(-lamb * (t - s) / 2))

            xt = x0_coeff[:,None] * (batch.P.T @ x0) + xt_coeff[:,None] * (batch.P.T @ xt)
            var = ((1 / lamb) * (1- torch.exp(-lamb * s)- torch.exp(-lamb * (t - s))+ torch.exp(-lamb * t))/ (1 - torch.exp(-lamb * t)))
            xt += torch.randn_like(xt) * torch.sqrt(var)[:, None]
            xt = batch.P @ xt
            if self.args.self_condition_inv_logits:
                batch['protein'].input_feat = res_pred
            else:
                batch['protein'].input_feat = torch.argmax(res_pred, dim=1)[:, None]

            sol.append(xt)
            model_pred.append(x0)

        if self.args.save_inference and (batch_idx == 0 or self.args.save_all_batches) and self.inference_counter % self.args.inference_save_freq == 0 or self.stage == "pred" and self.args.save_inference and self.args.save_all_batches:
            self.inference_counter = 0
            save_trajectory_pdb(self.args, batch, sol, model_pred, extra_string=f'{self.stage}_{self.trainer.global_step}globalStep')
        return x0, xt, res_pred


    def get_discrete_metrics(self, batch, res_pred):
        prot_bid = batch["protein"].batch
        res_true = batch['protein'].feat[:, 0].view(-1)
        to_predict_mask = batch['protein'].designable_mask
        pocket_res_pred = res_pred[to_predict_mask]
        pocket_res_true = res_true[to_predict_mask]
        all_res_loss = torch.nn.functional.cross_entropy(res_pred, res_true, reduction='none')
        all_res_loss = scatter_mean(all_res_loss, prot_bid, -1)
        designable_loss = torch.nn.functional.cross_entropy(pocket_res_pred, pocket_res_true, reduction='none')
        designable_loss = scatter_mean(designable_loss, prot_bid[to_predict_mask], -1)
        discrete_loss = all_res_loss if self.args.num_all_res_train_epochs > self.current_epoch else designable_loss
        all_res_accuracy = scatter_mean((torch.argmax(res_pred, dim=1) == res_true).float(), prot_bid, dim=-1)
        accuracy = scatter_mean((torch.argmax(pocket_res_pred, dim=1) == pocket_res_true).float(), prot_bid[to_predict_mask], dim=-1)
        allmean_accuracy = ((torch.argmax(pocket_res_pred, dim=1) == pocket_res_true).float().flatten())
        allmean_all_res_accuracy = ((torch.argmax(pocket_res_pred, dim=1) == pocket_res_true).float().flatten())
        all_res_blosum_score = get_blosum_score(torch.argmax(res_pred, dim=1),res_true, prot_bid)
        blosum_score = get_blosum_score(torch.argmax(pocket_res_pred, dim=1),pocket_res_true, prot_bid[to_predict_mask])
        all_res_unnorm_blosum_score = get_unnorm_blosum_score(torch.argmax(res_pred, dim=1), res_true, prot_bid)
        unnorm_blosum_score = get_unnorm_blosum_score(torch.argmax(pocket_res_pred, dim=1), pocket_res_true,prot_bid[to_predict_mask])
        all_res_cooccur_score = get_cooccur_score(torch.argmax(res_pred, dim=1), res_true, prot_bid)
        cooccur_score = get_cooccur_score(torch.argmax(pocket_res_pred, dim=1), pocket_res_true,prot_bid[to_predict_mask])

        return discrete_loss, designable_loss, all_res_loss, accuracy, all_res_accuracy, allmean_accuracy, allmean_all_res_accuracy, blosum_score, all_res_blosum_score, cooccur_score, all_res_cooccur_score, unnorm_blosum_score, all_res_unnorm_blosum_score



    def get_log_mean(self, log):
        out = {}
        out['trainer/global_step'] = float(self.trainer.global_step)
        out['epoch'] = float(self.trainer.current_epoch)
        if self.train_data is not None:
            out['fake_lig_ratio'] = float(self.train_data.fake_lig_ratio)

        temporary_log = {}
        aggregated_log = {}
        if self.stage == "pred" and not 'iter_name' in list(log.keys()):
            for key, value in log.items():
                if isinstance(value, list) and len(value) == len(log['pred_num_res']):
                    aggregated_list = []
                    for i in range(max(log['pred_batch_idx']) + 1):
                        values_for_batch = np.array(value)[np.where(np.array(log['pred_batch_idx']) == i)[0]]
                        aggregated_list.append(values_for_batch.reshape(self.args.num_inference, -1))
                    temporary_log[key] = np.concatenate(aggregated_list, axis=1)
                else:
                    aggregated_log[key] = value

            for key, value in temporary_log.items():
                if 'rmsd' in key and not '_std' in key:
                    pass
                top10_rmsd_order = np.argsort(temporary_log['pred_rmsd'][:10], axis=0)
                top5_rmsd_order = np.argsort(temporary_log['pred_rmsd'][:5], axis=0)
                aggregated_log[key + '_msdTop10'] = np.take_along_axis(temporary_log[key][:10], top10_rmsd_order, axis=0)[0]
                aggregated_log[key + '_msdTop5'] = np.take_along_axis(temporary_log[key][:5], top5_rmsd_order, axis=0)[0]
                if self.args.residue_loss_weight > 0:
                    top10_aar_order = np.argsort(temporary_log['pred_accuracy'][:10], axis=0)[::-1]
                    top5_aar_order = np.argsort(temporary_log['pred_accuracy'][:5], axis=0)[::-1]
                    aggregated_log[key + '_aarTop10'] = np.take_along_axis(temporary_log[key][:10], top10_aar_order, axis=0)[:10][0]
                    aggregated_log[key + '_aarTop5'] = np.take_along_axis(temporary_log[key][:5], top5_aar_order, axis=0)[:5][0]
                try:
                    aggregated_log[key] = log[key]
                    aggregated_log[key + '_std'] = temporary_log[key].std(axis=0)
                except:
                    pass
        else:
            aggregated_log = log

        for key, value in aggregated_log.items():
            if 'rmsd' in key and not '_std' in key:
                out[key + '_median'] = np.median(np.array(value)[np.isfinite(value)])
        for key in aggregated_log:
            try:
                out[key] = np.mean(aggregated_log[key])
            except:
                pass
        return out

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.stage = "pred"
        logs = {}
        batch.logs = logs
        prot_bid = batch['protein'].batch.cpu()
        full_prot_bid = batch['full_protein'].batch.cpu()
        designed_seqs = []
        designed_res_list = []
        logit_seqs = []
        logit_res_list = []
        resid_chainid_list = []
        for i in range(self.args.num_inference):
            batch_ = copy.deepcopy(batch)
            if self.args.flow_matching:
                x1_out, x1, res_pred = self.flow_match_inference(batch_, batch_idx, production_mode=True)
            else:
                x1_out, x1, res_pred = self.harmonic_inference(batch_, batch_idx)
            full_designed_mask = torch.logical_and(batch['full_protein'].designable_mask, batch['full_protein'].pocket_mask)
            assert full_designed_mask.sum() == batch['protein'].designable_mask.sum()

            # get the full sequence with the designed residues inserted
            designed_seq = batch['full_protein'].aatype_num.clone()
            designed_res = torch.argmax(res_pred[batch['protein'].designable_mask], dim=1)
            if (designed_res == len(RESTYPES)).any():
                print('Warning: miscallenaeous token predicted, not including that in the redesigned sequence')
            designed_res[designed_res == len(RESTYPES)] = batch['protein'].aatype_num[batch['protein'].designable_mask][designed_res == len(RESTYPES)] # do not redisign residues if miscallenaeous token is predicted
            designed_seq[full_designed_mask] = designed_res
            designed_seqs.append([np.array(RESTYPES)[designed_seq.cpu().numpy()][torch.where(full_prot_bid == i)] for i in range(len(batch.pdb_id))])

            # get the full sequence of logitidences/logits with the designed residues inserted
            logit_seq = np.zeros((len(batch['full_protein'].pos), len(atom_features_list['residues_canonical'])))
            logit_seq[full_designed_mask.cpu()] = res_pred[batch['protein'].designable_mask].cpu().numpy()
            logit_seqs.append([logit_seq[torch.where(full_prot_bid == i)] for i in range(len(batch.pdb_id))])

            # get the individual designed residues / logitidences
            designed_res_list.append([np.array(RESTYPES)[designed_res.cpu().numpy()][torch.where(prot_bid[batch['protein'].designable_mask.cpu()] == i)] for i in range(len(batch.pdb_id))])
            logit_res_list.append([res_pred[batch['protein'].designable_mask].cpu().numpy()[torch.where(prot_bid[batch['protein'].designable_mask.cpu()] == i)] for i in range(len(batch.pdb_id))])
            chain_id_letters = np.array(list('ACDEFGHIKLMNPQRSTVWY'))[batch['protein'].pdb_chain_id[batch['protein'].designable_mask.cpu()].cpu().numpy()]
            res_id_chars = batch['protein'].pdb_res_id[batch['protein'].designable_mask].cpu().numpy().astype(str)
            resid_chainid_list.append([np.char.add(chain_id_letters,res_id_chars)[torch.where(prot_bid[batch['protein'].designable_mask.cpu()] == i)] for i in range(len(batch.pdb_id))])

        # transpose list of lists
        designed_seqs = list(map(list, zip(*designed_seqs)))
        logit_seqs = list(map(list, zip(*logit_seqs)))
        designed_res_list = list(map(list, zip(*designed_res_list)))
        logit_res_list = list(map(list, zip(*logit_res_list)))
        resid_chainid_list = list(map(list, zip(*resid_chainid_list)))

        # write output to files
        for i in range(batch.num_graphs):
            sample_out_dir = os.path.join(self.args.out_dir, batch.pdb_id[i])
            os.makedirs(sample_out_dir, exist_ok=True)
            with open(os.path.join(sample_out_dir, 'full_sequences.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sequence'])
                writer.writerows([[''.join(seq)] for seq in designed_seqs[i]])
            with open(os.path.join(sample_out_dir, 'designed_residues.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(resid_chainid_list[i][0])
                writer.writerows(designed_res_list[i])
            np.save(os.path.join(sample_out_dir, 'designed_logits.npy'), np.stack(logit_res_list[i]))
            np.save(os.path.join(sample_out_dir, 'full_sequence_logits.npy'), np.stack(logit_seqs[i]))
        return designed_seqs, designed_res_list, logit_seqs, logit_res_list, resid_chainid_list

    def on_predict_epoch_end(self):
        log = self._log
        log = {key: log[key] for key in log if "inf_" in key}
        lg(str(self.get_log_mean(log)))
        time = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        path = os.path.join(os.environ["MODEL_DIR"], f"inf_{time}.csv")
        pd.DataFrame(log).to_csv(path)
        lg("Finished and saved all predictions to: ", self.args.out_dir)

    def on_train_epoch_end(self):
        self.on_epoch_end("train")

        # for training with fake ligands
        self.fake_ratio_scheduler.step()
        self.train_data.fake_lig_ratio = self.fake_ratio_storage.param_groups[0]['lr']

    def on_validation_epoch_end(self):
        self.on_epoch_end("val")

    def on_test_epoch_end(self):
        self.on_epoch_end("pred")

    def on_epoch_end(self, stage):
        log = self._log
        log = {key: log[key] for key in log if f"{stage}_" in key}
        log = gather_log(log, self.trainer.world_size)
        log['invalid_grads_per_epoch'] = self.num_invalid_gradients
        if self.trainer.is_global_zero:
            print('Run name:', self.args.run_name)
            lg(str(self.get_log_mean(log)))
            self.log_dict(self.get_log_mean(log), batch_size=1)
            if self.args.wandb:
                wandb.log(self.get_log_mean(log), step=self.trainer.global_step)
            path = os.path.join(os.environ["MODEL_DIR"], f"{stage}_{self.trainer.current_epoch}.csv")
            log_clone = copy.deepcopy(log)
            for key in list(log_clone.keys()):
                if f"{stage}_" in key and ('lowT' in key or '_time' in key) or 'allmean_' in key or 'angle_loss' in key or 'invalid_grads_per_epoch' in key:
                    del log_clone[key]
            pd.DataFrame(log_clone).to_csv(path)
        for key in list(log.keys()):
            if f"{stage}_" in key:
                del self._log[key]
        self.num_invalid_gradients = 0


