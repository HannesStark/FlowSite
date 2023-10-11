# <span style="color:#016fff;">FlowSite</span> and <span style="color:#f99e01;">HarmonicFlow</span>

### [Paper on arXiv](https://arxiv.org/abs/comingsoon)

Code of FlowSite and HarmonicFlow. HarmonicFlow generates binding structures for single ligands or "multi-ligands" (multiple small molecules and ions bound to the same pocket). 
FlowSite builds on HarmonicFlow and generates residues types for binding sites to bind a specific (multi-)ligand.

**<span style="color:#f99e01;">For using FlowSite to design your own binding sites:</span>** 
I am currently writing a better user interface and training a more "production ready" model on a mixture of pdbbind and Binding MOAD after pretraining with fake ligands on all of PDB and using backbone noise. <br>
If you want, you can put your email here and I will notify you once that is done: https://forms.gle/KzfiaoLij6YHPy6R8

Feel free to reach out about any questions! [hstark@mit.edu](hstark@mit.edu)

FlowSite generative process:
![Alt Text](images/figure1.png)
HarmonicFlow multi-ligand structure generation gif:
![Alt Text](images/multi-lig-video.gif)


## Setup Environment

We will set up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).
This is an example for how to set up a working conda environment to run the code. Make sure that the pytorch and pytorch-geometric versions you use are compatible with each other. Also, if you do not use a GPU, rather use `cpu` instead of `cu121` in the last line (also make sure that you use the correct cuda version if you do have a gpu).

    conda create -c conda-forge -n flowsite rdkit python
    pip install torch torchvision torchaudio
    pip install torch_geometric
    pip install pyyaml wandb biopython spyrmsd einops biopandas plotly prody tqdm lightning imageio
    pip install e3nn

    pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

Alternatively there is also an `environment.yml` file that you can use with `conda env create -f environment.yml`.

## Design your Binding sites

**<span style="color:#f99e01;">For using FlowSite to design your own binding sites:</span>** 
I am currently writing a better user interface and training a more "production ready" model on a mixture of pdbbind and Binding MOAD after pretraining with fake ligands on all of PDB and using backbone noise. <br>
If you want, you can put your email here and I will notify you once that is done: https://forms.gle/KzfiaoLij6YHPy6R8

Also feel free to reach out about any feedback, ideas or what you would like to UI to be! [hstark@mit.edu](hstark@mit.edu)

## Retrain FlowSite/HarmonicFlow or run trained model on test set

### Dataset

The files in `index` contain the names for the sequence and time splits of pdbbind and moad.

To obtain the pdbbind data:
1. download it from [zenodo](https://zenodo.org/record/6408497)
2. unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`

To obtain the binding MOAD data:
1. download it from [here](https://bindingmoad.org/)
2. unzip the directory and place it into `data` such that you have the path `data/moad/BindingMOAD_2020`
3. run `python moad_unit2pdb.py`

### Running on the test set with trained models

Run on the test set using trained model weights like follows. You can download the weights for the following two examples here: [https://drive.google.com/file/d/1QGQ6U3BDlEZ682yv7dLo2wbuulOPArSY/view?usp=sharing](https://drive.google.com/file/d/1QGQ6U3BDlEZ682yv7dLo2wbuulOPArSY/view?usp=sharing)
```
CUDA_VISIBLE_DEVICES="0" python -m train --run_name test_HarmonicFlow_seqSimSplit_RadPock --wandb --run_test --checkpoint pocket_gen/0np1y80l/checkpoints/best.ckpt --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 1 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities --self_condition_x 

CUDA_VISIBLE_DEVICES="0" python -m train --run_name test_FlowSite_pdbbind_seqSimSplit --wandb --run_test --checkpoint pocket_gen/89b8ojq8/checkpoints/best.ckpt --batch_size 16 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --pocket_residue_cutoff 12 --clamp_loss 10 --epochs 150 --num_inference 10 --layer_norm --gradient_clip_val 1 --save_inference --check_nan_grads --num_all_res_train_epochs 100000 --fake_constant_dur 100000 --fake_decay_dur 10 --fake_ratio_start 0.2 --fake_ratio_end 0.2 --residue_loss_weight 0.2 --use_tfn --use_inv --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --num_angle_pred 5 --ns 32 --nv 8 --batch_norm --self_condition_x --self_condition_inv --no_tfn_self_condition_inv --self_fancy_init
```

### Retrain HarmonicFlow and FlowSite
If training stopped for some reason you can restart by using `--checkpoint wandbProjectName/IDofTheRun/last.ckpt`. 

FlowSite PDBBind and Binding MOAD:
```
CUDA_VISIBLE_DEVICES="0" python -m train --run_name FlowSite_pdbbind --wandb --batch_size 16 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --pocket_residue_cutoff 12 --clamp_loss 10 --epochs 150 --num_inference 5 --layer_norm --gradient_clip_val 1 --save_inference --check_nan_grads --fake_constant_dur 100000 --fake_decay_dur 10 --fake_ratio_start 0.2 --fake_ratio_end 0.2 --residue_loss_weight 0.2 --use_tfn --use_inv --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --num_angle_pred 5 --ns 32 --nv 8 --batch_norm --self_condition_x --self_condition_inv --no_tfn_self_condition_inv --standard_style_self_condition_inv --self_fancy_init

CUDA_VISIBLE_DEVICES="0" python -m train --run_name FlowSite_moad --wandb --batch_size 16 --train_split_path index/moad_mmseqs_30sim_train --val_split_path index/moad_mmseqs_30sim_val --predict_split_path index/moad_mmseqs_30sim_test --data_dir data/moad/moad_prepared --data_source moad --biounit1_only --pocket_residue_cutoff 12 --clamp_loss 10 --epochs 150 --num_inference 5 --layer_norm --gradient_clip_val 1 --save_inference --check_nan_grads --fake_constant_dur 100000 --fake_decay_dur 10 --fake_ratio_start 0.2 --fake_ratio_end 0.2 --residue_loss_weight 0.2 --use_tfn --use_inv --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --num_angle_pred 5 --ns 32 --nv 8 --batch_norm --self_condition_x --self_condition_inv --min_num_contacts 2 --no_tfn_self_condition_inv --self_fancy_init --correct_moad_lig_selection --check_val_every_n_epoch 3 --standard_style_self_condition_inv
```

HarmonicFlow on PDBBind with residue identities:
```
CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_pdbbind_seqSimSplit_radPock --wandb --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 1 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities --self_condition_x

CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_pdbbind_seqSimSplit_distPock --wandb --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 1 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --tfn_use_aa_identities --self_condition_x

CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_pdbbind_timesplit_distPock --wandb --batch_size 4 --train_split_path index/timesplit_no_lig_overlap_train --val_split_path index/timesplit_no_lig_overlap_val --predict_split_path index/timesplit_test --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 5 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --tfn_use_aa_identities --self_condition_x

CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_pdbbind_timesplit_radPock --wandb --batch_size 4 --train_split_path index/timesplit_no_lig_overlap_train --val_split_path index/timesplit_no_lig_overlap_val --predict_split_path index/timesplit_test --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 5 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities --self_condition_x
```

HarmonicFlow on Binding MOAD with residue identities:
```
CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_moad_RadPock --wandb --batch_size 4 --train_split_path index/moad_mmseqs_30sim_train --val_split_path index/moad_mmseqs_30sim_val --predict_split_path index/moad_mmseqs_30sim_test --data_dir data/moad/moad_prepared --data_source moad --biounit1_only --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --num_all_res_train_epochs 100000 --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 1 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities --self_condition_x --correct_moad_lig_selection --flow_matching --flow_matching_sigma 0.5

CUDA_VISIBLE_DEVICES="0" python -m train --run_name EigenFold_moad_RadPock --wandb --batch_size 4 --train_split_path index/moad_mmseqs_30sim_train --val_split_path index/moad_mmseqs_30sim_val --predict_split_path index/moad_mmseqs_30sim_test --data_dir data/moad/moad_prepared --data_source moad --biounit1_only --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --num_all_res_train_epochs 100000 --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 1 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities --self_condition_x --correct_moad_lig_selection
```


Binding Site Recovery PDBBind
```
CUDA_VISIBLE_DEVICES="0" python -m train --run_name noLig_pdbbind --wandb --batch_size 16 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --pocket_residue_cutoff 12 --clamp_loss 10 --batch_norm --gradient_clip_val 1 --save_inference --check_nan_grads --residue_loss_weight 1 --use_true_pos --use_inv --ignore_lig

CUDA_VISIBLE_DEVICES="0" python -m train --run_name 2dLig_pdbbind --wandb --batch_size 16 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --pocket_residue_cutoff 12 --clamp_loss 10 --batch_norm --gradient_clip_val 1 --save_inference --check_nan_grads --residue_loss_weight 1 --use_true_pos --use_inv --ignore_lig --lig2d_mpnn --lig_mpnn_layers 4 --lig2d_batch_norm

CUDA_VISIBLE_DEVICES="0" python -m train --run_name randomLigPos_pdbbind --wandb --batch_size 16 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --pocket_residue_cutoff 12 --clamp_loss 10 --layer_norm --gradient_clip_val 1 --save_inference --check_nan_grads --residue_loss_weight 1 --use_true_pos --use_inv --mask_lig_translation --mask_lig_pos

CUDA_VISIBLE_DEVICES="0" python -m train --run_name groundTruthPospdbbind --wandb --batch_size 16 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --pocket_residue_cutoff 12 --clamp_loss 10 --batch_norm --gradient_clip_val 1 --save_inference --check_nan_grads --residue_loss_weight 1 --use_true_pos --use_inv --fake_constant_dur 10000 --fake_decay_dur 40 --fake_ratio_start 0.2 --fake_ratio_end 0.2
```

Binding Site Recovery MOAD:
```
CUDA_VISIBLE_DEVICES="0" python -m train --run_name noLig_moad --wandb --batch_size 16 --train_split_path index/moad_mmseqs_30sim_train --val_split_path index/moad_mmseqs_30sim_val --predict_split_path index/moad_mmseqs_30sim_test --data_dir data/moad/moad_prepared --data_source moad --biounit1_only --pocket_residue_cutoff 12 --clamp_loss 10 --batch_norm --gradient_clip_val 1 --save_inference --check_nan_grads --residue_loss_weight 1 --use_true_pos --use_inv --min_num_contacts 2 --correct_moad_lig_selection --ignore_lig

CUDA_VISIBLE_DEVICES="0" python -m train --run_name 2dLig_moad --wandb --batch_size 16 --train_split_path index/moad_mmseqs_30sim_train --val_split_path index/moad_mmseqs_30sim_val --predict_split_path index/moad_mmseqs_30sim_test --data_dir data/moad/moad_prepared --data_source moad --biounit1_only --pocket_residue_cutoff 12 --clamp_loss 10 --batch_norm --gradient_clip_val 1 --save_inference --check_nan_grads --residue_loss_weight 1 --use_true_pos --use_inv --ignore_lig --lig2d_mpnn --lig_mpnn_layers 4 --lig2d_batch_norm --min_num_contacts 2 --correct_moad_lig_selection

CUDA_VISIBLE_DEVICES="0" python -m train --run_name randomLigPos_moad --wandb --batch_size 16 --train_split_path index/moad_mmseqs_30sim_train --val_split_path index/moad_mmseqs_30sim_val --predict_split_path index/moad_mmseqs_30sim_test --data_dir data/moad/moad_prepared --data_source moad --biounit1_only --pocket_residue_cutoff 12 --clamp_loss 10 --layer_norm --gradient_clip_val 1 --save_inference --check_nan_grads --residue_loss_weight 1 --use_true_pos --use_inv --mask_lig_translation --mask_lig_pos --correct_moad_lig_selection --min_num_contacts 2

CUDA_VISIBLE_DEVICES="0" python -m train --run_name groundTruthPosmoad --wandb --batch_size 16 --train_split_path index/moad_mmseqs_30sim_train --val_split_path index/moad_mmseqs_30sim_val --predict_split_path index/moad_mmseqs_30sim_test --data_dir data/moad/moad_prepared --data_source moad --biounit1_only --pocket_residue_cutoff 12 --clamp_loss 10 --batch_norm --gradient_clip_val 1 --save_inference --check_nan_grads --residue_loss_weight 1 --use_true_pos --use_inv --min_num_contacts 2 --correct_moad_lig_selection --fake_constant_dur 100000 --fake_decay_dur 10 --fake_ratio_start 0.2 --fake_ratio_end 0.2
```

Flow matching investigation / ablation on PDBBind:
```
CUDA_VISIBLE_DEVICES="0" python -m train --run_name StandardTFNLayers --wandb --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 4 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities --self_condition_x --fixed_lig_pos --update_last_when_fixed 

CUDA_VISIBLE_DEVICES="0" python -m train --run_name EigenFold_diffusion --wandb --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 1 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities

CUDA_VISIBLE_DEVICES="0" python -m train --run_name VelocityPrediction --wandb --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 4 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities --self_condition_x --velocity_prediction

CUDA_VISIBLE_DEVICES="0" python -m train --run_name NoSelfConditioning --wandb --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 1 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities

CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_sigma0 --wandb --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 1 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities --self_condition_x

CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_sigma0.5 --wandb --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 1 --cross_radius 50 --protein_radius 30 --lig_radius 50 --pocket_residue_cutoff 12 --ns 32 --nv 8 --diffdock_pocket --tfn_use_aa_identities --self_condition_x
```

HarmonicFlow blind docking on PDBBind:
```
CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_blind_seqSimSplit --wandb --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 250 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 5 --cross_radius 50 --protein_radius 30 --lig_radius 50 --full_prot_diffusion_center --ns 32 --nv 8 --tfn_use_aa_identities --self_condition_x 

CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_blind_timesplit --wandb --batch_size 4 --train_split_path index/timesplit_no_lig_overlap_train --val_split_path index/timesplit_no_lig_overlap_val --predict_split_path index/timesplit_test --clamp_loss 10 --epochs 250 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 5 --cross_radius 50 --protein_radius 30 --lig_radius 50 --full_prot_diffusion_center --ns 32 --nv 8 --tfn_use_aa_identities --self_condition_x 

CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_blind_seqSimSplit_noResidueIdentities --wandb --batch_size 4 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --clamp_loss 10 --epochs 250 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 5 --cross_radius 50 --protein_radius 30 --lig_radius 50 --full_prot_diffusion_center --ns 32 --nv 8 

CUDA_VISIBLE_DEVICES="0" python -m train --run_name HarmonicFlow_blind_timesplit_noResidueIdentities --wandb --batch_size 4 --train_split_path index/timesplit_no_lig_overlap_train --val_split_path index/timesplit_no_lig_overlap_val --predict_split_path index/timesplit_test --clamp_loss 10 --epochs 250 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 150 --num_workers 4 --check_val_every_n_epoch 5 --cross_radius 50 --protein_radius 30 --lig_radius 50 --full_prot_diffusion_center --ns 32 --nv 8 
```
