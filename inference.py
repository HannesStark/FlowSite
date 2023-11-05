import os
import shutil
from datetime import datetime
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datasets.inference_dataset import InferenceDataset
from utils.logging import warn_with_traceback, Logger, lg
import warnings
import sys
"""
warnings.filterwarnings("error", message=".*An output.*", category=UserWarning)
"""
#warnings.filterwarnings("error", message="An output", category=UserWarning)

from lightning_modules.flowsite_module import FlowSiteModule
from models.flowsite_model import FlowSiteModel
os.environ['KMP_DUPLICATE_LIB_OK']='True' # for running on a macbook
import wandb
import torch
from torch_geometric.loader import DataLoader
from datasets.complex_dataset import ComplexDataset
from utils.parsing import parse_train_args

def main_function():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parse_train_args()
    args.run_name_timed = args.run_name + '_' + datetime.fromtimestamp(datetime.now().timestamp()).strftime("%Y-%m-%d_%H-%M-%S")
    torch.set_float32_matmul_precision(precision=args.precision)
    os.environ['MODEL_DIR'] = os.path.join('runs', args.run_name_timed)
    os.makedirs(os.environ['MODEL_DIR'], exist_ok=True)
    sys.stdout = Logger(logpath=os.path.join(os.environ['MODEL_DIR'], f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(os.environ['MODEL_DIR'], f'log.log'), syspart=sys.stderr)

    if args.debug:
        warnings.showwarning = warn_with_traceback

    if args.wandb:
        wandb_logger = WandbLogger(entity='coarse-graining-mit',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args)
    else:
        wandb_logger = None

    predict_data = InferenceDataset(args)
    predict_loader = DataLoader(predict_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FlowSiteModel(args, device)
    model_module = FlowSiteModule(args=args, device=device, model=model)

    trainer = Trainer(logger=wandb_logger,
                        default_root_dir=os.environ['MODEL_DIR'],
                        num_sanity_val_steps=0,
                        log_every_n_steps=args.print_freq,
                        max_epochs=args.epochs,
                        enable_checkpointing=True,
                        limit_test_batches=args.limit_test_batches or 1.0,
                        limit_train_batches=args.limit_train_batches or 1.0,
                        limit_val_batches=args.limit_val_batches or 1.0,
                        check_val_every_n_epoch=args.check_val_every_n_epoch,
                        gradient_clip_val=args.gradient_clip_val,
                        callbacks=[ModelCheckpoint(monitor=('val_accuracy' if not args.all_res_early_stop else 'val_all_res_accuracy') if args.residue_loss_weight > 0 else 'val_rmsd<2', mode='max', filename='best', save_top_k=1, save_last=True, auto_insert_metric_name=True, verbose=True)]
                      )
    numel = sum([p.numel() for p in model_module.model.parameters()])
    lg(f'Model with {numel} parameters')

    lg("Starting inference and saving predictions to: ", args.out_dir)
    trainer.predict(model=model_module, dataloaders=predict_loader, ckpt_path=args.checkpoint)

if __name__ == '__main__':
    main_function()


