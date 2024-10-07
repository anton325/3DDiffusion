import os
import pathlib

import torch
import lightning.pytorch as pl

from gecco_torch.models.load_model import load_model
from gecco_torch.custom_checkpoint import CustomModelCheckpoint

from gecco_torch.ema import EMACallback
from gecco_torch.data.gaussian_pc_dataset import Gaussian_pc_DataModule
from gecco_torch.vis_gaussian import PCVisCallback
from pytorch_lightning.loggers import WandbLogger
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import datetime
from argparse import ArgumentParser
import sys


slurm_job_id = os.getenv('SLURM_JOB_ID')
print(f"CUDA LAUNCH BLOCKING: {os.getenv('CUDA_LAUNCH_BLOCKING')}")

version = "448115"
model, render_fn, data, mode, reparam, epoch = load_model(version)

# mode = [Mode.in_world_space,Mode.cov_matrix,Mode.splatting_loss,Mode.rgb]
starting_step = 0 # für splatting loss
splatting_loss = {
    'lambda' : 50000, # weight on splatting loss
    'starting_step' : starting_step, # 0. epoche ohne, weil da die bilder noch zu hässlich aussehen und das würde das model nur verwirren

    'warmup_steps' : 0, # number of steps where only the xyz loss is optimized (if warmup is activated)
    }

model.loss.splatting_loss = splatting_loss

wandb_mode = 'online'

on_cluster = len(version) > 6
config_dest = pathlib.Path(pathlib.Path.home(),"..","..",'globalwork',"giese","gecco_shapenet","logs",f'version_{version}')
if on_cluster:
    config_dest = pathlib.Path(pathlib.Path.home(),"..","..","work","wc101072","gecco_logs",f"version_{version}")

# LOCAL_DEBUG = True
# if not LOCAL_DEBUG:
#     epoch_size = 1000 # 10000
#     batch_size = 48
#     val_size = 100 # gibt absolute anzahl an elementen im val dataset an
#     wandb_mode = 'online'
#     single_example = False
# else:
#     epoch_size = 3 # 10000
#     batch_size = 3
#     val_size = 3
#     wandb_mode = 'disabled' # 'online' 'disabled'
#     single_example = False

# dataset_path = pathlib.Path("/globalwork/giese/gaussians")
# dataset_images = pathlib.Path("/globalwork/giese/shapenet_rendered")
# # dataset_images = pathlib.Path("/globalwork/giese/shapenet_rendered.zip")
# datamodule = Gaussian_pc_DataModule
# num_workers = 8
# data = datamodule(
#     dataset_path,
#     dataset_images,
#     group = "good",
#     epoch_size=epoch_size, # 5_000 , # 10k for 500k steps
#     batch_size=batch_size, #  original conditional: A100 GPU, 40GB, 2048 points, batch 48, SDE sampler with 128 steps
#     num_workers=num_workers,
#     val_size = val_size, #1000, # wieviele instances pro validation epoch abgefragt werden sollen
#     single_example = single_example,
#     mode = mode,
#     restrict_to_categories=['02958343'] # 02958343 car # 02691156 plane
# )

def trainer(config_dest, **config):
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        strategy = 'ddp'
    else:
        strategy = 'auto'
    print(f"Number of GPUs available: {num_gpus}")
    return pl.Trainer(
        default_root_dir=os.path.split(__file__)[0],
        callbacks=[
            EMACallback(decay=0.99),
            CustomModelCheckpoint(
                start_epoch=epoch,
                dirpath=os.path.join(config_dest, "checkpoints"),
                filename="{epoch}-{val_loss:.2f}",
                save_top_k=2,  # -1 Saves all epochs
                verbose=True,
                monitor="val_loss",
                mode="min",
            ), # save model after every epoch
            # pl.callbacks.ModelCheckpoint( # saves model with minimum validation loss
            #     monitor="val_loss",
            #     filename="{epoch}-{val_loss:.3f}",
            #     save_top_k=2,
            #     mode="min",
            # ),
            PCVisCallback(n=8,
                          mode = config['mode'],
                          n_steps=128, 
                          point_size=0.01,
                          mesh_save_folder_id=config_dest,
                          render_fn=render_fn),
        ],
        max_epochs=50-epoch, # orig
        devices=num_gpus,  # This hardcodes training to use 4 GPUs. Remove or adjust depending on your setup.
        strategy=strategy,  # Use Distributed Data Parallel for multi-GPU training
        precision=config['precision'],
        # precision=32,
        # use_distributed_sampler=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
        num_sanity_val_steps = 0,
        accelerator="gpu",
        logger=WandbLogger(project='gecco_gaussian',name=config['run_name'],mode=wandb_mode),
    )

if __name__ == "__main__":
    parser = ArgumentParser(description="Training Parameters")
    parser.add_argument('--name', type=str, default=str(datetime.datetime.now()).replace(" ",""), help="Name of the run")
    args = parser.parse_args(sys.argv[1:])
    run_name = args.name + version+"_cont_" + str(slurm_job_id)
    CONFIG = {
        'run_name' : run_name,
        'precision' : '16-mixed',# 32 "16-mixed"
        'mode' : mode,
    }
    model = torch.compile(model)
    t = trainer(config_dest,**CONFIG)
    t.fit(model, data)
