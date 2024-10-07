import os
import json
import pathlib
import shutil

import torch
import lightning.pytorch as pl # pytorch-lightning            2.2.1

from gecco_torch.diffusionsplat import EDMPrecond, Diffusion, LogUniformSchedule, EDMLoss
from gecco_torch.reparam import GaussianReparam, get_reparam_parameters
from gecco_torch.models.set_transformer import SetTransformer
from gecco_torch.models.gaussian_ray import RayNetwork
from gecco_torch.models.feature_pyramid import ConvNeXtExtractor, DinoV2Extractor
from gecco_torch.models.activation import GaussianActivation
from gecco_torch.data.gaussian_pc_dataset import Gaussian_pc_DataModule
from gecco_torch.data.gaussian_pc_dataset_zip import Gaussian_pc_DataModule_zip
from gecco_torch.structs import Mode, enum_serializer

from gecco_torch.utils.render_tensors import get_render_fn

from gecco_torch.ema import EMACallback
from gecco_torch.vis_gaussian import PCVisCallback
from gecco_torch.likelihood_reverse_sample import LikelihoodCallback
from pytorch_lightning.loggers import WandbLogger
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import zipfile

import datetime
from argparse import ArgumentParser
import sys

"""
wir wollen in 3 gruppen trainieren, deswegen legen wir für jede gruppe ein .lst file an
Die gruppen sind folgende:
gruppe 1 all:: alle (haben wir schon in create_train_splits_ins_level) besteht aus 35.300 objekten
gruppe 2 good: nur 0 overlap (also die die gar kein einziges bild mit mehrs als 5 overlap haben) -> also das gute dataset besteht aus 31.500 objekten
gruppe 3 mid: alles bis 30 overlap (die danach hauen wir raus) -> also das mittelgute dataset # besteht aus 35.000 objekten
"""

LOCAL_DEBUG = True
group = "good" # "good" "mid" # "all"

# WORK
# mode = [Mode.in_world_space, Mode.rgb, Mode.normal, Mode.rotational_distance, Mode.gecco_projection, Mode.dino_triplane] # , Mode.ctx_splat, Mode.splatting_loss

mode = [Mode.in_world_space, Mode.procrustes, Mode.rgb, Mode.gecco_projection, Mode.rotational_distance, Mode.log_grads, Mode.splatting_loss, Mode.ctx_splat]

# mode = [Mode.in_world_space, Mode.rgb, Mode.activated_scales, Mode.rotational_distance, Mode.gecco_projection, Mode.log_grads, Mode.splatting_loss , Mode.ctx_splat]#, Mode.splatting_loss
# mode = [Mode.in_world_space, Mode.rgb, Mode.activated_scales, Mode.rotational_distance, Mode.gecco_projection, Mode.log_grads, Mode.splatting_loss , Mode.ctx_splat]#, Mode.splatting_loss
# mode = [Mode.in_world_space, Mode.rgb, Mode.procrustes, Mode.rotational_distance, Mode.gecco_projection, Mode.log_grads, Mode.ctx_splat, Mode.splatting_loss] # , Mode.ctx_splat, Mode.splatting_loss
# mode = [Mode.in_world_space,Mode.log_L, Mode.splatting_loss]

# mode = [Mode.in_world_space, Mode.activated_scales, Mode.rgb, Mode.gecco_projection, Mode.rotational_distance, Mode.log_grads, Mode.splatting_loss]
# mode = [Mode.in_world_space, Mode.normal_opac, Mode.rgb, Mode.gecco_projection, Mode.rotational_distance, Mode.log_grads, Mode.splatting_loss, Mode.ctx_splat]

# mode = [Mode.in_world_space, Mode.normal, Mode.rgb, Mode.gecco_projection, Mode.rotational_distance, Mode.log_grads, Mode.splatting_loss, Mode.ctx_splat]

# mode = [Mode.in_world_space, Mode.cholesky, Mode.rgb, Mode.cholesky_distance, Mode.log_grads, Mode.gecco_projection, Mode.splatting_loss, Mode.ctx_splat] # , Mode.ctx_splat, Mode.splatting_loss

# mode = [Mode.in_world_space, Mode.procrustes, Mode.splatting_loss, Mode.rgb, Mode.depth_projection, Mode.log_grads, Mode.rotational_distance, Mode.ctx_splat]

# mode = [Mode.in_world_space, Mode.rgb, Mode.log_L, Mode.cholesky_distance, Mode.gecco_projection, Mode.splatting_loss, Mode.ctx_splat, Mode.log_grads] # , Mode.ctx_splat, Mode.splatting_loss
print(f"Train with: {mode}")
model_size = {
    'convnext_size' : "tiny", # tiny small
    'n_layers' : 6, # num layers set transformer # 6 8 
    'num_inducers' : 64, # 64 128
    'mlp_depth' : 1, # 1 2
}

splatting_loss_starting_step = 15000 if Mode.warmup_xyz in mode else 0 # 5000

splatting_loss = {
    'lambda' : 10000, # weight on splatting loss
    'starting_step' : splatting_loss_starting_step, # 0. epoche ohne, weil da die bilder noch zu hässlich aussehen und das würde das model nur verwirren

    'warmup_steps' : 5000, # number of steps where only the xyz loss is optimized (if warmup is activated)
    }
    
render_fn = get_render_fn(mode)

slurm_job_id = os.getenv('SLURM_JOB_ID')
print(f"CUDA LAUNCH BLOCKING: {os.getenv('CUDA_LAUNCH_BLOCKING')}")

worker_zipfile_instances = {}

def each_worker_one_zip(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # Access dataset object
    zip_path = dataset.location_zip_file  # Access zipfile path from dataset
    # Open zipfile for each worker and store it in the global dictionary
    worker_zipfile_instances[worker_id] = zipfile.PyZipFile(zip_path, 'r')
    print(f"loaded zipfile for worker {worker_id}")

on_cluster = False
dataset_path = pathlib.Path("/globalwork/giese/gaussians")
dataset_images = pathlib.Path("/globalwork/giese/shapenet_rendered")
datamodule = Gaussian_pc_DataModule
worker_init_fn = None
num_workers = 8

# dataset_images = pathlib.Path("/globalwork/giese/shapenet_rendered.zip")
# datamodule = Gaussian_pc_DataModule_zip
# worker_init_fn = each_worker_one_zip
# num_workers = 2

if not dataset_path.exists():
    print("on cluster")
    on_cluster = True
    dataset_path = pathlib.Path("/home/wc101072/gaussians")
    dataset_images = pathlib.Path("/work/wc101072/shapenet_rendered.zip")
    datamodule = Gaussian_pc_DataModule_zip
    num_workers = 8
    worker_init_fn = each_worker_one_zip

if not LOCAL_DEBUG:
    epoch_size = 5000 # 10000
    batch_size = 48
    if Mode.dino_triplane in mode:
        batch_size = 36
    val_size = 100 # gibt absolute anzahl an elementen im val dataset an
    wandb_mode = 'online'
    single_example = False
    num_sanity_val_steps = 0
else:
    epoch_size = 3 # 10000
    batch_size = 3
    val_size = 3
    wandb_mode = 'disabled' # 'online' 'disabled'
    single_example = False
    num_sanity_val_steps = 0

data = datamodule(
    dataset_path,
    dataset_images,
    group = group,
    epoch_size=epoch_size, # 5_000 , # 10k for 500k steps
    batch_size=batch_size, #  original conditional: A100 GPU, 40GB, 2048 points, batch 48, SDE sampler with 128 steps
    num_workers=num_workers,
    val_size = val_size, #1000, # wieviele instances pro validation epoch abgefragt werden sollen
    single_example = single_example,
    worker_init_fn = worker_init_fn,
    worker_zipfile_instances = worker_zipfile_instances,
    mode = mode,
    restrict_to_categories=['02958343'] # 02958343 car # 02691156 plane
)

mean, sigma = get_reparam_parameters(mode)

reparam = GaussianReparam( # wie sind sie auf diese Werte gekommen? 
    mean=mean,
    sigma=sigma,
)

if Mode.dino in mode or Mode.dino_triplane in mode:
    conditioner = DinoV2Extractor(model="small")
    context_dims = [384] # als list damit im gaussian ray die summe genommen werden kann, 384 small, 784 base
else:
    conditioner = ConvNeXtExtractor(model = model_size['convnext_size'])
    context_dims = (96, 192, 384) # 672

dinov2 = None
if Mode.dino_triplane in mode:
    dinov2 = DinoV2Extractor(model="small")

feature_dims = 3 * 128 # 384
network = RayNetwork(
    backbone=SetTransformer(
        feature_dim=feature_dims,
        t_embed_dim=1,
        num_heads=8,
        activation=GaussianActivation,
        **model_size,
    ),
    reparam=reparam,
    context_dims=context_dims,  # NICHT SICHER OB RICHTIG GEWÄHLT, einfach von taskonomy übernommen
    mode = mode,
    render_fn=render_fn,
    dinov2=dinov2,
)

# config_dest = pathlib.Path(pathlib.Path.home(),"Documents","gecco",'gecco-torch',"example_configs","lightning_logs",f'version_{slurm_job_id}')

config_dest = pathlib.Path(pathlib.Path.home(),"..","..",'globalwork',"giese","gecco_shapenet","logs",f'version_{slurm_job_id}')
if on_cluster:
    config_dest = pathlib.Path(pathlib.Path.home(),"..","..","work","wc101072","gecco_logs",f"version_{slurm_job_id}")
config_dest.mkdir(parents=True, exist_ok=True)

print("Putting together the model...")
model = Diffusion(
    backbone=EDMPrecond(
        model=network,
        mode=mode,
    ),
    conditioner=conditioner,
    reparam=reparam,
    loss=EDMLoss(
        schedule=LogUniformSchedule(
            max=165,
        ),
        mode = mode,
        splatting_loss = splatting_loss,
    ),
    save_path_benchmark=config_dest,
    render_fn = render_fn,
    mode = mode,
)

# path_to_checkpoint = list(pathlib.Path("/work/wc101072/gecco_logs/version_48576041/saver_checkpoints/").glob("*"))[0]
# path_to_checkpoint = list(pathlib.Path("/globalwork/giese/gecco_shapenet/logs/version_450802/saver_checkpoints/").glob("*"))[0]
# checkpoint = torch.load(path_to_checkpoint, map_location=model.device)
# model.load_state_dict(checkpoint)

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
            LikelihoodCallback(
                mode=mode,
                batch_size = batch_size,
                n = 48,
                n_steps = 64,
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(config_dest, "checkpoints"),
                filename="{epoch}-{benchmark_mean_splatting_loss:.4f}",
                save_top_k=2,  # -1 Saves all epochs
                verbose=True,
                monitor="benchmark_mean_splatting_loss", # val_loss
                mode="min",
            ), # save model after every epoch
            PCVisCallback(n=8,
                          mode = config['mode'],
                          n_steps=128, 
                          point_size=0.01,
                          visualize_save_folder=config_dest,
                          render_fn=render_fn),
        ],
        max_epochs = 20, # orig
        devices=num_gpus,  # This hardcodes training to use 4 GPUs. Remove or adjust depending on your setup.
        strategy=strategy,  # Use Distributed Data Parallel for multi-GPU training
        precision=config['precision'],
        # precision=32,
        # use_distributed_sampler=False,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        num_sanity_val_steps = num_sanity_val_steps,
        accelerator="gpu",
        logger=WandbLogger(project='gecco_gaussian',name=config['run_name'],mode=wandb_mode),
    )

if __name__ == "__main__":
    parser = ArgumentParser(description="Training Parameters")
    parser.add_argument('--name', type=str, default=str(datetime.datetime.now()).replace(" ",""), help="Name of the run")
    args = parser.parse_args(sys.argv[1:])
    run_name = args.name + "_" +str(slurm_job_id)
    CONFIG = {
        'dataset_path' : str(dataset_path),
        'epoch_size' : epoch_size,
        'batch_size' : batch_size,
        'context_dims' : context_dims,
        'feature_dims' : feature_dims,
        'run_name' : run_name,
        'precision' : '16-mixed',# '32' "16-mixed"
        'mode' : mode,
        'reparam_mean':mean.cpu().numpy().tolist(),
        'reparam_sigma' : sigma.cpu().numpy().tolist(),
        'group' : group
    }
    
    config_dest.mkdir(parents=False, exist_ok=True)
    with open(config_dest / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=4, default=enum_serializer)  
    if on_cluster:
        shutil.copy("/home/wc101072/code/gecco/gecco-torch/example_configs/gaussians_conditional.py",config_dest)
        shutil.copy("/home/wc101072/code/gecco/gecco-torch/src/gecco_torch/diffusionsplat.py",config_dest)
        shutil.copy("/home/wc101072/code/gecco/gecco-torch/src/gecco_torch/models/gaussian_ray.py",config_dest)
        shutil.copy("/home/wc101072/code/gecco/gecco-torch/src/gecco_torch/utils/render_tensors.py",config_dest)
        shutil.copy("/home/wc101072/code/gecco/gecco-torch/src/gecco_torch/data/gaussian_pc_dataset_template.py",config_dest)
        shutil.copy("/home/wc101072/code/gecco/wrappers/wrapper_cluster_gaussian.sh",config_dest)
        shutil.copytree("/home/wc101072/code/gecco/gecco-torch/src/gecco_torch/train_forward", config_dest / "train_forward", dirs_exist_ok=True)
        shutil.copytree("/home/wc101072/code/gecco/gecco-torch/src/gecco_torch/projection", config_dest / "projection", dirs_exist_ok=True)
    else:
        shutil.copy("/home/giese/Documents/gecco/gecco-torch/example_configs/gaussians_conditional.py",config_dest)
        shutil.copy("/home/giese/Documents/gecco/gecco-torch/src/gecco_torch/diffusionsplat.py",config_dest)
        shutil.copy("/home/giese/Documents/gecco/gecco-torch/src/gecco_torch/models/gaussian_ray.py",config_dest)
        shutil.copy("/home/giese/Documents/gecco/gecco-torch/src/gecco_torch/utils/render_tensors.py",config_dest)
        shutil.copy("/home/giese/Documents/gecco/gecco-torch/src/gecco_torch/data/gaussian_pc_dataset_template.py",config_dest)
        shutil.copy("/home/giese/Documents/gecco/wrappers/wrapper_cluster_gaussian.sh",config_dest)
        shutil.copytree("/home/giese/Documents/gecco/gecco-torch/src/gecco_torch/train_forward", config_dest / "train_forward", dirs_exist_ok=True)
        shutil.copytree("/home/giese/Documents/gecco/gecco-torch/src/gecco_torch/projection", config_dest / "projection", dirs_exist_ok=True)

    # model = torch.compile(model)
    model = model.cuda()
    t = trainer(config_dest,**CONFIG)
    print("Start fit")
    t.fit(model, data)