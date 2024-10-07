import torch
from gecco_torch.diffusion import EDMPrecond
from gecco_torch.reparam import GaussianReparam
from gecco_torch.diffusion import Diffusion, LogUniformSchedule, EDMLoss
from gecco_torch.models.set_transformer import SetTransformer
from gecco_torch.models.ray import RayNetwork
from gecco_torch.models.feature_pyramid import ConvNeXtExtractor
from gecco_torch.models.activation import GaussianActivation
from gecco_torch.data.shapenet_cond import ShapenetCondDataModule
import lightning.pytorch as pl
from pathlib import Path
import numpy as np
from gecco_torch.structs import Context3d
from gecco_torch.metrics import chamfer_distance_naive_occupany_networks, chamfer_distance_kdtree_occupancy_network # chamfer_distance_naive_occupany_networks_np 
import pickle

def check_on_dataloader(dataloader,output_dir,split):
    vals = 75
    cds = []
    cds_upsampled = []
    cat_cds = {}
    cat_cds_upsampled = {}
    for i,batch in  enumerate(dataloader):
        if i>vals:
            break
        s = (batch.data.shape[0],2048,3)
        # with torch.autocast('cuda', dtype=torch.float16):
        # batch.ctx = batch.ctx.apply_to_tensors(map_device)
        ctx = Context3d(image=map_device(batch.ctx.image),K=map_device(batch.ctx.K),category=batch.ctx.category)
        print("sampling...")
        outputs = model.sample_stochastic(s,context=ctx,pbar=True)  # Perform inference

        for j,(gt,sample) in enumerate(zip(batch.data,outputs)):
            where_to = Path(output_dir,split,f"b_{str(i).zfill(3)}_s_{str(j).zfill(3)}")
            where_to.mkdir(parents=True,exist_ok=True)
            
            gt = map_device(gt)

            subset = np.random.permutation(gt.shape[0])[: 2048]
            downsampled = gt[subset]

            cd = chamfer_distance_naive_occupany_networks(sample.unsqueeze(0),downsampled.unsqueeze(0))
            print(f"batch {i}, instance {j}, chamfer distance 2048: {cd}")

            onectx = Context3d(image=map_device(batch.ctx.image[j].unsqueeze(0)),K=map_device(batch.ctx.K[j].unsqueeze(0)),category=batch.ctx.category[j])
            print("upsampling...")
            upsampled = model.upsample(
                n_new=100_000,
                data=sample.unsqueeze(0),
                context=onectx,
                with_pbar=True,
                num_steps=32,
                )
            cd_upsampled = chamfer_distance_kdtree_occupancy_network(gt.unsqueeze(0),upsampled)
            print(f"batch {i}, instance {j}, chamfer distance upsampled: {cd_upsampled}")
            result = {
                'gt' : gt.cpu().numpy(),
                'downsampled' : downsampled.cpu().numpy(),
                'sample' : sample.cpu().numpy(),
                'upsampled' : upsampled.cpu().numpy(),
                'cd' : cd,
                'image' : onectx.image.cpu().numpy(),
                'K': onectx.K.cpu().numpy(),
            }
            cds.append(cd.cpu().numpy()[0])
            if cat_cds.get(batch.ctx.category[j]) is None:
                cat_cds[batch.ctx.category[j]] = [cd.cpu().numpy()[0]]
            else:
                cat_cds[batch.ctx.category[j]].append(cd.cpu().numpy()[0])

            cds_upsampled.append(cd_upsampled.cpu().numpy()[0])
            if cat_cds_upsampled.get(batch.ctx.category[j]) is None:
                cat_cds_upsampled[batch.ctx.category[j]] = [cd_upsampled.cpu().numpy()[0]]
            else:
                cat_cds_upsampled[batch.ctx.category[j]].append(cd_upsampled.cpu().numpy()[0])

            print(cat_cds)
            with open(Path(where_to,"eval.pkl"),"wb") as f:
                pickle.dump(result,f)
    return cds, cat_cds, cds_upsampled, cat_cds_upsampled
        

# Define the model architecture as before
def create_model():
    reparam = GaussianReparam( # mit i=200 berechnet
        mean=torch.tensor([0.0, -0.007, 1.393]),
        sigma=torch.tensor([0.18, 0.162, 0.235]),
    )
    context_dims = (96, 192, 384)
    feature_dim = 3 * 128
    network = RayNetwork(
        backbone=SetTransformer(
            n_layers=6,
            num_inducers=64,
            feature_dim=feature_dim,
            t_embed_dim=1,
            num_heads=8,
            activation=GaussianActivation,
        ),
        reparam=reparam,
        context_dims=context_dims,  # NICHT SICHER OB RICHTIG GEWÄHLT, einfach von taskonomy übernommen
    )
    

    model = Diffusion(
        backbone=EDMPrecond(
            model=network,
        ),
        conditioner=ConvNeXtExtractor(),
        reparam=reparam,
        loss=EDMLoss(
            schedule=LogUniformSchedule(
                max=165.0,
            ),
        ),
        save_path_benchmark=None,
    )
    
    return model

dataset_path = Path(Path.home(),"..","..","globalwork","giese","gecco_shapenet","ShapeNet")

batch_size = 1
val_size = 1000
single_example = False

data = ShapenetCondDataModule(
    dataset_path,
    epoch_size=10000, # 5_000 , # 10k for 500k steps
    batch_size=batch_size, #  original conditional: A100 GPU, 40GB, 2048 points, batch 48, SDE sampler with 128 steps
    num_workers=8,
    val_size = val_size, #1000, # wieviele instances pro validation epoch abgefragt werden sollen
    single_example = single_example,
    downsample_points=False,
)
data.setup()

train_dataloader = data.train_dataloader()
val_dataloader = data.val_dataloader()

# Assuming create_model() correctly recreates the architecture used during training
model = create_model()

# Load the checkpoint state dict manually
which_version = 444375
root_checkpoint = f'/home/giese/Documents/gecco/logs/version_{which_version}'
which_epoch = 0
checkpoint_path = Path(root_checkpoint,"checkpoints")
checkpoint_path = [x for x in checkpoint_path.iterdir() if int(x.name.split("-")[0].split("=")[1]) == which_epoch][0]
checkpoint_path = "/globalwork/giese/gecco_shapenet/release_checkpoint_shapenet_cond.ckpt"
checkpoint_state_dict = torch.load(checkpoint_path, map_location='cuda')
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
model_state_dict = checkpoint_state_dict['ema_state_dict']
model.load_state_dict(model_state_dict)

# Continue with data preparation and inference as before

if torch.cuda.is_available():
    map_device = lambda x: x.to(device='cuda')
else:
    map_device = lambda x: x

model: Diffusion = map_device(model).eval()

# gts = np.load(Path(root_checkpoint,"benchmark-checkpoints","chamfer_distance_squared",f"epoch_{which_epoch}","gt_vertices.npz"))
# samples = np.load(Path(root_checkpoint,"benchmark-checkpoints","chamfer_distance_squared",f"epoch_{which_epoch}","vertices.npz"))

output_dir = Path(root_checkpoint,"eval")
cds_val, cds_cat_val, cds_val_upsampled, cds_cat_val_upsampled = check_on_dataloader(val_dataloader,output_dir,"val")
print(f"cds val: {cds_val}, mean: {np.mean(cds_val)}")
print(f"cds val upsampled: {cds_val_upsampled}, mean: {np.mean(cds_val_upsampled)}")
# cds_train,cds_cat_train = check_on_dataloader(train_dataloader,output_dir,"train")
# print(f"cds train: {cds_train}, mean: {np.mean(cds_train)}")
data.setup(stage="test")
test_dataloader = data.test_dataloader()
cds_test, cds_cat_test,cds_test_upsampled, cds_cat_test_upsampled = check_on_dataloader(test_dataloader,output_dir,"test")
print(f"cds test: {cds_test}, mean: {np.mean(cds_test)}")
print(f"cds test: {cds_test_upsampled}, mean: {np.mean(cds_test_upsampled)}")

print("cds cat val:")
for k in cds_cat_val.keys():
    print(f"{k}: {np.mean(cds_cat_val[k])}")
print("cds cat val upsampled: ")
for k in cds_cat_val_upsampled.keys():
    print(f"{k}: {np.mean(cds_cat_val_upsampled[k])}")

print("cds cat test:")
for k in cds_cat_test.keys():
    print(f"{k}: {np.mean(cds_cat_test[k])}")

print("cds cat test upsampled:")
for k in cds_cat_test_upsampled.keys():
    print(f"{k}: {np.mean(cds_cat_test_upsampled[k])}")