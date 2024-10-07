import wandb
import torch
from pathlib import Path
import numpy as np
import random
from gecco_torch.structs import Mode
from gecco_torch.scene.gaussian_model import GaussianModel
from gecco_torch.utils.loss_utils import l1_loss

SIGMA_THRESHOLD = 0.5 # 0.1 # 0.3 # 0.05

def splatting_loss(self, train_step, phase, render_fun, data, context, log_fun, sigma):
    render_worked = True
    # gaussians2 = GaussianModel(3)    
    # gaussians2.load_ply(f"/globalwork/giese/gaussians/02958343/8c6c271a149d8b68949b12cf3977a48b/point_cloud/iteration_10000/point_cloud.ply")
    # data[0,:,:3] = gaussians2._xyz[:4000]
    # data[0,:,3:6] = gaussians2._features_dc.reshape(-1,3)[:4000]
    # data[0,:,6:9] = gaussians2._scaling[:4000]
    # data[0,:,9:13] = gaussians2._rotation[:4000]
    # data[0,:,13] = gaussians2._opacity[:4000].squeeze(1)
    # data[1] = data[0]
    # data[2] = data[0]
    if Mode.ctx_splat in self.mode:
        splatting_cam_choice = None
    else:
        splatting_cam_choice = np.random.randint(0,len(context.splatting_cameras))
    try:
        if Mode.splatting_loss in self.mode and self.splatting_loss['starting_step'] <= train_step and phase == "train":
            # render_fun macht bei in camera space und normal IN PLACE Änderungen an data!!!! 
            images_dict = render_fun(data.clone(),context, self.mode, splatting_cam_choice, step=train_step)
            images = images_dict['render']
            print("Train render successful (grad)")
        else:
            with torch.no_grad():
                # render_fun macht bei in camera space und normal IN PLACE Änderungen an data!!!! 
                images_dict = render_fun(data.clone(),context, self.mode,step=train_step)
                images = images_dict['render']
                print("Train render successful (no grad)")
        splatting_losses = []
        for i in range(data.shape[0]):
            rendered_image = images[i]
            if torch.isnan(rendered_image).any():
                print(f"rendered image has nan at {train_step}")

            if torch.isnan(context.image[i]).any():
                print(f"gt image has nan at {train_step}")
            try:
                if random.random() < 5/48000:
                    # save prozess von torchvison.utils.save_image
                    wandb_img = rendered_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy() 
                    wandb_img = wandb.Image(rendered_image, caption=f"step_{train_step}_s_{sigma[i].squeeze().detach().cpu().numpy():0.3f}_l_{splatting_losses[i].detach().cpu().numpy():0.3f}")
                    wandb.log({f"train_pics": [wandb_img]})
            except Exception as e:
                print(f"wandb failed {e}")

            if sigma[i].squeeze() > SIGMA_THRESHOLD:
                continue
            
            if Mode.ctx_splat in self.mode:
                splatting_losses.append(l1_loss(rendered_image, context.image[i]))
            else:
                splatting_losses.append(l1_loss(rendered_image, context.splatting_cameras[splatting_cam_choice][1][i]))
                # save_image(rendered_image,f"img_{i}.png")
                # save_image(context.splatting_cameras[splatting_cam_choice][1][i],f"img_gt_{i}.png")
                # context.splatting_cameras[anzahl splatting cameras][0=camera, 1=img][batch]
                # context.splating_cameras[anzahl splatting cameras][0].world_view_transform[batch]

            # imageio.imsave(f"/home/giese/Documents/gecco/rend/rend/{phase}_render_{train_step}_{str(i).zfill(3)}_sigma_{sigma[i].squeeze().detach().cpu().numpy()}_loss_{splatting_losses[i].detach().cpu().numpy()}.png",(255*rendered_image).type(torch.uint8).permute(1,2,0).cpu().numpy())
            # imageio.imsave(f"/home/giese/Documents/gecco/rend/gt/{phase}_gt_{train_step}_{str(i).zfill(3)}.png",(255*context.image[i]).type(torch.uint8).permute(1,2,0).cpu().numpy())
            
            # pro epoche sind wir hier 1000*48 mal -> wir wollen 1 pro 5000 -> 10 pro epoche -> 10/48000
        # print(f"Splatting losses: {splatting_losses}")
        mean_splatting_losses = sum(splatting_losses)/len(splatting_losses)
        
        if mean_splatting_losses > 0.03:
            mean_splatting_losses = torch.clip(mean_splatting_losses, 0, 0.03)
        log_fun("mean_splatting_loss", mean_splatting_losses, on_step=True)
    except Exception as e:
        render_worked = False
        # save_path = Path(f"/home/giese/Documents/gecco/rend/broken/")
        print(f"render splatting loss failed step {train_step} with error {e}")
        # np.savez(Path(save_path,f"{phase}_data_{train_step}.npz"),data=np.array([1]))

    if render_worked:
        return mean_splatting_losses
    else:
        return None