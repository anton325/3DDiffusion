import numpy as np
import torch
from torchvision.utils import save_image
import cv2

from gecco_torch.gaussian_renderer_unchanged import render
from gecco_torch.scene.dataset_readers import readSpiralCircleCamInfos, readCamerasFromTransforms
from gecco_torch.utils.camera_utils import cameraList_from_camInfos


class Args:
    def __init__(self):
        self.resolution = 1
        self.data_device = "cuda"

class Pipe:
    def __init__(self):
        self.debug = False
        self.compute_cov3D_python = False
        self.convert_SHs_python = False

def get_cameras():
    path = "/globalwork/giese/shapenet_rendered/02958343/b8f6994a33f4f1adbda733a39f84326d"
    white_background = True
    bg_color = [1,1,1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, ".png")
    circle_cam_infos, spiral_cam_infos = readSpiralCircleCamInfos(path,white_background,train_cam_infos[0])

    args = Args()
    pipe = Pipe()
    
    cameras = cameraList_from_camInfos(circle_cam_infos, 1, args)
    return cameras, pipe, background

def render_gaussian(gaussians, **kwargs):
    if kwargs is None:
        kwargs = {}
    kwargs['rasterizer'] = 'depth'
    
    cameras, pipe, background = get_cameras()
    writer = cv2.VideoWriter("debug_images/aufnahme.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps=60,frameSize=(400,400))

    for idx, view in enumerate(cameras):
        rendering = render(view, gaussians, pipe, background, **kwargs)
        rendering = rendering["render"]
        if torch.isnan(rendering).any():
            print("NA in rendered image")
        # torchvision.utils.save_image(rendering,"rend.png")
        save_image(rendering, 'debug_images/img_{0:05d}'.format(idx) + ".png")
        rendering = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        writer.write(cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR))
    writer.release()


def data_to_gaussian(gaussians, data):
    gaussians._xyz = torch.from_numpy(data[:,:3]).cuda()
    gaussians._features_dc = torch.from_numpy(data[:,3:6]).cuda().reshape(-1, 1, 3)
    gaussians._scaling = torch.from_numpy(data[:,6:9]).cuda()
    gaussians._rotation = torch.from_numpy(data[:,9:13]).cuda()
    gaussians._opacity = torch.from_numpy(data[:,13:]).cuda()
    gaussians._features_rest = torch.zeros((data.shape[0], 15, 3)).cuda()
    return gaussians

if __name__ == "__main__":
    from gecco_torch.scene.gaussian_model import GaussianModel
    # import numpy as np
    # data = np.load("cloud.npz")['data']
    ply_path = "/home/giese/Documents/gaussian-splatting/output/green_airplane/point_cloud/iteration_10000/point_cloud.ply"
    ply_path = "/globalwork/giese/gaussians/02958343/2d41d907b7cb558db6f3ca49e992ad8/point_cloud/iteration_10000/point_cloud.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path)
    # gaussians = data_to_gaussian(gaussians, data[24])
    render_gaussian(gaussians)