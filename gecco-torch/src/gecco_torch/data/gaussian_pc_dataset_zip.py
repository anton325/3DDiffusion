from pathlib import Path
from functools import partial
import torch
import numpy as np
import imageio as iio
import lightning.pytorch as pl
from tqdm.auto import tqdm
import multiprocess as mp
import os
import json
from plyfile import PlyData
import math
import zipfile
import gecco_torch.utils.graphics_utils as graphic_utils
from gecco_torch.data.zipreader import read_file

from gecco_torch.structs import GaussianExample, GaussianContext3d, Camera, InsInfo
from gecco_torch.data.gaussian_pc_dataset_template import GaussianPCModel_Template
from gecco_torch.structs import Mode


NUM_SPLATTING_CAMERAS = 2
IM_SIZE = 400  # 400 x 400 pixels


class Gaussian_pc_DataModule_zip(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        root_images: str,
        group:str,
        mode: Mode = None,
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
        val_size: int | None = 10_000,
        single_example: bool = False,
        worker_init_fn = None,
        worker_zipfile_instances = None,
        restrict_to_categories = None,
    ):
        print("Building gaussian pointcloud data module...")
        super().__init__()
        self.root = root
        self.root_images = root_images
        self.location_zip_file = root_images
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size
        self.val_size = val_size
        self.single_example = single_example
        self.zipfile = zipfile.PyZipFile(self.root_images)
        self.root_images_name = str(self.root_images.name).split(".")[0]
        self.group = group
        self.worker_init_fn = worker_init_fn
        self.worker_zipfile_instances = worker_zipfile_instances
        self.mode = mode
        self.restrict_to_categories = restrict_to_categories
        
    def setup(self, stage: str = "fit"):
        if stage == "fit":
            print("Setup fit...")
            kw = {"single_view" : False,
                  "single_example" : self.single_example,
                  "group" : self.group,
                  "location_zip_file":self.location_zip_file,
                  "worker_zipfile_instances" : self.worker_zipfile_instances,
                  "mode" : self.mode,
                  "restrict_to_categories" : self.restrict_to_categories,
                  }
            self.train = GaussianPC(self.root, self.root_images_name, self.zipfile, "train",**kw)
            print(f"len self train {len(self.train)}")

            self.val = GaussianPC(self.root, self.root_images_name,self.zipfile, "val",**kw)
            print(f"len self val {len(self.val)}")

        elif stage == "test":
            kw = {"single_view" : True,
                  "group" : self.group,
                  "mode" : self.mode,
                  "restrict_to_categories" : self.restrict_to_categories,
                  }
            self.test = GaussianPC(self.root, self.root_images_name,self.zipfile, "test",kw)

    def train_dataloader(self):
        print("get train dataloader")
        if self.epoch_size is None:
            return torch.utils.data.DataLoader(
                self.train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
                worker_init_fn = self.worker_init_fn,
            )
        train_dataloader = torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=torch.utils.data.RandomSampler(
                self.train, replacement=True, num_samples=self.epoch_size * self.batch_size # dann kriegt man 10_000 * 48 samples, also 10000 steps (wenn ein step ein batch ist)
            ),
            pin_memory=True,
            shuffle=False,
            worker_init_fn = self.worker_init_fn
        )
        # for i,x in enumerate(train_dataloader):
        #     print(i)
        print(f"In get train dataloader epoch size {self.epoch_size}")
        print(f"In get train dataloader batch size {self.batch_size}")
        print(f"In get train dataloader len train_dataloader {len(train_dataloader)}")
        return train_dataloader

    def val_dataloader(self):
        if self.val_size is None:
            sampler = None
        else:
            # funktionsweise random sampler: jedes epoch wird einmal gesampled vom validation dataset. Da kommen dann val_size viele sample bei raus.
            # dann gibt es so viele validation steps, wie es die kombination aus der anzahl samples (val_size) und der batch size ergibt
            # also val_size / batch_size viele steps
            sampler = torch.utils.data.RandomSampler(
                self.val, replacement=True, num_samples=self.val_size
            )

        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
            shuffle=False,
            worker_init_fn = self.worker_init_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn = self.worker_init_fn,
        )
    

class GaussianPC(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        root_images: str,
        zipfile: zipfile.PyZipFile,
        split: str, # split bedeutet hier train, val, test
        **kw,
    ):
        self.location_zip_file = kw['location_zip_file']
        print(f"Building GaussianPC for {root}...")
        # subroots = [Path(potential_category) for potential_category in  zipfile.namelist() if potential_category.count("/") == 2]
        subroots = []
        for maybe_dir in os.listdir(root):
            maybe_dir_path = os.path.join(root, maybe_dir)
            if not os.path.isdir(maybe_dir_path):
                continue
            subroots.append(maybe_dir)
        print(f"split {split} len subroots {len(subroots)}")
        print(f"Building GaussianPC for subroots:\n{subroots} ")
        models = []
        restrict_to_categories = kw.get("restrict_to_categories",None)
        for subroot in subroots:
            if kw.get("single_example", False):
                if "0269" not in subroot:
                    continue
            if restrict_to_categories is not None and not kw.get("single_example", False):
                if subroot not in restrict_to_categories:
                    continue
            print(f"For subroot {subroot}")
            models.append(GaussianPCClass(os.path.join(root,subroot), os.path.join(root_images,subroot),zipfile, split, **kw))
        super().__init__(models)

class GaussianPCClass(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        root_images: str,
        zipfile: zipfile.PyZipFile,
        split: str,
        **kw,
    ):
        print(f"Building GaussianPCClass for {root}...")
        # with zipfile.open(root / f"{split}.lst") as lst_file:
        #     contents = lst_file.read().decode('utf-8')
            
        #     # Split the contents by newline to get a list of items
        #     items = contents.splitlines()
        #     split_ids = [line.strip() for line in items]
        train_on_which_group = ""
        if kw['group'] == "all":
            train_on_which_group = f"shapenet_{split}.lst"
        elif kw['group'] == "good":
            train_on_which_group = f"shapenet_{split}_good.lst"
        elif kw['group'] == "mid":
            train_on_which_group = f"shapenet_{split}_mid.lst"
        else:
            raise Exception(f"train on group {kw['group']} not implemented")
        
        with open(os.path.join(root, train_on_which_group)) as split_file:
            split_ids = [line.strip() for line in split_file]
        paths = [((root+"/"+ id),os.path.join(root_images, id)) for id in split_ids]
        if kw.get("single_example", False):
            paths = [('/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133', '/globalwork/giese/shapenet_rendered/02691156/1a04e3eab45ca15dd86060f189eb133')]
            print(paths)
        kw['zipfile'] = zipfile
        make_model = partial(GaussianPCModel, **kw)

        if kw.get("posed", False) or kw.get("skip_fixed", False):
            # takes a while to load npzs to it's good to parallelize
            with mp.Pool() as pool:
                subsets = list(pool.imap(make_model, paths))
        else:
            # faster to not pay multiprocess overhead
            subsets = list(map(make_model, paths))

        super().__init__(subsets)
        self.root = root
        self.root_images = root_images
        self.split = split

class GaussianPCModel(GaussianPCModel_Template):
    def __init__(
        self,
        root_and_images: str,
        # root_images: str,
        single_view: bool = False,
        **kw
    ):
        super(GaussianPCModel, self).__init__(root_and_images=root_and_images,
                                              single_view=single_view,
                                              **kw)
        self.zipfile = kw['zipfile']
        self.worker_zipfile_instances = kw['worker_zipfile_instances']

        # den /work/wc10101072 teil abschneiden
        if len(self.root_images.split("/")) > 3: # irgendwie nur richtig auf vision geräten, auf dem cluster ist das schon richtig
            self.root_images = "/".join(self.root_images.split("/")[3:])

    def get_camera_params(self, index: int, which_zipfile):
        """
        ******CAMERA MATRIX REFERS TO CAMERA INTRINSIC MATRIX******
        """

        if self.wmats is None:

            jsons = ['transforms_train.json','transforms_val.json','transforms_test.json']
            wmats = []
            cam_to_world_mats = []
            cam_intrinsics = []
            cameras = []
            image_paths = []
            for json_split in jsons:
                path = os.path.join(self.root_images, json_split)
                contents = read_file(which_zipfile,path)

                fovx = contents["camera_angle_x"]

                frames = contents["frames"]
                for idx, frame in enumerate(frames):
                    f_x = f_y = 437.5
                    c_x = IM_SIZE/2
                    c_y = IM_SIZE/2
                    camera_intrinsic_matrix = np.array([
                                                    [f_x, 0, c_x],
                                                    [0, f_y, c_y],
                                                    [0, 0, 1]
                                                        ])

                    image_path_modified = frame["file_path"] + ".png"
                    image_path_modified = image_path_modified[2:]
                    image_path = os.path.join(self.root_images, image_path_modified)
                    image_paths.append(image_path)

                    # NeRF 'transform_matrix' is a camera-to-world transform
                    c2w = np.array(frame["transform_matrix"])
                    # print(c2w.shape)
                    # gaussian splatting preprocessing
                    # print(c2w.shape)
                    # c2w_original = c2w.copy()
                    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                    c2w[:3, 1:3] *= -1
                    # print(c2w.shape)

                    # get the world-to-camera transform and set R, T
                    w2c = np.linalg.inv(c2w)
                    # print(w2c.shape)
                    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                    T = w2c[:3, 3]
                    fovy = self.focal2fov(self.fov2focal(fovx, IM_SIZE), IM_SIZE)
                    FovY = fovy 
                    FovX = fovx
                    tanfovx = math.tan(FovX * 0.5)
                    tanfovy = math.tan(FovY * 0.5)

                    zfar = 100.0
                    znear = 0.01

                    trans = np.array([0.0, 0.0, 0.0])
                    scale = 1.0

                    self.world_view_transform = torch.tensor(graphic_utils.getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                    self.projection_matrix = graphic_utils.getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1)
                    cam = Camera(world_view_transform=self.world_view_transform, 
                                    projection_matrix=self.projection_matrix, 
                                    tanfovx=tanfovx, 
                                    tanfovy=tanfovy,
                                    imsize = IM_SIZE)
                    

                    # gecco operations
                    # c2w = c2w[:3,:3]
                    camera_intrinsic_matrix /= np.array([IM_SIZE + 1, IM_SIZE + 1, 1]).reshape(3, 1)

                    wmats.append(w2c[:3,:]) # nur 3x4

                    cam_intrinsics.append(camera_intrinsic_matrix)

                    cameras.append(cam)
                    cam_to_world_mats.append(c2w)

            self.wmats = np.array(wmats).astype(np.float32)
            self.cam_intrinsics = np.array(cam_intrinsics).astype(np.float32)
            self.image_paths = image_paths
            self.cameras = cameras
            self.c2w_mats = np.array(cam_to_world_mats).astype(np.float32)
            assert len(self.wmats) == len(self.cam_intrinsics) == len(self.image_paths) == len(self.cameras), "Lengths of wmats, cam_intrinsics, cameras, and image_paths do not match"
            assert len(self.wmats) == 50, "Length of wmats is not 50"

        return self.wmats[index], self.c2w_mats[index], self.cam_intrinsics[index], self.image_paths[index], self.cameras[index]

    def prepare_image(self, image_path, which_zipfile):
        image = read_file(which_zipfile,image_path).astype(np.float32) #iio.imread(image_path).astype(np.float32)
        image = np.asarray(image)

        # Normalize the RGB channels to 0-1 range
        image[:, :, :3] /= 255

        # Check if the image has an alpha channel
        if image.shape[2] == 4:
            # Separate the RGB and alpha channels
            rgb = image[:, :, :3]
            alpha = image[:, :, 3] / 255  # Normalize alpha to 0-1 if it's not already

            # Create a white background with the same shape as the original image
            background = np.ones_like(rgb, dtype=np.float32)

            # Blend the image with the white background based on the alpha channel
            image = rgb * alpha[..., None] + background * (1 - alpha[..., None])

        image = image.transpose(2, 0, 1)
        return image
    
    def get_camera_image_tuple(self, index, which_zipfile):
        _, _, _, image_path, camera = self.get_camera_params(index, which_zipfile)
        image = self.prepare_image(image_path, which_zipfile)
        return (camera, image)
    
    def __getitem__(self, index) -> GaussianExample:
        image_indices = [index]
        worker_id = torch.utils.data.get_worker_info().id
        # print(f"worker id {worker_id}")
        which_zipfile = self.worker_zipfile_instances[worker_id]

        wmat, c2w, camera_intrinsics, image_path, camera = self.get_camera_params(index,which_zipfile)
        # load 3D pointcloud
        gaussian_pc, mask_points = self.points_world(wmat,self.mode)

        image = self.prepare_image(image_path, which_zipfile)

        splatting_cameras = []
        while len(splatting_cameras) < NUM_SPLATTING_CAMERAS:
            idx = np.random.randint(0,50)
            if idx in image_indices:
                continue
            splatting_cameras.append(self.get_camera_image_tuple(idx, which_zipfile))
            image_indices.append(idx)
        
    # def __getitem__(self, index) -> GaussianExample:
        # worker_id = torch.utils.data.get_worker_info().id
        # # print(f"worker id {worker_id}")
        # which_zipfile = self.worker_zipfile_instances[worker_id]
        # # which_zipfile = self.zipfile    

        # wmat, c2w, camera_intrinsics, image_path, camera = self.get_camera_params(index,which_zipfile)
        # # print(wmat.shape)
        # # print(cmat.shape)
        # # load 3D pointcloud
        # gaussian_pc, mask_points = self.points_world(wmat,self.mode)
        # # print(gaussian_pc.shape)

        # image = read_file(which_zipfile,image_path).astype(np.float32) #iio.imread(image_path).astype(np.float32)
        # # image = image[:,:,:3]/255 # -> images are supposed to be in range 0-1 and no alpha
        # image = np.asarray(image)
        # # if image.ndim == 2:  # grayscale to rgb
        # #     image = image[..., None].repeat(3, 2)

        # # Normalize the RGB channels to 0-1 range
        # image[:, :, :3] /= 255

        # # Check if the image has an alpha channel
        # if image.shape[2] == 4:
        #     # Separate the RGB and alpha channels
        #     rgb = image[:, :, :3]
        #     alpha = image[:, :, 3] / 255  # Normalize alpha to 0-1 if it's not already

        #     # Create a white background with the same shape as the original image
        #     background = np.ones_like(rgb, dtype=np.float32)

        #     # Blend the image with the white background based on the alpha channel
        #     image = rgb * alpha[..., None] + background * (1 - alpha[..., None])
        # image = image.transpose(2, 0, 1)

        # image_indices = [index]
        # NUM_SPLATTING_CAMERAS = 2
        # splatting_cameras = []
        # while len(splatting_cameras) < NUM_SPLATTING_CAMERAS:
        #     idx = np.random.randint(0,50)
        #     if idx in image_indices:
        #         continue
        #     _, _, _, image_path, camera = self.get_camera_params(idx,which_zipfile)
        #     image = read_file(which_zipfile,image_path).astype(np.float32) #iio.imread(image_path).astype(np.float32)
        #     # image = image[:,:,:3]/255 # -> images are supposed to be in range 0-1 and no alpha
        #     image = np.asarray(image)
        #     # if image.ndim == 2:  # grayscale to rgb
        #     #     image = image[..., None].repeat(3, 2)

        #     # Normalize the RGB channels to 0-1 range
        #     image[:, :, :3] /= 255

        #     # Check if the image has an alpha channel
        #     if image.shape[2] == 4:
        #         # Separate the RGB and alpha channels
        #         rgb = image[:, :, :3]
        #         alpha = image[:, :, 3] / 255  # Normalize alpha to 0-1 if it's not already

        #         # Create a white background with the same shape as the original image
        #         background = np.ones_like(rgb, dtype=np.float32)

        #         # Blend the image with the white background based on the alpha channel
        #         image = rgb * alpha[..., None] + background * (1 - alpha[..., None])
        #     image = image.transpose(2, 0, 1)
            
        #     splatting_cameras.append((camera,image))
        #     image_indices.append(idx)
        

        example = GaussianExample(
            data=gaussian_pc,
            ctx=GaussianContext3d(
                image=image,
                K=camera_intrinsics.copy(),  # to avoid accidental mutation of self.cmats
                c2w=c2w,
                w2c = wmat,
                camera=camera,
                splatting_cameras = splatting_cameras,
                mask_points=mask_points,
                insinfo=InsInfo(category=self.root.split("/")[-2], instance=self.root.split("/")[-1]),
            ),
        )

        return example
    