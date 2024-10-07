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
import gecco_torch.utils.graphics_utils as graphic_utils
from typing import List
from gecco_torch.structs import GaussianExample, GaussianContext3d, Camera, InsInfo, Mode
from gecco_torch.data.gaussian_pc_dataset_template import GaussianPCModel_Template


NUM_SPLATTING_CAMERAS = 2
IM_SIZE = 400  # 400 x 400 pixels (in 3 x 400 x 400)


class Gaussian_pc_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        root_images: str,
        group: str,
        mode: List[Mode],
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
        val_size: int | None = 10_000,
        single_example: bool = False,
        worker_init_fn = None,
        worker_zipfile_instances = None,
        restrict_to_categories = None,
        include_test_idx = -1, # wenn nicht -1 dann können fürs training auch die test views verwendet werden
        exclude_test_pics = False, # wenn true, dann werden test bilder nicht vom dataloader geladen
        
    ):
        print("Building gaussian pointcloud data module...")
        super().__init__()
        self.root = root
        self.root_images = root_images
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size
        self.val_size = val_size
        self.single_example = single_example
        self.group = group
        self.mode = mode
        self.restrict_to_categories = restrict_to_categories
        self.include_test_idx = include_test_idx
        self.exclude_test_pics = exclude_test_pics

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            print("Setup fit...")
            kw = {"single_view" : False,
                  "single_example" : self.single_example,
                  "group" : self.group,
                  "mode": self.mode,
                  "restrict_to_categories" : self.restrict_to_categories,
                  "include_test_idx" : self.include_test_idx,
                    "exclude_test_pics" : self.exclude_test_pics,
                  }
            self.train = GaussianPC(self.root, self.root_images, "train",**kw)
            print(f"len self train {len(self.train)}")

            self.val = GaussianPC(self.root, self.root_images, "val",**kw)
            print(f"len self val {len(self.val)}")

        elif stage == "test":
            kw = {"single_view" : True,
                  "group" : self.group,
                  "mode": self.mode,
                  "restrict_to_categories" : self.restrict_to_categories,
                  "include_test_idx" : self.include_test_idx,
                  "exclude_test_pics" : self.exclude_test_pics,
                  }
            self.test = GaussianPC(self.root, self.root_images, "test",**kw)

    def train_dataloader(self):
        print("get train dataloader")
        if self.epoch_size is None:
            return torch.utils.data.DataLoader(
                self.train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
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
                self.val, replacement=True, num_samples= self.val_size
            )

        val_dataload = torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
            shuffle=False,
        )
        
        return val_dataload
        

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )
    

class GaussianPC(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        root_images: str,
        split: str, # split bedeutet hier train, val, test
        **kw,
    ):
        print(f"Building GaussianPC for {root}...")
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
            print(f"For subroot {subroot}")
            if kw.get("single_example", False):
                if "0269" not in subroot:
                    continue
            if restrict_to_categories is not None and not kw.get("single_example", False):
                if subroot not in restrict_to_categories:
                    continue
                
            print(f"For subroot {subroot}")
            models.append(GaussianPCClass(os.path.join(root,subroot), os.path.join(root_images,subroot), split, **kw))
        super().__init__(models)

class GaussianPCClass(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        root_images: str,
        split: str,
        **kw,
    ):
        print(f"Building GaussianPCClass for {root}...")
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
        paths = [(os.path.join(root, id),os.path.join(root_images, id)) for id in split_ids]
        if kw.get("single_example", False):
            paths = [('/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133', '/globalwork/giese/shapenet_rendered/02691156/1a04e3eab45ca15dd86060f189eb133')]
            print(paths)
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
        # self.root,self.root_images = root_and_images
        # # print(self.root,self.root_images)
        # # self.root_images = root_images
        # self.single_view = single_view
        # self.tensor_specification= kw["tensor_specification"]
        # implemented = ['normal',"isotropic_rgb"]
        # if self.tensor_specification not in implemented:
        #     raise Exception(f"Not implemented {self.tensor_specification}, not in implemented list: {implemented}")

        # self.wmats, self.cam_intrinsics, self.cameras = None,None,None # (ehemals wmats, cmats), wobei wmat w2c und cmat die camera intrinsics matrix war
        super(GaussianPCModel, self).__init__(root_and_images=root_and_images,
                                              single_view=single_view,
                                              **kw)

    def get_camera_params(self, index: int):
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
                with open(os.path.join(self.root_images, json_split)) as json_file:
                    contents = json.load(json_file)

                    fovx = contents["camera_angle_x"]

                    frames = contents["frames"]
                    for idx, frame in enumerate(frames):
                        f_x = f_y = 437.5
                        c_x = IM_SIZE / 2
                        c_y = IM_SIZE / 2
                        camera_intrinsic_matrix = np.array([
                                                            [f_x, 0, c_x],
                                                            [0, f_y, c_y],
                                                            [0, 0, 1]
                                                          ])


                        image_path = os.path.join(self.root_images, frame["file_path"] + ".png")
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


    # def mult_quats_numpy(self,q1,q2):
    #     a0, a1, a2, a3 = q1[0],q1[1],q1[2],q1[3]
    #     b0, b1, b2, b3 = q2[0],q2[1],q2[2],q2[3]
    #     return np.array([
    #         a0*b0 - a1*b1 - a2*b2 - a3*b3,
    #         a0*b1 + a1*b0 + a2*b3 - a3*b2,
    #         a0*b2 - a1*b3 + a2*b0 + a3*b1,
    #         a0*b3 + a1*b2 - a2*b1 + a3*b0
    #     ])

   
    # def quaternion_to_rotation_matrix(self,q):
    #     r, x, y, z = q[0],q[1],q[2],q[3]
    #     rotation_matrix = np.array([
    #         [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y)],
    #         [2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x)],
    #         [2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)]
    #     ])
    #     return rotation_matrix

    @property
    def pointcloud_npz_path(self):
        # print(os.path.join(self.root, "point_cloud","iteration_10000","point_cloud.ply"))
        return os.path.join(self.root, "point_cloud","iteration_10000","point_cloud.ply")

    def get_gaussian_scene(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        plydata = PlyData.read(self.pointcloud_npz_path)
        return plydata
    
    def prepare_image(self, image_path):
        image = iio.imread(image_path).astype(np.float32)
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
    
    def get_camera_image_tuple(self, index):
        _, _, _, image_path, camera = self.get_camera_params(index)
        image = self.prepare_image(image_path)
        return (camera, image)

    def get_splatting_cameras(self,image_indices, initial_forward_vector, min_dot):
        tried = []
        splatting_cameras = []
        
        while len(splatting_cameras) < NUM_SPLATTING_CAMERAS:
            if len(tried) > 35:
                return splatting_cameras
            # select cameras that are in the opposite half of the dataset -> so at least 110 degrees away from the selected camera
            idx = np.random.randint(0, 50)
            if idx in tried:
                continue
            if idx in image_indices:
                continue
            _, c2w, _, image_path, camera = self.get_camera_params(idx)
            initial_forward_vector_splat_cam = c2w[:3, 2]
            dot = np.dot(initial_forward_vector, initial_forward_vector_splat_cam)
            if dot > min_dot: # smaller than 90 degrees
                tried.append(idx)
                continue
            image = self.prepare_image(image_path)
            # save_image(torch.tensor(image), f"image_sel{idx}.png")
            splatting_cameras.append((camera, image))
            image_indices.append(idx)
        return splatting_cameras
    
    def __getitem__(self, index) -> GaussianExample:
        if self.exclude_test_pics:
            # dont allow index > 45
            if index > 45:
                index = np.random.randint(0,45)
        index = 22
        image_indices = [index]
        wmat, c2w, camera_intrinsics, image_path, camera = self.get_camera_params(index)
        # load 3D pointcloud
        gaussian_pc, mask_points = self.points_world(wmat,self.mode)
        initial_forward_vector = c2w[:3, 2]  # Take the first 3 elements of the third column
        # from torchvision.utils import save_image
        # import torch
        image = self.prepare_image(image_path)


        # save_image(torch.tensor(image), "image.png")

        splatting_cameras = self.get_splatting_cameras(image_indices=image_indices, initial_forward_vector=initial_forward_vector, min_dot = 0)
        if len(splatting_cameras) < NUM_SPLATTING_CAMERAS:
            # wenn nicht genug gefunden wurden (e.g. das initiale bild ist von exakt ganz oben geschossen), mache weichere condition
            splatting_cameras = self.get_splatting_cameras(image_indices=image_indices, initial_forward_vector=initial_forward_vector, min_dot=0.5)

        # falls es gar nicht geklappt hat, auffüllen
        while len(splatting_cameras) < NUM_SPLATTING_CAMERAS:
            # wenn nicht genug gefunden wurden (e.g. das initiale bild ist von exakt ganz oben geschossen), dann mache random
            idx = np.random.randint(0, 50)
            if idx in image_indices:
                continue
            wmat, c2w, camera_intrinsics, image_path, camera = self.get_camera_params(0)
            image = self.prepare_image(image_path)
            splatting_cameras.append((camera, image))
            image_indices.append(idx)
        
        if self.include_test_idx != -1:
            wmat, c2w, camera_intrinsics, image_path, camera = self.get_camera_params(45+self.include_test_idx)
            image = self.prepare_image(image_path)
            splatting_cameras.append((camera, image))
            # iio.imwrite("test1.png",255*image.astype(np.uint8).transpose(1,2,0))

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
    