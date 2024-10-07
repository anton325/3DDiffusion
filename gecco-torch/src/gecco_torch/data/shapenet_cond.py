import os
import pathlib
import re
from typing import NamedTuple, Union, List
from functools import partial

import torch
import numpy as np
import imageio as iio
import multiprocess as mp
import lightning.pytorch as pl
from tqdm.auto import tqdm

from gecco_torch.structs import Example, Context3d

IM_SIZE = 137  # 137 x 137 pixels
WORLD_MAT_RE = re.compile(r"world_mat_(\d+)")
CAMERA_MAT_RE = re.compile(r"camera_mat_(\d+)")



# data class for ONE single pointcloud
# gives you the pointcloud and one image that you can specify by index
# returns Example, consisting of Example(data=pointcloud of shape (2048,3),
#                                        ctx=Context3d(image, K))
class ShapeNetVolModel:
    def __init__(
        self,
        root: str,
        n_points: int = 2048,
        single_view: bool = False,
        limit_val: int = None,
        **kw,
    ):
        self.root = root
        self.n_points = n_points
        self.single_view = single_view
        self.downsample_points = kw["downsample_points"]

        self.wmats, self.cmats = None, None

    def get_camera_params(self, index: int):
        """
        ******CAMERA MATRIX REFERS TO CAMERA INTRINSIC MATRIX******
        """
        if self.wmats is None:
            npz = np.load(os.path.join(self.root, "img_choy2016", "cameras.npz"))

            world_mat_ids = set()
            camera_mat_ids = set()

            for key in npz.keys():
                if (m := WORLD_MAT_RE.match(key)) is not None:
                    world_mat_ids.add(int(m.group(1)))
                    continue
                if (m := CAMERA_MAT_RE.match(key)) is not None:
                    camera_mat_ids.add(int(m.group(1)))
                    continue

            assert world_mat_ids == camera_mat_ids

            indices = np.array(sorted(list(world_mat_ids)))
            if (indices != np.arange(24)).all():
                raise AssertionError("Bad shapenet model")

            world_mats = np.stack([npz[f"world_mat_{i}"] for i in indices])
            camera_mats = np.stack([npz[f"camera_mat_{i}"] for i in indices])

            # normalize camera matrices
            # print("normalize")
            # print(camera_mats)
            camera_mats /= np.array([IM_SIZE + 1, IM_SIZE + 1, 1]).reshape(3, 1)
            # print(camera_mats)

            self.wmats = world_mats.astype(np.float32)
            self.cmats = camera_mats.astype(np.float32)
            # print(f"wmats shape {self.wmats.shape}") # (24, 3, 4)
            # print(f"cmats shape {self.cmats.shape}") # (24, 3, 3)

        return self.wmats[index], self.cmats[index]

    @property
    def pointcloud_npz_path(self):
        return os.path.join(self.root, "pointcloud.npz")

    def points_scale_loc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.pointcloud_npz_path) as pc:
            points = pc["points"].astype(np.float32)
            scale = pc["scale"].astype(np.float32)
            loc = pc["loc"].astype(np.float32)

        return points, scale, loc

    def points_world(self):
        points, scale, loc = self.points_scale_loc()
        # print(f"has {points.shape} points ")
        if self.downsample_points:
            if self.n_points is not None:
                subset = np.random.permutation(points.shape[0])[: self.n_points]
                points = points[subset]
        return points * scale + loc[None, :]

    def __len__(self):
        if self.single_view:
            return 1
        else:
            return 24

    def __getitem__(self, index) -> Example:
        wmat, cmat = self.get_camera_params(index)

        # load 3D pointcloud
        points_world = self.points_world()
        
        # transform from world coordinates to camera coordinates -> wmat indicates where the camera is located w.r.t world origin
        # wmat is world to camera matrix?
        # points_view = np.einsum("ab,nb->na", wmat[:, :3], points_world) + wmat[:, -1] # adding translation part

        image_index = index
        image_path = os.path.join(
            self.root,
            "img_choy2016",
            f"{image_index:03d}.jpg",
        )
        image = iio.imread(image_path).astype(np.float32) / 255
        image = np.asarray(image)
        if image.ndim == 2:  # grayscale to rgb
            image = image[..., None].repeat(3, 2)
        # m = 0
        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         print(image[i,j])
        #         print(m)
        #         m = max(m, max(image[i,j]))
        # print(f"max pixel value {m}")
        # images are in range 0-1

        image = image.transpose(2, 0, 1)
        # print(image.shape)
        # print(wmat)
        w2c4x4 = np.concatenate([wmat,np.array([[0,0,0,1]],dtype=wmat.dtype)],axis=0)
        return Example(
            data=points_world,
            ctx=Context3d(
                image=image,
                K=cmat.copy(),  # to avoid accidental mutation of self.cmats
                w2c=w2c4x4,
                category=self.root.split("/")[-2] if type(self.root) == type("string") else self.root.parent.name,
                # wmat=wmat.copy(),
            ),
        )


class ShapeNetVolClass(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        split: str,
        **kw,
    ):
        print(f"Building ShapeNetVolClass for {root}...")
        with open(os.path.join(root, f"{split}.lst")) as split_file:
            split_ids = [line.strip() for line in split_file]
        paths = [os.path.join(root, id) for id in split_ids]
        if kw.get("single_example", False):
            paths = ['/globalwork/giese/gecco_shapenet/ShapeNet/02691156/414f3305033ad38934f276985b6d695']
            print(paths)
        make_model = partial(ShapeNetVolModel, **kw)

        if kw.get("posed", False) or kw.get("skip_fixed", False):
            # takes a while to load npzs to it's good to parallelize
            with mp.Pool() as pool:
                subsets = list(pool.imap(make_model, paths))
        else:
            # faster to not pay multiprocess overhead
            subsets = list(map(make_model, paths))

        super().__init__(subsets)
        self.root = root
        self.split = split


class ShapeNetVol(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        split: Union[str, List[str]], # split bedeutet hier train, val, test
        **kw,
    ):
        print(f"Building ShapeNetVol for {root}...")
        if isinstance(split, str):
            subroots = []
            for maybe_dir in os.listdir(root):
                maybe_dir_path = os.path.join(root, maybe_dir)
                if not os.path.isdir(maybe_dir_path):
                    continue
                # if not "0269" in str(maybe_dir_path):
                #     continue
                subroots.append(maybe_dir_path)
            print(f"split {split} len subroots {len(subroots)}")
            print(f"Building ShapeNetVolClass for subroots:\n{subroots} ")
            models = []
            for subroot in subroots:
                print(f"For subroot {subroot}")
                if kw.get("single_example", False):
                    if "0269" not in subroot:
                        continue
                models.append(ShapeNetVolClass(subroot, split, **kw))
            # models = [
                # ShapeNetVolClass(subroot, split, **kw) for subroot in tqdm(subroots)
            # ]
        else: # else split is a LIST of strings
            assert isinstance(split, (list, tuple))
            assert all(isinstance(path, str) for path in split)

            models = [ShapeNetVolModel(path, **kw) for path in tqdm(split)]

        super().__init__(models)


class ShapenetCondDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        n_points: int = 2048,
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
        val_size: int | None = 10_000,
        single_example: bool = False,
        downsample_points:bool = True
    ):
        print("Building ShapenetCondDataModule...")
        super().__init__()
        self.root = root
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size
        self.val_size = val_size
        self.single_example = single_example
        self.downsample_points = downsample_points

    def setup(self, stage: str = "fit"):
        kw = dict(n_points=self.n_points)
        if stage == "fit":
            kw["single_view"] = False
            kw["single_example"] = self.single_example
            kw["downsample_points"] = self.downsample_points
            print("Setup fit...")
            self.train = ShapeNetVol(self.root, "train", **kw)
            print(f"len self train {len(self.train)}")
            
            self.val = ShapeNetVol(self.root, "val", **kw)
            print(f"len self val {len(self.val)}")
        elif stage == "test":
            kw["single_view"] = True
            kw["downsample_points"] = self.downsample_points
            self.test = ShapeNetVol(self.root, "test", **kw)

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
                self.val, replacement=True, num_samples=self.val_size
            )

        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

class SingleClassShapenetSCondDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str, # root to class folder
        n_points: int = 2048,
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
        val_size: int | None = 10_000,
    ):
        super().__init__()
        self.root = root
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size
        self.val_size = val_size

    def setup(self, stage: str = "fit"):
        kw = dict(n_points=self.n_points)
        if stage == "fit":
            self.train = SingleClassShapeNetDataset(self.root, "train", **kw)
            self.val = SingleClassShapeNetDataset(self.root, "val", **kw)
        elif stage == "test":
            kw["single_view"] = True
            self.test = SingleClassShapeNetDataset(self.root, "test", **kw)

    def train_dataloader(self):
        if self.epoch_size is None:
            return torch.utils.data.DataLoader(
                self.train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
            )

        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=torch.utils.data.RandomSampler(
                self.train, replacement=True, num_samples=self.epoch_size
            ),
            pin_memory=True,
            shuffle=False,
        )

    def val_dataloader(self):
        if self.val_size is None:
            sampler = None
        else:
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
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

class SingleClassShapeNetDataset(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root: str,
        split: str,
        **kw,
    ):
        
        paths_to_instances = os.listdir(pathlib.Path(root))
        with open(os.path.join(root, f"{split}.lst")) as split_file:
            instances_split = [line.strip() for line in split_file]
        paths_to_instances = [pathlib.Path(root,instance_path) for instance_path in paths_to_instances if instance_path in instances_split]

        models = [ShapeNetVolModel(path, **kw) for path in paths_to_instances]

        super().__init__(models)



"""

GAUSSIAN PART: Nehme unser gaussian scene dataset und mache splatting NUR auf den XYZ koordinaten um zu schauen, ob das alles funktioniert

"""
# gaussian
import gecco_torch.utils.graphics_utils as graphic_utils
import math
from gecco_torch.structs import GaussianExample, GaussianContext3d, Camera
# from gecco_torch.scene.gaussian_model import GaussianModel
import json
from plyfile import PlyData


# IM_SIZE = 400  # 400 x 400 pixels

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2)) # sollte das nicht eigentlich sensor width statt pixels sein?????

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class Gaussian_pc_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        root_images: str,
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
        val_size: int | None = 10_000,
        single_example: bool = False,
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

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            print("Setup fit...")
            kw = {"single_view" : False,
                  "single_example" : self.single_example}
            self.train = GaussianPC(self.root, self.root_images, "train",**kw)
            print(f"len self train {len(self.train)}")

            self.val = GaussianPC(self.root, self.root_images, "val",**kw)
            print(f"len self val {len(self.val)}")

        elif stage == "test":
            kw = {"single_view" : True}
            self.test = GaussianPC(self.root, self.root_images, "test",kw)

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
                self.val, replacement=True, num_samples=self.val_size
            )

        val_dataloader = torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
            shuffle=False,
        )
        print(f"In get val dataloader len val_dataloader {len(val_dataloader)}")
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        print(f"In get test dataloader len test_dataloader {len(test_dataloader)}")
        return test_dataloader
    

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
        for subroot in subroots:
            print(f"For subroot {subroot}")
            if kw.get("single_example", False):
                if "0269" not in subroot:
                    continue
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
        with open(os.path.join(root, f"{split}.lst")) as split_file:
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

class GaussianPCModel:
    def __init__(
        self,
        root_and_images: str,
        # root_images: str,
        single_view: bool = False,
        **kw
    ):
        self.root,self.root_images = root_and_images
        # print(self.root,self.root_images)
        # self.root_images = root_images
        self.single_view = single_view

        self.wmats, self.cam_intrinsics, self.cameras = None,None,None # (ehemals wmats, cmats), wobei wmat w2c und cmat die camera intrinsics matrix war

    def get_camera_params(self, index: int):
        """
        ******CAMERA MATRIX REFERS TO CAMERA INTRINSIC MATRIX******
        """
        if self.wmats is None:
            im_size = 400
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
                        c_x = im_size/2
                        c_y = im_size/2
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
                        fovy = focal2fov(fov2focal(fovx, im_size), im_size)
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
                                     imsize = im_size)
                        

                        # gecco operations
                        # c2w = c2w[:3,:3]
                        camera_intrinsic_matrix /= np.array([im_size + 1, im_size + 1, 1]).reshape(3, 1)

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

    @property
    def pointcloud_npz_path(self):
        # print(os.path.join(self.root, "point_cloud","iteration_10000","point_cloud.ply"))
        return os.path.join(self.root, "point_cloud","iteration_10000","point_cloud.ply")

    def points_scale_loc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        plydata = PlyData.read(self.pointcloud_npz_path)
        return plydata
        

    def points_world(self):
        plydata = self.points_scale_loc()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3 # -3 because the first 3 are not in f_rest but in f_dc
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        # gaussian_pc = {
        #     "xyz": xyz, # shape (4000,3)
        #     "features_dc": features_dc, # shape (4000,3,1)
        #     "scales": scales, # shape (4000,3)
        #     "rots": rots, # shape (4000,4)
        #     "opacities": opacities, # shape (4000,1)
        # }
        target_num_points = 4000
        if xyz.shape[0] > target_num_points:
            num_points_to_delete = xyz.shape[0] - target_num_points
            for _ in range(num_points_to_delete):
                # Sample a random index
                index_to_delete = np.random.randint(0, xyz.shape[0])
                xyz = np.delete(xyz, index_to_delete, axis=0)
                features_dc = np.delete(features_dc, index_to_delete, axis=0)
                opacities = np.delete(opacities, index_to_delete, axis=0)
                scales = np.delete(scales, index_to_delete, axis=0)
                rots = np.delete(rots, index_to_delete, axis=0)

        elif xyz.shape[0] < target_num_points:
            num_points_to_add = target_num_points - xyz.shape[0]
            for _ in range(num_points_to_add):
                # Random location generation
                location = np.zeros((1, 3))  # Assuming similar to the torch.zeros(3) but reshaped for concatenation
                xyz = np.concatenate((xyz, location), axis=0)

                # Zero features
                features = np.zeros((1, 3))
                features_dc = np.concatenate((features_dc, features), axis=0)

                # Opacity
                opacity = np.full((1, 1), -10.0)  # Fill array with -10.0 for opacity
                opacities = np.concatenate((opacities, opacity), axis=0)

                # Scaling
                scaling = np.full((1, 3), 0.0001)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32)
                rots = np.concatenate((rots, rot), axis=0)

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,features_dc,scales,rots,opacities),dtype=np.float32)

        return gaussian_pc_tensor

    def __len__(self):
        if self.single_view:
            return 1
        else:
            return 50

    def __getitem__(self, index) -> GaussianExample:
        wmat,c2w, camera_intrinsics, image_path, camera = self.get_camera_params(index)
        # print(wmat.shape)
        # print(cmat.shape)
        # load 3D pointcloud
        gaussian_pc = self.points_world()
        # print(gaussian_pc.shape)
        
        # transform from world coordinates to camera coordinates für das projizieren auf das bild -> wmat indicates where the camera is located w.r.t world origin
        # gaussian_pc[:,:3] = np.einsum("ab,nb->na", wmat[:, :3], gaussian_pc[:,:3]) + wmat[:, -1] # adding translation part

        image = iio.imread(image_path).astype(np.float32)
        # image = image[:,:,:3]/255 # -> images are supposed to be in range 0-1 and no alpha
        image = np.asarray(image)
        # if image.ndim == 2:  # grayscale to rgb
        #     image = image[..., None].repeat(3, 2)

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

        gaussian_pc = gaussian_pc[:,:3]
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = gaussian_pc[:,0:3], features_dc=gaussian_pc[:,3:6], opacity = gaussian_pc[:,13],scaling=gaussian_pc[:,6:9],rotation=gaussian_pc[:,9:13])
        # bg = torch.tensor([0.0,0.0,0.0],device=torch.device('cuda'))
        # print(f"shape {camera.world_view_transform.shape}") # (1,4,4)
        # cam = Camera(
        #     world_view_transform = torch.tensor(camera.world_view_transform),
        #     projection_matrix = torch.tensor(camera.projection_matrix),
        #     tanfovx = torch.tensor(camera.tanfovx),
        #     tanfovy = torch.tensor(camera.tanfovy),
        #     imsize = torch.tensor(camera.imsize),
        # )
        # print(cam)
        # print(render_dict['render'])
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")
        # print("saved")
        # pc.load_ply("/home/giese/Documents/gaussian-splatting/output/done30000/point_cloud/iteration_30000/point_cloud.ply")

        return GaussianExample(
            data=gaussian_pc,
            ctx=Context3d(
                image=image,
                K=camera_intrinsics.copy(),  # to avoid accidental mutation of self.cmats
                w2c = wmat,
                category = str(self.root).split("/")[-2]
            ),
        )