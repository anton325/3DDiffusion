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

from gecco_torch.structs import Example, GaussianContext3d

IM_SIZE = 400  # 137 x 137 pixels
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
    ):
        self.root = root
        self.n_points = n_points
        self.single_view = single_view

        self.wmats, self.cmats = None, None

    def get_camera_params(self, index: int):
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
        # transform from world coordinates to camera coordinates? -> wmat indicates where the camera is located w.r.t world origin
        points_view = np.einsum("ab,nb->na", wmat[:, :3], points_world) + wmat[:, -1] # adding translation part

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

        return Example(
            data=points_view,
            ctx=GaussianContext3d(
                image=image,
                K=cmat.copy(),  # to avoid accidental mutation of self.cmats
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
        paths = [os.path.join(root, id) for id in split_ids][:100]
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
                if not "0269" in str(maybe_dir_path):
                    continue
                subroots.append(maybe_dir_path)
            print(f"split {split} len subroots {len(subroots)}")
            print(f"Building ShapeNetVolClass for subroots:\n{subroots} ")
            models = []
            for subroot in subroots:
                print(f"For subroot {subroot}")
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
    ):
        print("Building ShapenetCondDataModule...")
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
            print("Setup fit...")
            self.train = ShapeNetVol(self.root, "train", **kw)
            print(f"len self train {len(self.train)}")
            
            self.val = ShapeNetVol(self.root, "val", **kw)
            print(f"len self val {len(self.val)}")
        elif stage == "test":
            kw["single_view"] = True
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
            shuffle=False,
        )

class SingleClassShapenetCondDataModule(pl.LightningDataModule):
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