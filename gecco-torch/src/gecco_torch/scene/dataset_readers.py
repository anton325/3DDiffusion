#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from gecco_torch.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from gecco_torch.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from gecco_torch.utils.sh_utils import SH2RGB,RGB2SH
from gecco_torch.scene.gaussian_model import BasicPointCloud
import pathlib
import imageio.v2 as imageio

import gecco_torch.scene.dataloading.dataloader_SRN as dataloader_SRN
import gecco_torch.scene.dataloading.dataloader_NMR as dataloader_NMR
from gecco_torch.scene.depth_to_pointcloud import create_pointcloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth_map: np.array
    c2w: np.array
    image_orig: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    val_cameras: list
    test_cameras: list
    circle_cameras: list
    spiral_cameras: list
    nerf_normalization: dict
    ply_path: str
    pcd_init_with_depth_map: bool = False

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers) 
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    print("read colmap scene info")
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        print("cameras extrensics ",cam_extrinsics[1]) # dict, {1:Image,...,300:}
                                                           # BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
        print(cam_extrinsics[1].qvec) # (4,)
        print(cam_extrinsics[1].tvec) # (3,)
        print(cam_extrinsics[1].xys.shape) # (11661,2)
        print(cam_extrinsics[1].point3D_ids.shape) # (11661,)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file) # collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
        print("cameras intrinsics ",(cam_intrinsics)) # dict {1:}
        print("cameras intrinsics ",type(cam_intrinsics[1])) # dict {1:}
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           val_cameras=test_cam_infos,
                           test_cameras=test_cam_infos,
                           circle_cameras = test_cam_infos,
                           spiral_cameras = test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png",dimensions_reference_cam = None):
    cam_infos = []
    # print(path)
    if transformsfile == "transforms_circle.json" or transformsfile == "transforms_spiral.json":
        path = pathlib.Path(pathlib.Path.home(),"Documents","gaussian-splatting")

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            image_path = os.path.join(path, frame["file_path"] + extension)
            depth_path = os.path.join(path, frame["file_path"]+ "_depth0001" + extension)
            # print("cam name")
            # print(cam_name)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            c2w_original = c2w.copy()
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            # print("path")
            # print(path)
            # image_path = os.path.join(path, cam_name)
            # print("image path")
            # print(image_path)
            image_name = Path(image_path).stem

            if frame['file_path'] == "Circle" or frame['file_path'] == "Spiral":
                image = Image.fromarray(np.ones(dimensions_reference_cam,dtype=np.uint8))
                try:
                    depth_map = Image.fromarray(np.ones((dimensions_reference_cam.shape[0],dimensions_reference_cam.shape[1]),dtype=np.uint8))
                except:
                    depth_map = Image.fromarray(np.ones(dimensions_reference_cam,dtype=np.uint8))
            else:
                image = Image.open(image_path)
                try:
                    depth_map = np.array(Image.open(depth_path))
                    # print(depth_map)
                    # depth map needs to have 0 as the distance in areas where there is no geometry
                    # Step 2: Extract the alpha channel
                    alpha_channel = np.array(image)[:, :, 3]  # This gets the alpha channel

                    # Step 3: Identify pixels with zero alpha values
                    # This creates a boolean mask where the condition is True for pixels with alpha value 0
                    mask = alpha_channel == 0

                    # Step 4: Set corresponding depth values to 0
                    depth_map[mask] = 0
                except Exception as e:
                    depth_map = None
                    print(e)

            image_orig = np.array(image)
            im_data = np.array(image.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4]) # not necessary in my case
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            

            

            # when image.size[0] != image.size[0] dann haben wir 2 unterschiedliche focal lenghts für x und y
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth_map=depth_map, c2w=c2w_original, image_orig = image_orig,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readSpiralCircleCamInfos(path,white_background,reference_cam):
    circle_cam_infos = readCamerasFromTransforms(path, "transforms_circle.json", white_background, ".png",(reference_cam.width,reference_cam.height,3))
    spiral_cam_infos = readCamerasFromTransforms(path, "transforms_spiral.json", white_background, ".png",(reference_cam.width,reference_cam.height,3))

    circle_cam_infos = [circle_cam_info._replace(FovX = reference_cam.FovX) for circle_cam_info in circle_cam_infos]
    circle_cam_infos = [circle_cam_info._replace(FovY = reference_cam.FovY) for circle_cam_info in circle_cam_infos]
    # circle_cam_infos = [circle_cam_info._replace(width = reference_cam.width) for circle_cam_info in circle_cam_infos]
    # circle_cam_infos = [circle_cam_info._replace(height = reference_cam.height) for circle_cam_info in circle_cam_infos]

    spiral_cam_infos = [spiral_cam_info._replace(FovX = reference_cam.FovX) for spiral_cam_info in spiral_cam_infos]
    spiral_cam_infos = [spiral_cam_info._replace(FovY = reference_cam.FovY) for spiral_cam_info in spiral_cam_infos]
    # spiral_cam_infos = [spiral_cam_info._replace(width = reference_cam.width) for spiral_cam_info in spiral_cam_infos]
    # spiral_cam_infos = [spiral_cam_info._replace(height = reference_cam.height) for spiral_cam_info in spiral_cam_infos]

    return circle_cam_infos,spiral_cam_infos


def readShapenetRenderedInfo(path, white_background, eval, extension=".png",**kwargs):
    # print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(pathlib.Path(path,"train"), "transforms.json", white_background, extension)
    # print("Reading Val Transforms")
    val_cam_infos = readCamerasFromTransforms(pathlib.Path(path,"val"), "transforms.json", white_background, extension)
    # print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(pathlib.Path(path,"test"), "transforms.json", white_background, extension)
    if kwargs['spiral_circle']:
        # print("Reading Circle and spiral Transforms")
        circle_cam_infos,spiral_cam_infos = readSpiralCircleCamInfos(path,white_background,train_cam_infos[0])

    # wir machen immer mit eval split!
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = kwargs['n_init_points'] # orig 100_000, but choose fewer due to smalller images
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    if kwargs['spiral_circle']:
        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            val_cameras=val_cam_infos,
                            test_cameras=test_cam_infos,
                            circle_cameras = circle_cam_infos,
                            spiral_cameras = spiral_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    else:
        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            val_cameras=val_cam_infos,
                            test_cameras=test_cam_infos,
                            circle_cameras = test_cam_infos,
                            spiral_cameras = test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    return scene_info

def readCamsSRN(path,white_background, eval):
    images,poses,intrinsics,shapenet_id,mirror_rays,train_idx,val_idx,test_idx = dataloader_SRN.load_images_poses_focal(path)

    height = images[0].size[0]
    width = images[0].size[1]
    FovY = focal2fov(intrinsics.focal, height) # focal_length_y, but here focallengthx = focallengthy
    FovX = focal2fov(intrinsics.focal, width) # focal_length_x

    cam_infos = []

    for i,(img,pose) in enumerate(zip(images,poses)):
        c2w = pose
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        im_data = np.array(img.convert("RGBA"))
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4]) # not necessary in my case
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")


        cam_infos.append(CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=os.path.join(path,"rgb",str(i).zfill(6)), image_name=str(i).zfill(4), width=width, height=height))
    
    train_cam_infos = [cam_info for i,cam_info in enumerate(cam_infos) if i in train_idx]
    val_cam_infos = [cam_info for i,cam_info in enumerate(cam_infos) if i in val_idx]
    test_cam_infos = [cam_info for i,cam_info in enumerate(cam_infos) if i in test_idx]

    circle_cam_infos,spiral_cam_infos = readSpiralCircleCamInfos(path,white_background,train_cam_infos[0])

    print("----------------------------------------------------")
    print("Loaded {} train".format(len(train_cam_infos)))
    print("Loaded {} val".format(len(val_cam_infos)))
    print("Loaded {} test".format(len(test_cam_infos)))
    print("Loaded {} circle".format(len(circle_cam_infos)))
    print("Loaded {} spiral".format(len(spiral_cam_infos)))

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path): # DO ANYWAYS
        # Since this data set has no colmap data, we start with random points
        num_pts = 1000 # orig 100_000, but choose fewer due to smalller images
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           val_cameras=val_cam_infos,
                           test_cameras=test_cam_infos,
                           circle_cameras = circle_cam_infos,
                           spiral_cameras = spiral_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info

def readCamsNMR(path,white_background, eval):
    images,poses,intrinsics,shapenet_id,mirror_rays,train_idx,val_idx,test_idx = dataloader_NMR.load_images_poses_focal(path)

    height = images[0].size[0]
    width = images[0].size[1]
    FovY = focal2fov(intrinsics.focal, height) # focal_length_y, but here focallengthx = focallengthy
    FovX = focal2fov(intrinsics.focal, width) # focal_length_x

    cam_infos = []

    for i,(img,pose) in enumerate(zip(images,poses)):
        c2w = pose
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        im_data = np.array(img.convert("RGBA"))
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4]) # not necessary in my case
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")


        cam_infos.append(CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=os.path.join(path,"rgb",str(i).zfill(6)), image_name=str(i).zfill(4), width=width, height=height))
    
    train_cam_infos = [cam_info for i,cam_info in enumerate(cam_infos) if i in train_idx]
    val_cam_infos = [cam_info for i,cam_info in enumerate(cam_infos) if i in val_idx]
    test_cam_infos = [cam_info for i,cam_info in enumerate(cam_infos) if i in test_idx]

    circle_cam_infos,spiral_cam_infos = readSpiralCircleCamInfos(path,white_background,train_cam_infos[0])

    print("----------------------------------------------------")
    print("Loaded {} train".format(len(train_cam_infos)))
    print("Loaded {} val".format(len(val_cam_infos)))
    print("Loaded {} test".format(len(test_cam_infos)))
    print("Loaded {} circle".format(len(circle_cam_infos)))
    print("Loaded {} spiral".format(len(spiral_cam_infos)))

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path): # DO ANYWAYS
        # Since this data set has no colmap data, we start with random points
        num_pts = 1000 # orig 100_000, but choose fewer due to smalller images
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           val_cameras=val_cam_infos,
                           test_cameras=test_cam_infos,
                           circle_cameras = circle_cam_infos,
                           spiral_cameras = spiral_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info


def readNerfSyntheticInfo(path, white_background, eval, extension=".png",**kwargs):
    # print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    # print("Reading validation Transforms")
    val_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension)
    # print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    if kwargs['spiral_circle']:
        # print("Reading Circle and spiral Transforms")
        circle_cam_infos,spiral_cam_infos = readSpiralCircleCamInfos(path,white_background,train_cam_infos[0])

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # print("before if")
    # print(ply_path)
    if not os.path.exists(ply_path) or True: # DO ANYWAYS
        # print("in if")
        # Since this data set has no colmap data, we start with random points
        num_pts = kwargs['n_init_points'] # orig 100_000, but choose fewer due to smalller images

        shs = None
        if kwargs['init_method'] == "random":
            print(f"Generating random point cloud ({num_pts})...")
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3 # -0.5 would be better
        elif kwargs['init_method'] == "adjusted":
            print(f"Generating adjusted where the volume is a lot smaller point cloud ({num_pts})...")
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2 - 1 #* 2.6 - 1.3 # -0.5 would be better
        elif kwargs['init_method'] == "reconstruct_from_depth_map":
            print(f"Generating pointcloud using all depth maps ({num_pts})...")
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz,colors = create_pointcloud(train_cam_infos+val_cam_infos+test_cam_infos, num_pts, **kwargs)
            colors = colors/(255)
            shs = RGB2SH(colors)
        if shs is None:
            shs = np.random.random((num_pts, 3)) / 255.0
        # nur eine temporary solution um die informationen später ins gaussian model zu übertragen
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if kwargs['spiral_circle']:
        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            val_cameras=val_cam_infos,
                            test_cameras=test_cam_infos,
                            circle_cameras = circle_cam_infos,
                            spiral_cameras = spiral_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path,
                            pcd_init_with_depth_map = kwargs['init_method'] == "reconstruct_from_depth_map")
    else:
        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            val_cameras=val_cam_infos,
                            test_cameras=test_cam_infos,
                            circle_cameras = test_cam_infos,
                            spiral_cameras = test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path,
                            pcd_init_with_depth_map = kwargs['init_method'] == "reconstruct_from_depth_map")
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "SRN" : readCamsSRN,
    "NMR" : readCamsNMR,
    'ShapenetRender' : readShapenetRenderedInfo,
}