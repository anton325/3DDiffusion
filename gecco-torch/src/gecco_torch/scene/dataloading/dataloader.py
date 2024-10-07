import os
import pathlib
import numpy as np
import json
import imageio.v2 as imageio

from gecco_torch.scene.dataloading.transformations import get_translation,change_all_axis_viewing_direction_single,switch_y_z_axis_single, \
                                pose_spherical,rotate_z_axis_single,rotate_x_axis_single,rotate_y_axi_single, \
                                change_x_axis_viewing_direction_single 
import base64
import tqdm

def custom_sort_key_mildenhall(el):
    return int(el.split("_")[-1].split(".")[0])

def _nmr_blender_load_pngs(path):
    png_files = list(sorted(list(pathlib.Path(path).glob("*.png"))))
    images = []
    for img in tqdm.tqdm(png_files,desc="Reading images"):
        with open(pathlib.Path(path,img), "rb") as image_file:
                img_data = base64.b64encode(image_file.read())
                shape_img= imageio.imread(image_file).shape
                img_data = img_data.decode()
                img_data = "{}{}".format("data:image/jpg;base64, ", img_data)
                images.append(img_data)

    return images,shape_img

def _mildenhall_bagger_load_pngs(path):
    png_files = list(sorted([str(pic) for pic in pathlib.Path(path).glob("*.png")],key=custom_sort_key_mildenhall))
    images = []
    for img in tqdm.tqdm(png_files,desc="Reading images"):
        with open(pathlib.Path(path,img), "rb") as image_file:
                img_data = base64.b64encode(image_file.read())
                shape_img= imageio.imread(image_file).shape
                img_data = img_data.decode()
                img_data = "{}{}".format("data:image/jpg;base64, ", img_data)
                images.append(img_data)

    return images,shape_img


def custom_sort_key_(el):
    return int(el.split("_")[-1])

def _load_pose_json(path:str) -> np.array:
    # rendered by blender
    f = open(pathlib.Path(path,"transforms.json"))
    json_file = json.load(f)
    poses = np.zeros((len(json_file['frames']),4,4),dtype=np.float32)
    for i,transformation_matrix in enumerate(json_file['frames']):
        # poses[i] = rotate_z_axis_2(-0*np.pi/2) @ transformation_matrix['transform_matrix']
        # poses[i] = rotate_z_axis(transformation_matrix['transform_matrix'],-1)
        poses[i] = transformation_matrix['transform_matrix']
        poses[i] = rotate_z_axis_single(-1) @ poses[i]
    return poses

def c2w_to_origin_and_direction(world_matrix):
        dirs = np.sum([0, 0, -1] * world_matrix[:3, :3], axis=-1)
        origins = world_matrix[:3, -1]
        return origins,dirs

def _load_pose_npz(path:str) -> np.array:
    """
    Function to load the poses of the NMR dataset 
    However, as they don't follow the same conventions as for example the bagger poses, which are the
    poses this visualisation and NeRF is tuned to, we have to perform a number of transformations.
    When training the NeRF, this also includes mirroring the rays on the vertical center of the image.
    """
    cameras = np.load(pathlib.Path(path,"cameras.npz"))
    world_matrices_names = list(filter(lambda x: "world" in x and "inv" in x, list(sorted(cameras.keys(),key=custom_sort_key_))))
    poses = np.zeros((len(world_matrices_names),4,4),dtype=np.float32)
    for i,wm in enumerate(world_matrices_names):
        poses[i] = cameras[wm]

        # IMPORTANT: to rotate the viewing field of pose correct 
        # (i.e. top left corner, bottom left corner and right top corner are where they're supposed to be)
        poses[i] = switch_y_z_axis_single() @ poses[i]
        poses[i] = rotate_x_axis_single(2) @ poses[i]
        poses[i] = change_x_axis_viewing_direction_single() @ poses[i]


        # FOURTH WAY
        location = (poses[i][:3,-1]).copy()
        poses[i] = get_translation(-location[0],-location[1],-location[2]) @ poses[i]
        poses[i] = change_all_axis_viewing_direction_single() @ poses[i]
        poses[i] = get_translation(location[0],location[1],location[2]) @ poses[i]
        poses[i] = rotate_z_axis_single(-1) @ poses[i]
        poses[i] = change_x_axis_viewing_direction_single() @ poses[i]
        # poses[i] = rotate_x_axis_single(2) @ poses[i]
        poses[i] = rotate_y_axi_single(2) @ poses[i]
        poses[i] = rotate_z_axis_single(2) @ poses[i]
        poses[i] = change_x_axis_viewing_direction_single() @ poses[i]
        
        # THIRD WAY - DIDNT WORK, ROTATES RAYS AROUND Y AND X AXIS
        # location,_ = c2w_to_origin_and_direction(poses[i]) #(poses[i][:3,-1]).copy()
        # location = location.copy()
        # poses[i] = get_translation(-location[0],-location[1],-location[2]) @ poses[i]  

        # _,dir = c2w_to_origin_and_direction(poses[i])
        # rotations_x = np.arctan(dir[1]/dir[2]) if dir[2] != 0 else 0
        # rotation_degrees = 180*rotations_x/np.pi
    
        # poses[i] = rotate_x_axis_single(2*rotation_degrees/90) @ poses[i]
        # location = rotate_x_axis_single(2*rotation_degrees/90) @ np.append(location,1)

        # poses[i] = get_translation(-location[0],-location[1],-location[2]) @ poses[i]
        # poses[i] = rotate_z_axis_single(-1) @ poses[i] # 

        # SECOND WAY - DIDNT WORK
        # poses[i] = switch_y_z_axis_single() @ poses[i]

        # location,_ = c2w_to_origin_and_direction(poses[i]) #(poses[i][:3,-1]).copy()
        # location = location.copy()
        # poses[i] = get_translation(-location[0],-location[1],-location[2]) @ poses[i]
        # poses[i] = rotate_z_axis_single(2) @ poses[i] # 90 degree rotation around z-axis

        # # rotate around x and y axis:
        # # around y axis -> arctan(z/y)
      
        # _,dir = c2w_to_origin_and_direction(poses[i])
        # rotations_y = np.arctan(dir[2]/dir[0]) if dir[1] != 0 else 0
      
        # rotation_degrees = 180*rotations_y/np.pi
        # poses[i] = rotate_y_axi_single(2*rotation_degrees/90) @ poses[i] 

        # # around x axis -> arctan(z/x) (evtl vorzeichen verändern und mal zwei)
        # _,dir = c2w_to_origin_and_direction(poses[i])
        # rotations_x = np.arctan(dir[1]/dir[2]) if dir[2] != 0 else 0
        # rotation_degrees = 180*rotations_x/np.pi
    
        # poses[i] = rotate_x_axis_single(2*rotation_degrees/90) @ poses[i]
        # location = rotate_x_axis_single(2*rotation_degrees/90) @ np.append(location,1)


        # poses[i] = get_translation(location[0],location[1],location[2]) @ poses[i]
        # poses[i] = rotate_z_axis_single(-1) @ poses[i] # 
       


        # - first way -> didnt work
        # poses[i] = switch_y_z_axis_single() @ poses[i]
        # # poses[i] = change_all_axis_viewing_direction_single() @ poses[i]
        # location = (poses[i][:3,-1]).copy()
        # poses[i] = get_translation(-location[0],-location[1],-location[2]) @ poses[i]
        # poses[i] = change_all_axis_viewing_direction_single() @ poses[i]
        # poses[i] = get_translation(location[0],location[1],location[2]) @ poses[i]
        # poses[i] = rotate_z_axis_single(-1) @ poses[i]
        # poses[i] = change_x_axis_viewing_direction_single() @ poses[i]
        # # poses[i] = rotate_x_axis_single(2) @ poses[i]
        # poses[i] = rotate_y_axi_single(2) @ poses[i]
        
    return poses

def _load_pose_bagger(path:str) -> np.array:
    return np.load(pathlib.Path(path,"tiny_nerf_data.npz"))['poses']

def _load_poses_txt(path:str) -> np.array:
    txt_files = list(sorted(list(pathlib.Path(path).glob("*.txt"))))
    poses = np.zeros((len(txt_files),4,4),dtype=np.float32)
    for i,txt_file in enumerate(txt_files):
        poses[i] = _load_pose_txt(pathlib.Path(path,txt_file))
        poses[i][:3, 1:3] *= -1
        # poses[i] = rotate_x_axis_single(2) @ poses[i]
        # poses[i] = change_x_axis_viewing_direction_single() @ poses[i]

        # location = (poses[i][:3,-1]).copy()
        # poses[i] = get_translation(-location[0],-location[1],-location[2]) @ poses[i]
        # poses[i] = change_all_axis_viewing_direction_single() @ poses[i]
        # poses[i] = get_translation(location[0],location[1],location[2]) @ poses[i]
        # poses[i] = rotate_z_axis_single(-1) @ poses[i]
        # poses[i] = change_x_axis_viewing_direction_single() @ poses[i]
        # poses[i] = rotate_y_axi_single(2) @ poses[i]
        # poses[i] = rotate_z_axis_single(1) @ poses[i]
    return poses

def _load_pose_txt(path_to_txt:str) -> np.array:
    with open(path_to_txt, 'r') as file:
        contents = file.read()
        pose = np.array([float(x) for x in contents.split(" ")]).reshape(4,4)
        # pose = pose.transpose()
        return pose
    
def create_spherical_viewpoints():
    num_views = 24
    poses = np.zeros((num_views,4,4))
    for i, theta in enumerate(np.linspace(0.0, 360.0, num_views, endpoint=False)): # num_views is number of points between 0 and 360
        poses[i] = pose_spherical(theta, phi = -30.0, t = 2.732)
    return poses

def load_poses(path):
    """
    Main part of this loader, figure out what kind of data we're dealing with and calling the corresponding function
    """
    print(path)
    if path == "synthetic_viewpoints":
        return create_spherical_viewpoints()
    
    elif pathlib.Path(path,"transforms.json").exists():
        # read transforms routine
        print("transforms json dir")
        return _load_pose_json(path)
    
    elif pathlib.Path(path,"cameras.npz").exists():
        # NMR dataset with npz camera world matrices
        print("npz dir")
        return _load_pose_npz(path)
    
    elif len(list(pathlib.Path(path).glob("*.txt"))) > 0:
        # each pose has single txt file with 16 numbers that can be rowwise gathered into a matrix
        print("txt dir")
        return _load_poses_txt(path)
    
    elif pathlib.Path(path,"tiny_nerf_data.npz").exists():
        # bagger dataset
        print("npz bagger")
        return _load_pose_bagger(path)
    
    else:
        print("Could not figure out what kind of data we're dealing with")

def load_images(path):
    print(path)
    if pathlib.Path(path,"0000.png").exists() or pathlib.Path(path,"000.png").exists() or pathlib.Path(path,"000000.png").exists():
        # NMR or blender
        print("NMR or blender render or SRN images")
        return _nmr_blender_load_pngs(path)
    elif pathlib.Path(path,"r_0.png").exists():
        return _mildenhall_bagger_load_pngs(path)

