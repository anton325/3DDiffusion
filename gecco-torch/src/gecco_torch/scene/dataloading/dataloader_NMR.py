import os
import numpy as np
import pathlib
import imageio.v2 as imageio
import typing
from PIL import Image

from gecco_torch.scene.dataloading.transformations import switch_y_z_axis_single,change_all_axis_viewing_direction_single,get_translation,rotate_z_axis_single,change_x_axis_viewing_direction_single,rotate_y_axi_single,rotate_x_axis_single

from gecco_torch.scene.dataloading.intrinsics import Intrinsics


DIMENSION_IMAGE = 64
NEAR = 1
FAR = 7

def get_intrinsics(focal):
    return Intrinsics(DIMENSION_IMAGE,DIMENSION_IMAGE,focal,NEAR,FAR)

def custom_sort_key_(el):
    return int(el.split("_")[-1])


def load_focal_length(path:str) -> np.array:
    cameras = np.load(pathlib.Path(path,"cameras.npz"))
    camera_matrices_names = list(filter(lambda x: "camera" in x and not "inv" in x, list(sorted(cameras.keys(),key=custom_sort_key_))))
    # assert that they are all the same
    for cm1 in camera_matrices_names:
        assert cameras[cm1][0,0] == cameras[cm1][1,1], "Focal length x and focal length y are not the same"
        for cm2 in camera_matrices_names:
            assert (cameras[cm1] == cameras[cm2]).all(), "Camera matrices should be the same"
    return np.array([(cameras[camera_matrices_names[0]][0,0] + cameras[camera_matrices_names[0]][1,1])/2],dtype=np.float32)

def load_poses(path:str) -> np.array:
    cameras = np.load(pathlib.Path(path,"cameras.npz"))
    world_matrices_names = list(filter(lambda x: "world" in x and "inv" in x, list(sorted(cameras.keys(),key=custom_sort_key_))))
    poses = np.zeros((len(world_matrices_names),4,4),dtype=np.float32)
    for i,wm in enumerate(world_matrices_names):
        poses[i] = cameras[wm]
        poses[i][:3, 1:3] *= -1

        # do weird transformations

        # IMPORTANT TO make the viewing field of pose correct
        # poses[i] = switch_y_z_axis_single() @ poses[i]
        # poses[i] = rotate_x_axis_single(2) @ poses[i]
        # poses[i] = change_x_axis_viewing_direction_single() @ poses[i]


        # # FOURTH WAY
        # location = (poses[i][:3,-1]).copy()
        # poses[i] = get_translation(-location[0],-location[1],-location[2]) @ poses[i]
        # poses[i] = change_all_axis_viewing_direction_single() @ poses[i]
        # poses[i] = get_translation(location[0],location[1],location[2]) @ poses[i]
        # poses[i] = rotate_z_axis_single(-1) @ poses[i]
        # poses[i] = change_x_axis_viewing_direction_single() @ poses[i]
        # # poses[i] = rotate_x_axis_single(2) @ poses[i]
        # poses[i] = rotate_y_axi_single(2) @ poses[i]
        # poses[i] = rotate_z_axis_single(2) @ poses[i]
        # poses[i] = change_x_axis_viewing_direction_single() @ poses[i]







        # --------- DEPRECATED this part probably doesnt work correct, might also work but wasnt tested
        # poses[i] = switch_y_z_axis_single() @ poses[i]

        # location = (poses[i][:3,-1]).copy()
        # poses[i] = get_translation(-location[0],-location[1],-location[2]) @ poses[i]
        # poses[i] = change_all_axis_viewing_direction_single() @ poses[i]
        # poses[i] = get_translation(location[0],location[1],location[2]) @ poses[i]
        # poses[i] = rotate_z_axis_single(-1) @ poses[i] # 90 degree rotation around z-axis
        # poses[i] = change_x_axis_viewing_direction_single() @ poses[i]

    return poses

# def load_images(path:str) -> np.array:
#     files = list(sorted(filter(lambda x: ".png" in x, os.listdir(path))))
#     images = np.zeros((len(files),DIMENSION_IMAGE,DIMENSION_IMAGE,3),dtype=np.float32)#np.uint8)
#     # images = np.zeros((len(files),DIMENSION_IMAGE,DIMENSION_IMAGE,3),dtype=np.uint8)
#     for i,file in enumerate(files):
#         im = imageio.imread(pathlib.Path(path,file))#.astype(np.uint8)
#         #print(im.shape)
#         images[i] = im/255
#         #imageio.imsave("here.png",im)
#     # print(images.dtype)
#     return images

def load_images(path:str) -> np.array:
    files = list(sorted(filter(lambda x: ".png" in x, os.listdir(path))))
    images = []
    for i,file in enumerate(files):
        # im = imageio.imread(pathlib.Path(path,file))
        # print(im.shape)
        im = Image.open(pathlib.Path(path,file))
        images.append(im)
        # print(im.size)
        # imageio.imsave(pathlib.Path(path,"img_{}.png".format(str(i).zfill(4))),im[:,:,:3])
        # images[i] = im[:,:,:3]/255
        # _=[print(image) for image in images[i]]
        # print(images[i])
    return images
    
        

def load_images_poses_focal(root_path,set_id:str = None,model_id:str = None,test_split:str = None) -> typing.Tuple[np.array,np.array,np.array,str,bool,int,int]:
    """
    return
      images,
      poses, 
      name of diretory stuff gets saved in, 
      boolean indicating if rays should be mirrored -> yes for nmr datasets, no for everything else,
      How many examples to use for training,
      How many examples to use for testing
    """
    if set_id == None:
        set_id = pathlib.Path(str(root_path).split("/")[-2],str(root_path).split("/")[-1])
        root_path = pathlib.Path(root_path,"..","..")

    if test_split == None:
        test_split = "train" # other option is test
    assert test_split == "train" or test_split == "test" , "Test split should either be train or test"

    if model_id == None:
        # was called from kplanes, where all information is in set_id like cars_test/1079efee042629d4ce28f0f1b509eda
        model_id = str(set_id).split("/")[1]
        set_id = str(set_id).split("/")[0]
    
    if test_split == "test":
        set_id = set_id.split("_")[0] + "_test"

    path_to_model = pathlib.Path(root_path,set_id,model_id)

    focal = load_focal_length(path_to_model)
    focal = np.array([70],dtype=np.float32)
    intrinsics = get_intrinsics(focal)
    poses = load_poses(path_to_model)
    images = load_images(pathlib.Path(path_to_model,"image"))
    # assert images.shape[0] == poses.shape[0], "Different number of poses than of images"
    total_number_of_samples = poses.shape[0]
    vals_idx = np.array([0,6,12,18])
    test_idx = np.array([8])
    train_idx = [i for i in np.arange(total_number_of_samples) if i not in test_idx and i not in vals_idx]
    # return images, poses, name of diretory stuff gets saved in, boolean indicating if rays should be mirrored -> yes for nmr datasets, no for everything else
    return images,poses,intrinsics,set_id+"_"+model_id,True,train_idx,vals_idx,test_idx

if __name__ == "__main__":
    load_images_poses_focal("02691156","1a04e3eab45ca15dd86060f189eb133")


# general dataloading: dictionary mit name mapped zu der dazugehörigen funktion