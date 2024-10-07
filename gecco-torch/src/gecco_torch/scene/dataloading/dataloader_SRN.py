import os
import numpy as np
import pathlib
import imageio.v2 as imageio
from PIL import Image
import typing

from gecco_torch.scene.dataloading.transformations import switch_y_z_axis_single,change_all_axis_viewing_direction_single,get_translation,rotate_z_axis_single,change_x_axis_viewing_direction_single,rotate_y_axi_single,rotate_x_axis_single
from gecco_torch.scene.dataloading.dataloader import _load_poses_txt

import pickle

from gecco_torch.scene.dataloading.intrinsics import Intrinsics

DIMENSION_IMAGE = 128
NEAR = 1
FAR = 7

def get_intrinsics(focal):
    return Intrinsics(DIMENSION_IMAGE,DIMENSION_IMAGE,focal,NEAR,FAR)

def load_train_val_test_split(path:str):
    with open(pathlib.Path(path,"train_test_val.pkl"),"rb") as f:
        train_val_test_split = pickle.load(f)
    return train_val_test_split

def custom_sort_key_(el):
    return int(el.split("_")[-1])


def load_focal_length(path:str) -> np.array:
    with open(pathlib.Path(path,"intrinsics.txt"),"r") as file:
        focal = np.array([float(file.read().split(" ")[0])],dtype=np.float32)
        return focal
    
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
    
# def load_poses(path:str) -> np.array:
#     poses[i] = rotate_x_axis_single(2) @ poses[i]
#     poses[i] = change_x_axis_viewing_direction_single() @ poses[i]

#     location = (poses[i][:3,-1]).copy()
#     poses[i] = get_translation(-location[0],-location[1],-location[2]) @ poses[i]
#     poses[i] = change_all_axis_viewing_direction_single() @ poses[i]
#     poses[i] = get_translation(location[0],location[1],location[2]) @ poses[i]
#     poses[i] = rotate_z_axis_single(-1) @ poses[i]
#     poses[i] = change_x_axis_viewing_direction_single() @ poses[i]
#     poses[i] = rotate_y_axi_single(2) @ poses[i]
#     poses[i] = rotate_z_axis_single(1) @ poses[i]
        

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
    # focal = np.array([45],dtype=np.float32)
    intrinsics = get_intrinsics(focal)
    poses = _load_poses_txt(pathlib.Path(path_to_model,"pose"))
    images = load_images(pathlib.Path(path_to_model,"rgb"))

    assert len(images) == len(poses), "Different number of poses than of images"

    train_val_test_split = load_train_val_test_split(root_path)

    train_idx = train_val_test_split[set_id][model_id]['train']
    val_idx = train_val_test_split[set_id][model_id]['val']
    test_idx = train_val_test_split[set_id][model_id]['test']

    return images,poses,intrinsics,set_id+"_"+model_id,True,train_idx,val_idx,test_idx

# if __name__ == "__main__":
#     load_images_poses_focal("cars_test","1079efee042629d4ce28f0f1b509eda")