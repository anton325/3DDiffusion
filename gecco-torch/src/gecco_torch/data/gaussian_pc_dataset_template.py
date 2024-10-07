import numpy as np
from tqdm.auto import tqdm
import os
import json
from plyfile import PlyData
import math
import jaxlie
import jax
import jax.numpy as jnp

from gecco_torch.utils.lie_utils import quaternion_to_lietorch_tangential
from gecco_torch.structs import Mode
from gecco_torch.utils.rotation_utils import rotation_matrix_to_quaternion_numpy, mult_quats_vectorized_numpy, quaternion_to_rotation_matrix_scipy
from gecco_torch.utils.build_cov_matrix_torch import reverse_strip_lowerdiag
import gecco_torch.utils.sh_utils as sh_utils

class GaussianPCModel_Template:
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
        implemented = ['normal',"isotropic_rgb"]

        self.wmats, self.cam_intrinsics, self.cameras = None, None ,None # (ehemals wmats, cmats), wobei wmat=w2c und cmat=die camera intrinsics matrix war
        self.mode = kw["mode"]
        self.include_test_idx = kw.get("include_test_idx",-1)
        self.exclude_test_pics = kw.get("exclude_test_pics",False)
        
        self.check_modes(self.mode)

    def check_modes(self,mode):
        incompatible_pairs = [[Mode.normal,Mode.isotropic_rgb],
                              [Mode.normal, Mode.rotation_matrix_mode], 
                              [Mode.normal, Mode.procrustes],
                              [Mode.normal, Mode.only_xyz],
                              [Mode.in_world_space, Mode.in_camera_space],
                              [Mode.isotropic_rgb,Mode.rotation_matrix_mode],
                              [Mode.isotropic_rgb,Mode.procrustes],
                              [Mode.only_xyz,Mode.procrustes],
                              ]
        
        for pairs in incompatible_pairs:
            if pairs[0] in mode and pairs[1] in mode:
                raise Exception(f"es kann nicht {pairs[0]} und {pairs[1]} in mode sein {mode}")

    def fov2focal(self,fov, pixels):
        return pixels / (2 * math.tan(fov / 2)) # sollte das nicht eigentlich sensor width statt pixels sein?????

    def focal2fov(self,focal, pixels):
        return 2*math.atan(pixels/(2*focal))

    def points_world(self,wmat,mode):
        if Mode.normal in mode or Mode.lie_rotations in mode:
            # normal bedeutet erstmal nur 0-2 xyz, 3-5 spherical harmonics (oder generell farbe), 6-8 scaling, 9-12 rotation, 13 opacity
            gaussians,mask_points =  self.points_all(wmat,mode)
        elif Mode.normal_opac in mode:
            gaussians,mask_points =  self.points_all_opac(wmat,mode)
        elif Mode.lie_rotations in mode:
            gaussians,mask_points =  self.points_rotations_xyzw(wmat,mode)
        elif Mode.isotropic_rgb in mode:
            gaussians,mask_points = self.points_world_isotropic_rgb() # keine rotations weil spheres, deswegen auch kein world oder camera mode
        elif Mode.lie_rotations_wrong in mode:
            gaussians,mask_points = self.points_world_lie_wrong(wmat,mode)
        elif Mode.rotation_matrix_mode in mode:
            gaussians,mask_points = self.points_rotation_matrix(wmat,mode)
        elif Mode.log_L in mode:
            gaussians,mask_points = self.points_log_L(wmat,mode)
        elif Mode.only_xyz in mode:
            gaussians,mask_points = self.points_world_onlyxyz()
        elif Mode.fill_xyz in mode:
            gaussians,mask_points = self.points_world_fillxyz()
        elif Mode.xyz_sh in mode:
            gaussians,mask_points = self.points_xyz_sh(wmat,mode)
        elif Mode.xyz_scaling in mode:
            gaussians,mask_points = self.points_xyz_scales(wmat,mode)
        elif Mode.xyz_rotation in mode:
            gaussians,mask_points = self.points_xyz_rotation(wmat,mode)
        elif Mode.xyz_opacity in mode:
            gaussians,mask_points = self.points_xyz_opacity(wmat,mode)
        elif Mode.xyz_scaling_rotation in mode:
            gaussians,mask_points = self.pointpoints_xyz_scaling_rotations(wmat,mode)
        elif Mode.normal_gt_rotations in mode:
            gaussians,mask_points = self.normal_gt_rotations(wmat,mode)
        elif Mode.cov_matrix_3x3 in mode:
            gaussians,mask_points = self.points_cov_matrix_3x3(wmat,mode)
        elif Mode.procrustes in mode:
            gaussians,mask_points = self.points_rot_matrix_3x3(wmat,mode)

        elif Mode.gt_rotations_canonical in mode:
            gaussians,mask_points = self.gt_rotations_canonical(wmat,mode)
        elif Mode.so3_diffusion in mode:
            gaussians, mask_points = self.points_rotations_xyzw(wmat,mode)
        elif Mode.so3_x0 in mode:
            gaussians, mask_points = self.points_rotations_xyzw(wmat,mode)
        elif Mode.cholesky in mode:
            gaussians, mask_points = self.points_cholesky(wmat,mode)
        elif Mode.no_rotation in mode:
            gaussians, mask_points = self.points_all_no_rotation(wmat,mode)
        elif Mode.activated_scales in mode:
            gaussians, mask_points = self.points_activated_scales(wmat,mode)
        elif Mode.activated_lie in mode:
            gaussians, mask_points = self.points_activated_scales_lie(wmat,mode)
        else:
            raise Exception(f"points world: Not implemented {mode}")
        # transform from world coordinates to camera coordinates für das projizieren auf das bild -> wmat indicates where the camera is located w.r.t world origin
        if Mode.in_camera_space in self.mode:
            gaussians[:,:3] = np.einsum("ab,nb->na", wmat[:, :3], gaussians[:,:3]) + wmat[:, -1] # adding translation part
        return gaussians,mask_points
        
    def __len__(self):
        if self.single_view:
            return 1
        else:
            return 50
        
    @property
    def pointcloud_npz_path(self):
        # print(os.path.join(self.root, "point_cloud","iteration_10000","point_cloud.ply"))
        return os.path.join(self.root, "point_cloud","iteration_10000","point_cloud.ply")

    def get_gaussian_scene(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        plydata = PlyData.read(self.pointcloud_npz_path)
        return plydata

    def points_all(self,wmat,mode):
        """
        der tensor besteht aus 14 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6,7,8: scaling
        9, 10, 11, 12: rotation (als quaternion)
        13: opacity
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32) # wxyz format
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,features_dc,scales,rots,opacities),dtype=np.float32)
        return gaussian_pc_tensor, mask_points

    def points_all_opac(self,wmat,mode):
        """
        der tensor besteht aus 14 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6,7,8: scaling
        9, 10, 11, 12: rotation (als quaternion)
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32) # wxyz format
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,features_dc,scales,rots),dtype=np.float32)
        return gaussian_pc_tensor, mask_points
    
    def points_activated_scales(self,wmat,mode):
        """
        der tensor besteht aus 14 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6,7,8: scaling -> aktiviert
        9, 10, 11, 12: rotation (als quaternion)
        13: opacity
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32) # wxyz format
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        scales_activated = np.exp(scales)
        gaussian_pc_tensor = np.hstack((xyz,features_dc,scales_activated,rots,opacities),dtype=np.float32)
        return gaussian_pc_tensor, mask_points
    
    def points_activated_scales_lie(self,wmat,mode):
        """
        der tensor besteht aus 14 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6,7,8: scaling -> aktiviert
        9: opacity
        10, 11, 12, 13: rotation (als quaternion) -> IN XYZW format
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32) # wxyz format
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        scales_activated = np.exp(scales)
        rots = rots[:,[1,2,3,0]] # -> xyzw
        gaussian_pc_tensor = np.hstack((xyz,features_dc,scales_activated, opacities, rots),dtype=np.float32)
        return gaussian_pc_tensor, mask_points
    
    def points_all_no_rotation(self,wmat,mode):
        """
        der tensor besteht aus 14 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6,7,8: scaling
        9: opacity
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32) # wxyz format
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,features_dc,scales,opacities),dtype=np.float32)
        return gaussian_pc_tensor, mask_points
    
    def points_cholesky(self,wmat,mode):
        """
        der tensor besteht aus 14 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6: opacity
        7,8,9: scaling
        10, 11, 12, 13: rotation (als quaternion) -> ganz normal in wxyz
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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

        """
        aktiviere scales
        """

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation


        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -5)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[0.8694,0.4830,0.0386,0.0966]], dtype=np.float32) # wxyz format
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        scales_exp = np.exp(scales)
        gaussian_pc_tensor = np.hstack((xyz,features_dc,opacities,scales_exp,rots),dtype=np.float32)
        return gaussian_pc_tensor, mask_points
    
    def points_rotations_xyzw(self,wmat,mode):
        """
        der tensor besteht aus 14 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6,7,8: scaling
        9: opacity
        10, 11, 12, 13: rotation (als quaternion) -> IN XYZW CONVENTION
        """

        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        """
        XYZW
        """
        rots = rots[:,[1,2,3,0]] # wxyz -> xyzw

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[0, 0, 0, 1]], dtype=np.float32) # xyzw format
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000 - num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz, features_dc, scales, opacities, rots), dtype=np.float32)
        return gaussian_pc_tensor, mask_points
    
    def gt_rotations_canonical(self,wmat,mode):
        """
        der tensor besteht aus 14 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6,7,8: scaling
        9: opacity
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # sortiere scales [größtes, zweitgrößtes, kleinstes]
        scales = np.sort(scales, axis=1)[:, ::-1]

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

        if xyz.shape[0] > target_num_points:
            num_points_to_delete = xyz.shape[0] - target_num_points
            for _ in range(num_points_to_delete):
                # nicht random löschen, sonst aligned das nicht
                xyz = xyz[:-1]
                features_dc = features_dc[:-1]
                opacities = opacities[:-1]
                scales = scales[:-1]
                rots = rots[:-1]

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32)
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,features_dc,scales,opacities),dtype=np.float32)
        return gaussian_pc_tensor, mask_points
    
    def normal_gt_rotations(self,wmat,mode):
        """
        der tensor besteht aus 10 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6,7,8: scaling
        9: opacity
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

        if xyz.shape[0] > target_num_points:
            num_points_to_delete = xyz.shape[0] - target_num_points
            for _ in range(num_points_to_delete):
                # Sample a random index
                # index_to_delete = np.random.randint(0, xyz.shape[0])
                # xyz = np.delete(xyz, index_to_delete, axis=0)
                # features_dc = np.delete(features_dc, index_to_delete, axis=0)
                # opacities = np.delete(opacities, index_to_delete, axis=0)
                # scales = np.delete(scales, index_to_delete, axis=0)
                # rots = np.delete(rots, index_to_delete, axis=0)
                
                # nicht random löschen, sonst aligned das nicht
                xyz = xyz[:-1]
                features_dc = features_dc[:-1]
                opacities = opacities[:-1]
                scales = scales[:-1]
                rots = rots[:-1]



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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32)
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        gaussian_pc_tensor = np.hstack((xyz,features_dc,scales,opacities),dtype=np.float32)
        return gaussian_pc_tensor, mask_points
    
    def pointpoints_xyz_scaling_rotations(self,wmat,mode):
        """
        der tensor besteht aus 10 dimensionen:
        0,1,2 : xyz
        3,4,5: scaling
        6,7,8,9 rotation (als quaternion)
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32)
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,scales,rots),dtype=np.float32)
        return gaussian_pc_tensor, mask_points

    def points_xyz_sh(self,wmat,mode):
        """
        der tensor besteht aus 6 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32)
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,features_dc),dtype=np.float32)

        return gaussian_pc_tensor, mask_points
    
    def points_xyz_scales(self,wmat,mode):
        """
        der tensor besteht aus 6 dimensionen:
        0,1,2 : xyz
        3,4,5: scaling
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32)
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,scales),dtype=np.float32)
        return gaussian_pc_tensor, mask_points
    
    def points_xyz_rotation(self,wmat,mode):
        """
        der tensor besteht aus 6 dimensionen:
        0,1,2 : xyz
        3,4,5, 6: rotation
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32)
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,rots),dtype=np.float32)

        return gaussian_pc_tensor, mask_points
    
    def points_xyz_opacity(self,wmat,mode):
        """
        der tensor besteht aus 6 dimensionen:
        0,1,2 : xyz
        3,4,5, 6: rotation
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0]], dtype=np.float32)
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,opacities),dtype=np.float32)

        return gaussian_pc_tensor, mask_points
    
    def points_world_onlyxyz(self):
        """
        der tensor besteht aus 3 dimensionen:
        0,1,2 : xyz
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        target_num_points = 4000
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

        if xyz.shape[0] > target_num_points:
            num_points_to_delete = xyz.shape[0] - target_num_points
            for _ in range(num_points_to_delete):
                # Sample a random index
                index_to_delete = np.random.randint(0, xyz.shape[0])
                xyz = np.delete(xyz, index_to_delete, axis=0)

        elif xyz.shape[0] < target_num_points:
            num_points_to_add = target_num_points - xyz.shape[0]
            for _ in range(num_points_to_add):
                # Random location generation
                location = np.zeros((1, 3))  # Assuming similar to the torch.zeros(3) but reshaped for concatenation
                xyz = np.concatenate((xyz, location), axis=0)

            mask_points = 4000-num_points_to_add

        # shape 4000, 3
        gaussian_pc_tensor = np.hstack((xyz[None,...]),dtype=np.float32)

        return gaussian_pc_tensor, mask_points
    
    def points_world_fillxyz(self):
        """
        der tensor besteht aus 14 dimensionen:
        0,1,2 : xyz
        3-13: 0
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        target_num_points = 4000
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

        if xyz.shape[0] > target_num_points:
            num_points_to_delete = xyz.shape[0] - target_num_points
            for _ in range(num_points_to_delete):
                # Sample a random index
                index_to_delete = np.random.randint(0, xyz.shape[0])
                xyz = np.delete(xyz, index_to_delete, axis=0)

        elif xyz.shape[0] < target_num_points:
            num_points_to_add = target_num_points - xyz.shape[0]
            for _ in range(num_points_to_add):
                # Random location generation
                location = np.zeros((1, 3))  # Assuming similar to the torch.zeros(3) but reshaped for concatenation
                xyz = np.concatenate((xyz, location), axis=0)

            mask_points = 4000-num_points_to_add

        # shape 4000, 3
        gaussian_pc_tensor = np.hstack((xyz,np.zeros((xyz.shape[0],11))),dtype=np.float32)

        return gaussian_pc_tensor, mask_points
    
    def SH2RGB(self,sh):
            C0 = 0.28209479177387814
            return sh * C0 + 0.5
    
    def points_world_isotropic_rgb(self):
        """
        der tensor besteht aus 8 dimensionen:
        0,1,2 : xyz
        3,4,5: rgb
        6: scaling (isotropic)
        7: opacity

        scaling haben wir nur eins, das muss verdreifacht werden beim rendern
        rotation haben wir gar keins, wenns eh ne kugel ist. muss mit [1,0,0,0] beim rendern gefüllt werden
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        rgb = self.SH2RGB(features_dc)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        scales_isotropic = np.median(scales, axis=1).reshape(-1, 1)

        target_num_points = 4000
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

        if xyz.shape[0] > target_num_points:
            num_points_to_delete = xyz.shape[0] - target_num_points
            for _ in range(num_points_to_delete):
                # Sample a random index
                index_to_delete = np.random.randint(0, xyz.shape[0])
                xyz = np.delete(xyz, index_to_delete, axis=0)
                rgb = np.delete(rgb, index_to_delete, axis=0)
                opacities = np.delete(opacities, index_to_delete, axis=0)
                scales_isotropic = np.delete(scales_isotropic, index_to_delete, axis=0)


        elif xyz.shape[0] < target_num_points:
            num_points_to_add = target_num_points - xyz.shape[0]
            for _ in range(num_points_to_add):
                # Random location generation
                location = np.zeros((1, 3))  # Assuming similar to the torch.zeros(3) but reshaped for concatenation
                xyz = np.concatenate((xyz, location), axis=0)

                # Zero features
                rgb_null = np.zeros((1, 3))
                rgb = np.concatenate((rgb, rgb_null), axis=0)

                # Opacity
                opacity = np.full((1, 1), -10.0)  # Fill array with -10.0 for opacity
                opacities = np.concatenate((opacities, opacity), axis=0)

                # Scaling
                scaling = np.full((1, 1), -10)
                scales_isotropic = np.concatenate((scales_isotropic, scaling), axis=0)

            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,rgb,scales_isotropic,opacities),dtype=np.float32)

        return gaussian_pc_tensor, mask_points
    
    def points_world_lie_wrong(self,wmat,mode):
        """
        der resulting tensor besteht aus 13 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6,7,8: scaling
        9, 10, 11: rotation -> in der lie algebra als skew matrix, die aus 3 werten besteht
        12: opacity
        """

        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)
        
        rots = quaternion_to_lietorch_tangential(rots)

        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[0, 0, 0]], dtype=np.float32) # 0,0,0 im tangential raum entspricht dem 1,0,0,0 quaternion
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,features_dc,scales,rots,opacities),dtype=np.float32)

        return gaussian_pc_tensor, mask_points

    
    def quaternion_to_rotation_matrix(self,q):
            # Assuming q is of shape (N, 4) with (w, x, y, z) format
            N = q.shape[0]
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            R = np.zeros((N, 3, 3))
            R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
            R[:, 0, 1] = 2 * (x*y - z*w)
            R[:, 0, 2] = 2 * (x*z + y*w)
            R[:, 1, 0] = 2 * (x*y + z*w)
            R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
            R[:, 1, 2] = 2 * (y*z - x*w)
            R[:, 2, 0] = 2 * (x*z - y*w)
            R[:, 2, 1] = 2 * (y*z + x*w)
            R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
            return R
    
    def points_rotation_matrix(self,wmat,mode):
        """
        Dieser Modus ist für lie space gedacht. Also die Rotation wird als 3x3 rotations matrix dargestellt

        der tensor besteht aus 19 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features
        6,7,8: scaling
        9, 10, 11, 12, 13, 14, 15, 16, 17: rotation (als normale 3x3 rotation matrix)
        18: opacity
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)

        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        if Mode.in_camera_space in mode:
            w2c_rotation = wmat[:3,:3]
            w2c_quat = rotation_matrix_to_quaternion_numpy(w2c_rotation)
            w2c_quat = np.tile(w2c_quat,(xyz.shape[0],1)) # broadcast to appropriate shape for vectorized multiplication
            # for i in range(xyz.shape[0]):
                # rotation_matrix = self.quaternion_to_rotation_matrix(rots[i,:])
                # new_rotation = np.matmul(w2c_rotation,rotation_matrix)
                # new_quat = rotation_matrix_to_quaternion_numpy(new_rotation)
                # rots[i,:] = self.mult_quats_torch_numpy(w2c_quat,rots[i,:])
            rots = mult_quats_vectorized_numpy(w2c_quat,rots)

        # stelle die quaternion als rotation matrix dar
        rots = self.quaternion_to_rotation_matrix(rots)

        # rots = quaternion_to_rotation_matrix_scipy(rots)
        rots = rots.reshape((xyz.shape[0],9))


        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

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
                scaling = np.full((1, 3), -10)
                scales = np.concatenate((scales, scaling), axis=0)

                # Rotation
                rot = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.float32) # the rotation matrix with ones on the diagonal
                rots = np.concatenate((rots, rot), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000 - num_points_to_add

        # shape 4000,19
        gaussian_pc_tensor = np.hstack((xyz,features_dc,scales,rots),dtype=np.float32)

        return gaussian_pc_tensor, mask_points



    def build_rotation(self,r):
        norm = np.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        R = np.zeros((q.shape[0], 3, 3))

        r = q[:, 0] # real part
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R
    
    def build_scaling_rotation(self,s, r):
        L = np.zeros((s.shape[0], 3, 3))
        R = self.build_rotation(r)

        L[:,0,0] = s[:,0]
        L[:,1,1] = s[:,1]
        L[:,2,2] = s[:,2]

        L = R @ L
        return L

    def strip_lowerdiag(self,L):
        """
        3x3 matrix
        [0,0] [0,1] [0,2]
        [1,0] [1,1] [1,2]
        [2,0] [2,1] [2,2]
        """
        uncertainty = np.zeros((L.shape[0], 6))

        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]

        return uncertainty

    def build_covariance_from_scaling_rotation(self,scaling, rotation):
        L = self.build_scaling_rotation(scaling, rotation)
        # torch actual_covariance = L @ L.transpose(1, 2)
        actual_covariance = L @ L.transpose(0, 2, 1)
        symm = self.strip_lowerdiag(actual_covariance)
        return symm

    def points_log_L(self,wmat,mode):
        """
        der tensor besteht aus 13 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features (or rgb)
        6: opacity
        7, 8, 9, 10, 11, 12: lower triangle cholesky
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        cov = self.build_covariance_from_scaling_rotation(np.exp(scales),rots)
        cov_matrices = np.zeros((cov.shape[0], 3, 3))

        # Assign the elements directly using advanced indexing
        cov_matrices[:, 0, 0] = cov[:, 0]
        cov_matrices[:, 0, 1] = cov[:, 1]
        cov_matrices[:, 0, 2] = cov[:, 2]
        cov_matrices[:, 1, 0] = cov[:, 1]  # Symmetric element
        cov_matrices[:, 1, 1] = cov[:, 3]
        cov_matrices[:, 1, 2] = cov[:, 4]
        cov_matrices[:, 2, 0] = cov[:, 2]  # Symmetric element
        cov_matrices[:, 2, 1] = cov[:, 4]  # Symmetric element
        cov_matrices[:, 2, 2] = cov[:, 5]

        """
        cov_matrices
        cov[0],cov[1],cov[2]
        cov[1],cov[3],cov[4]
        cov[2],cov[4],cov[5]
        """
        Ls = np.zeros((cov.shape[0],6))
        epsilon = 1e-10  # Small positive value
        # make symmetric
        cov_matrices = (cov_matrices + np.transpose(cov_matrices, (0, 2, 1))) / 2 
        cov_matrices += np.eye(3) * epsilon
        for i,matrix in enumerate(cov_matrices):
            try:
                L = np.linalg.cholesky(matrix)
            except:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices[i])
                eigenvalues[eigenvalues < 0] = 0  # Correct negative eigenvalues
                matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                L = np.linalg.cholesky(matrix)
            # die diagonal werte log, weil wir die später exp
            Ls[i,0] = np.log(L[0,0])
            Ls[i,1] = np.log(L[1,1])
            Ls[i,2] = np.log(L[2,2])
            Ls[i,3] = L[1,0]
            Ls[i,4] = L[2,0]
            Ls[i,5] = L[2,1]

        if Mode.in_camera_space in mode:
            raise Exception("in camera space not implemented für cov matrix prediction")
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

        if xyz.shape[0] > target_num_points:
            num_points_to_delete = xyz.shape[0] - target_num_points
            for _ in range(num_points_to_delete):
                # Sample a random index
                index_to_delete = np.random.randint(0, xyz.shape[0])
                xyz = np.delete(xyz, index_to_delete, axis=0)
                features_dc = np.delete(features_dc, index_to_delete, axis=0)
                opacities = np.delete(opacities, index_to_delete, axis=0)
                Ls = np.delete(Ls, index_to_delete, axis=0)


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

                # lower triangle 
                L = np.array([[-5, -5, -5, 0.001, 0.001, 0.001]], dtype=np.float32)
                Ls = np.concatenate((Ls, L), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,12
        gaussian_pc_tensor = np.hstack((xyz,features_dc,opacities, Ls),dtype=np.float32)

        return gaussian_pc_tensor, mask_points
    
    def points_cov_matrix_3x3(self,wmat,mode):
        """
        der tensor besteht aus 16 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features (or rgb)
        6: opacity
        7, 8, 9, 10, 11, 12, 13, 14, 15 entire covariance matrix
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        cov = self.build_covariance_from_scaling_rotation(np.exp(scales),rots)
        cov_matrices = np.zeros((cov.shape[0], 3, 3))

        # Assign the elements directly using advanced indexing
        cov_matrices[:, 0, 0] = cov[:, 0]
        cov_matrices[:, 0, 1] = cov[:, 1]
        cov_matrices[:, 0, 2] = cov[:, 2]
        cov_matrices[:, 1, 0] = cov[:, 1]  # Symmetric element
        cov_matrices[:, 1, 1] = cov[:, 3]
        cov_matrices[:, 1, 2] = cov[:, 4]
        cov_matrices[:, 2, 0] = cov[:, 2]  # Symmetric element
        cov_matrices[:, 2, 1] = cov[:, 4]  # Symmetric element
        cov_matrices[:, 2, 2] = cov[:, 5]
        cov_matrices = cov_matrices.reshape(cov.shape[0],9)
        """
        cov_matrices
        cov[0],cov[1],cov[2]
        cov[1],cov[3],cov[4]
        cov[2],cov[4],cov[5]
        """
        # Ls = np.zeros((cov.shape[0],6))
        # epsilon = 1e-10  # Small positive value
        # # make symmetric
        # cov_matrices = (cov_matrices + np.transpose(cov_matrices, (0, 2, 1))) / 2 
        # cov_matrices += np.eye(3) * epsilon
        # for i,matrix in enumerate(cov_matrices):
        #     try:
        #         L = np.linalg.cholesky(matrix)
        #     except:
        #         eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices[i])
        #         eigenvalues[eigenvalues < 0] = 0  # Correct negative eigenvalues
        #         matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        #         L = np.linalg.cholesky(matrix)
        #     # die diagonal werte log, weil wir die später exp
        #     Ls[i,0] = np.log(L[0,0])
        #     Ls[i,1] = np.log(L[1,1])
        #     Ls[i,2] = np.log(L[2,2])
        #     Ls[i,3] = L[1,0]
        #     Ls[i,4] = L[2,0]
        #     Ls[i,5] = L[2,1]

        if Mode.in_camera_space in mode:
            raise Exception("in camera space not implemented für cov matrix prediction")
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

        if xyz.shape[0] > target_num_points:
            num_points_to_delete = xyz.shape[0] - target_num_points
            for _ in range(num_points_to_delete):
                # Sample a random index
                index_to_delete = np.random.randint(0, xyz.shape[0])
                xyz = np.delete(xyz, index_to_delete, axis=0)
                features_dc = np.delete(features_dc, index_to_delete, axis=0)
                opacities = np.delete(opacities, index_to_delete, axis=0)
                cov_matrices = np.delete(cov_matrices, index_to_delete, axis=0)


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

                # lower triangle 
                cov3x3 = np.array([[0.0001,0,0,0,0.0001,0,0,0,0.0001]], dtype=np.float32)
                cov_matrices = np.concatenate((cov_matrices, cov3x3), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz, features_dc, opacities, cov_matrices),dtype=np.float32)

        return gaussian_pc_tensor, mask_points
    

    def points_rot_matrix_3x3(self,wmat,mode):
        """
        der tensor besteht aus 16 dimensionen:
        0,1,2 : xyz
        3,4,5: spherical harmonics features (or rgb)
        6: opacity
        7,8,9: scaling
        10, 11, 12, 13, 14, 15, 16, 17, 18 3x3 rotation matrix
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        rot_3x3 = np.array(jax.vmap(lambda x: jaxlie.SO3(jnp.array(x)).as_matrix())(rots))
        rot_9 = rot_3x3.reshape(-1,9)

        if Mode.in_camera_space in mode:
            raise Exception("in camera space not implemented für cov matrix prediction")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

        if xyz.shape[0] > target_num_points:
            num_points_to_delete = xyz.shape[0] - target_num_points
            for _ in range(num_points_to_delete):
                # Sample a random index
                index_to_delete = np.random.randint(0, xyz.shape[0])
                xyz = np.delete(xyz, index_to_delete, axis=0)
                features_dc = np.delete(features_dc, index_to_delete, axis=0)
                opacities = np.delete(opacities, index_to_delete, axis=0)
                scales = np.delete(scales, index_to_delete, axis=0)
                rot_9 = np.delete(rot_9, index_to_delete, axis=0)


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

                scale = np.full((1, 3), -10)
                scales = np.concatenate((scales, scale), axis=0)
                # lower triangle 
                r_9 = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.float32) # the rotation matrix with ones on the diagonal
                rot_9 = np.concatenate((rot_9, r_9), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz, features_dc, opacities, scales, rot_9),dtype=np.float32)

        return gaussian_pc_tensor, mask_points
    
    def points_cov_matrix_no_col_opac(self,wmat,mode):
        """
        der tensor besteht aus 14 dimensionen:
        0,1,2 : xyz
        6, 7, 8, 9, 10, 11: lower triangle covariance matrix
        """
        plydata = self.get_gaussian_scene()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = features_dc.reshape(-1,3)
        if Mode.rgb in mode:
            features_dc = self.SH2RGB(features_dc)

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
            rotation = np.asarray(plydata.elements[0][attr_name])
            rots[:, idx] = rotation

        cov = self.build_covariance_from_scaling_rotation(np.exp(scales),rots)
        cov_matrices = np.zeros((cov.shape[0], 3, 3))

        # Assign the elements directly using advanced indexing
        cov_matrices[:, 0, 0] = cov[:, 0]
        cov_matrices[:, 0, 1] = cov[:, 1]
        cov_matrices[:, 0, 2] = cov[:, 2]
        cov_matrices[:, 1, 0] = cov[:, 1]  # Symmetric element
        cov_matrices[:, 1, 1] = cov[:, 3]
        cov_matrices[:, 1, 2] = cov[:, 4]
        cov_matrices[:, 2, 0] = cov[:, 2]  # Symmetric element
        cov_matrices[:, 2, 1] = cov[:, 4]  # Symmetric element
        cov_matrices[:, 2, 2] = cov[:, 5]

        """
        cov_matrices
        cov[0],cov[1],cov[2]
        cov[1],cov[3],cov[4]
        cov[2],cov[4],cov[5]
        """
        Ls = np.zeros((cov.shape[0],6))
        epsilon = 1e-10  # Small positive value
        # make symmetric
        cov_matrices = (cov_matrices + np.transpose(cov_matrices, (0, 2, 1))) / 2 
        cov_matrices += np.eye(3) * epsilon
        for i,matrix in enumerate(cov_matrices):
            try:
                L = np.linalg.cholesky(matrix)
            except:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices[i])
                eigenvalues[eigenvalues < 0] = 0  # Correct negative eigenvalues
                matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                L = np.linalg.cholesky(matrix)
            # die diagonal werte log, weil wir die später exp
            Ls[i,0] = np.log(L[0,0])
            Ls[i,1] = np.log(L[1,1])
            Ls[i,2] = np.log(L[2,2])
            Ls[i,3] = L[1,0]
            Ls[i,4] = L[2,0]
            Ls[i,5] = L[2,1]

        if Mode.in_camera_space in mode:
            raise Exception("in camera space not implemented für cov matrix prediction")
        # pc = GaussianModel(3)
        # pc.create_from_values(xyz = torch.tensor(xyz), features_dc=torch.tensor(features_dc), opacity = torch.tensor(opacities),scaling=torch.tensor(scales),rotation=torch.tensor(rots))
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done/point_cloud/iteration_30000/point_cloud.ply")

        # die 4000 ist nur ca, es kann leichte abweichungen geben
        target_num_points = 4000
        # mask_attn_pool = torch.ones((8, 64, target_num_points), dtype=torch.bool) # für torch scaled dot product -> bei True wird attended
        # torch.zeros((target_num_points,64),dtype=torch.bool)  # für torch multihead attention -> bei false wird attended
        mask_points = 4000 # 4000 sagt es wurde nix gepadded  

        if xyz.shape[0] > target_num_points:
            num_points_to_delete = xyz.shape[0] - target_num_points
            for _ in range(num_points_to_delete):
                # Sample a random index
                index_to_delete = np.random.randint(0, xyz.shape[0])
                xyz = np.delete(xyz, index_to_delete, axis=0)
                features_dc = np.delete(features_dc, index_to_delete, axis=0)
                opacities = np.delete(opacities, index_to_delete, axis=0)
                Ls = np.delete(Ls, index_to_delete, axis=0)


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

                # lower triangle 
                L = np.array([[1, 1, 1, 0, 0, 0]], dtype=np.float32)
                Ls = np.concatenate((Ls, L), axis=0)

            # mask_attn_pool[:, :, target_num_points-num_points_to_add:] = False # ignore those

            # jetzt die maske für die query in der multihead attention
            # mask_multihead[target_num_points-num_points_to_add:,:] = True # ignore those
            mask_points = 4000-num_points_to_add

        # shape 4000,14
        gaussian_pc_tensor = np.hstack((xyz,Ls),dtype=np.float32)

        return gaussian_pc_tensor, mask_points