from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pickle

import torch
import numpy as np
import random
import glob
import os, sys
from scipy.spatial.transform import Rotation as R
import smplx
from human_body_prior.tools.model_loader import load_vposer
import torch.nn.functional as F
import torchgeometry as tgm

sys.path.append(os.getcwd())
# from models.fittingop import RotConverter
from models.baseops import RotConverter
from exp_GAMMAPrimitive.utils.utils_canonicalize_babel import get_body_model, marker_ssm_67
import pytorch3d
import pytorch3d.structures
import pytorch3d.transforms
import pdb
import trimesh
import pickle
from copy import deepcopy
from mesh_to_sdf import mesh_to_voxels, get_surface_point_cloud
import pyrender
from pathlib import Path

def apply_rot_noise(rot, noise):
    t, d = rot.shape
    rot = pytorch3d.transforms.axis_angle_to_matrix(rot.reshape(-1, 3))
    noise = pytorch3d.transforms.axis_angle_to_matrix(noise.reshape(-1, 3))
    result = torch.matmul(noise, rot)
    return pytorch3d.transforms.matrix_to_axis_angle(result).reshape(t, d)

unity_to_zup = np.array(
            [[-1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )

shapenet_to_zup = np.array(
            [[1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )

zup_to_shapenet = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]]
)

class BatchGeneratorAMASSCanonicalized(object):
    def __init__(self,
                amass_data_path,
                amass_subset_name=None,
                sample_rate=3,
                body_repr='cmu_41', #['smpl_params', 'cmu_41', 'ssm2_67', 'joint_location', 'bone_transform' ]
                read_to_ram=True
                ):
        self.rec_list = list()
        self.index_rec = 0
        self.amass_data_path = amass_data_path
        self.amass_subset_name = amass_subset_name
        self.sample_rate = sample_rate
        self.data_list = []
        self.jts_list = []
        self.body_repr = body_repr
        self.read_to_ram = read_to_ram
        self.max_len = 100 if 'x10' in amass_data_path else 10

        self.bm_batch_male = get_body_model('smplx', 'male', self.max_len, device='cuda')
        self.bm_batch_female = get_body_model('smplx', 'female', self.max_len, device='cuda')

    def reset(self):
        self.index_rec = 0
        if self.read_to_ram:
            random.shuffle(self.data_list)
            idx_permute = torch.randperm(self.data_all.shape[0])
            self.data_all = self.data_all[idx_permute]
            self.pose_all = self.pose_all[idx_permute]
            self.beta_all = self.beta_all[idx_permute]
            self.transl_all = self.transl_all[idx_permute]
            self.gender_all = self.gender_all[idx_permute]
        else:
            random.shuffle(self.rec_list)

    def reset_with_jts(self):
        '''
        - this script is to train the marker regressor with rollout.
        - the ground truth joint location is used to provide the canonical coordinate.
        '''
        self.index_rec = 0
        if self.read_to_ram:
            random.shuffle(self.data_list)
            idx_permute = torch.randperm(self.data_all.shape[0])
            self.data_all = self.data_all[idx_permute]
            self.jts_all = self.jts_all[idx_permute]
            self.pose_all = self.pose_all[idx_permute]
            self.beta_all = self.beta_all[idx_permute]
            self.transl_all = self.transl_all[idx_permute]
            self.gender_all = self.gender_all[idx_permute]
        else:
            random.shuffle(self.rec_list)


    def has_next_rec(self):
        if self.read_to_ram:
            if self.index_rec < len(self.data_list):
                return True
            return False
        else:
            if self.index_rec < len(self.rec_list):
                return True
            return False


    def get_rec_list(self, shuffle_seed=None,
                    to_gpu=False):

        if self.amass_subset_name is not None:
            ## read the sequence in the subsets
            self.rec_list = []
            print('subset:', self.amass_subset_name)
            for subset in self.amass_subset_name:
                self.rec_list += glob.glob(os.path.join(self.amass_data_path,
                                                       subset,
                                                       '*.npz'  ))
        else:
            ## read all amass sequences
            self.rec_list = glob.glob(os.path.join(self.amass_data_path,
                                                    '*/*.npz'))

        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.rec_list)
        else:
            random.shuffle(self.rec_list) # shuffle recordings, not frames in a recording.
        if self.read_to_ram:
            print('[INFO] read all data to RAM...')
            self.pose_list = []
            self.transl_list = []
            self.beta_list = []
            self.gender_list = []
            for rec in self.rec_list:
                with np.load(rec) as data_:
                    framerate = data_['mocap_framerate']
                    if framerate != 120:
                        continue
                    sample_rate = self.sample_rate
                    pose = data_['poses'][::sample_rate,:66] # 156d = 66d+hand
                    transl = data_['trans'][::sample_rate]
                    beta = data_['betas']
                    gender = data_['gender'].astype(str)

                    if np.isnan(pose).any() or np.isinf(pose).any() or np.isnan(transl).any() or np.isinf(transl).any():
                        continue
                    body_cmu_41 = data_['marker_cmu_41'][::sample_rate]
                    body_ssm2_67 = data_['marker_ssm2_67'][::sample_rate]
                    joints = data_['joints'][::sample_rate].reshape([-1,22,3])
                    transf_rotmat = data_['transf_rotmat']
                    transf_transl = data_['transf_transl']
                
                transl = transl[:self.max_len]
                pose = pose[:self.max_len]
                body_cmu_41 = body_cmu_41[:self.max_len]
                body_ssm2_67 = body_ssm2_67[:self.max_len]
                joints = joints[:self.max_len]
                
                vec_to_target, wpath, marker2tarloc_n = self._get_target_feature(joints, body_ssm2_67, transf_rotmat, transf_transl)

                #['smpl_params', 'cmu_41', 'ssm2_67', 'ssm2_67_marker2tarloc', 'joint_location', 'bone_transform' ]
                if self.body_repr == 'smpl_params':
                    body_feature = np.concatenate([transl, pose],axis=-1)
                elif self.body_repr == 'joints':
                    body_feature = joints.reshape([-1,22*3])
                elif self.body_repr == 'cmu_41':
                    body_feature = body_cmu_41.reshape([-1,67*3])
                elif self.body_repr == 'ssm2_67':
                    body_feature = body_ssm2_67.reshape([-1,67*3])
                elif self.body_repr == 'ssm2_67_marker2tarloc':
                    body_feature = np.concatenate([body_ssm2_67.reshape([-1,67*3]),
                                                   marker2tarloc_n.reshape([-1,67*3])],
                                                axis=-1)
                elif self.body_repr == 'bone_transform':
                    joint_loc = joints
                    joint_rot_aa = pose.reshape([-1, 22, 3])
                    body_feature = np.concatenate([joint_loc, joint_rot_aa], axis=-1)
                else:
                    raise NameError('[ERROR] not valid body representation. Terminate')
                self.data_list.append(body_feature)
                self.jts_list.append(joints)
                self.pose_list.append(pose)
                self.beta_list.append(beta)
                self.transl_list.append(transl)
                self.gender_list.append(gender)

            self.data_all = np.stack(self.data_list,axis=0) #[b,t,d]
            self.jts_all = np.stack(self.jts_list, axis=0) #[b,t, 22, 3]
            self.pose_all = np.stack(self.data_list, axis=0)  # [b,t,d]
            self.beta_all = np.stack(self.beta_list, axis=0)  # [b,t,d]
            self.transl_all = np.stack(self.transl_list, axis=0)  # [b,t,d]
            self.gender_all = np.array(self.gender_list)  # [b,t,d]
            if to_gpu:
                self.data_all = torch.cuda.FloatTensor(self.data_all)
                self.jts_all = torch.cuda.FloatTensor(self.jts_all)
                self.pose_all = torch.cuda.FloatTensor(self.pose_all)
                self.transl_all = torch.cuda.FloatTensor(self.transl_all)
                self.beta_all = torch.cuda.FloatTensor(self.beta_all)


    def next_batch(self, batch_size=64, noise=None):
        if noise is None:
            batch_data_ = self.data_all[self.index_rec:self.index_rec+batch_size]
            self.index_rec+=batch_size
            batch_data = torch.cuda.FloatTensor(batch_data_).permute(1,0,2) #[t,b,d]
        else:
            batch_data = []
            bb = 0
            while self.has_next_rec():
                if bb == batch_size:
                    break
                gender = self.gender_all[self.index_rec]
                # print(gender)
                transl = self.transl_all[self.index_rec]
                pose = self.pose_all[self.index_rec]
                betas = self.beta_all[self.index_rec]

                rot_noise = torch.normal(mean=0, std=noise, size=pose[:1, :].shape, dtype=pose.dtype,
                                         device=pose.device).expand(pose.shape)
                pose = apply_rot_noise(pose, rot_noise)
                # print('applied noise of ', noise)
                t = pose.shape[0]
                bodymodel_batch = self.bm_batch_male if gender == 'male' else self.bm_batch_female
                body_param = {}
                body_param['transl'] = transl
                body_param['global_orient'] = pose[:, :3]
                body_param['betas'] = betas[:10].unsqueeze(0).repeat(t, 1).cuda()
                body_param['body_pose'] = pose[:, 3:66]
                smplxout = bodymodel_batch(return_verts=True, **body_param)
                ### extract joints and markers
                # joints = smplxout.joints[:, :22, :].detach().squeeze().cpu().numpy()
                body_ssm2_67 = smplxout.vertices[:, marker_ssm_67, :].detach()  # [t, 67, 3]
                body_feature = body_ssm2_67.reshape([-1, 67 * 3])
                batch_data.append(body_feature)
                self.index_rec += 1
                bb += 1
                if self.index_rec == len(self.data_list):
                    break
            batch_data = torch.stack(batch_data).permute(1,0,2)  # [t, 67*3] -> [b, t, 67*3] -> [t, b, 67*3]

        return batch_data


    def next_batch_with_jts(self, batch_size=64, noise=None):
        batch_data_ = self.data_all[self.index_rec:self.index_rec+batch_size].permute(1,0,2)
        batch_jts_ = self.jts_all[self.index_rec:self.index_rec+batch_size].permute(1,0,2,3)
        self.index_rec+=batch_size
        return batch_data_, batch_jts_



    def _get_target_feature(self, joints, body_ssm2_67, rotmat=np.eye(3), transl=np.zeros((1,3))):
        '''normalized walking path'''
        wpath = joints[-1:]-joints
        wpath = wpath[:,0,:2] #(t,2)
        wpath_n = wpath/(1e-8+np.linalg.norm(wpath, axis=-1, keepdims=True))
        '''unnormalized target_marker - starting_marker '''
        vec_to_target = body_ssm2_67[-1:]-body_ssm2_67
        '''normalized target_location - starting_marker'''
        target_loc = joints[-1:, 0:1] # the pelvis (1,1,3)
        target_loc[:,:,-1] = target_loc[:,:,-1] - transl[None,...][:,:,-1]
        vec_to_target_loc = target_loc - body_ssm2_67
        vec_to_target_locn = vec_to_target_loc/np.linalg.norm(vec_to_target_loc, axis=-1, keepdims=True)
        return vec_to_target, wpath_n, vec_to_target_locn



    def next_sequence(self):
        '''
        - this function is only for produce files for visualization or testing in some cases
        - compared to next_batch with batch_size=1, this function also outputs metainfo, like gender, body shape, etc.
        '''
        rec = self.rec_list[self.index_rec]
        with np.load(rec) as data:
            framerate = data['mocap_framerate']
            sample_rate = self.sample_rate
            pose = data['poses'][::sample_rate,:66] # 156d = 66d+hand
            transl = data['trans'][::sample_rate]
            gender = data['gender']
            if np.isnan(pose).any() or np.isinf(pose).any() or np.isnan(transl).any() or np.isinf(transl).any():
                return None
            betas = data['betas'][:10]
            body_cmu_41 = data['marker_cmu_41'][::sample_rate]
            body_ssm2_67 = data['marker_ssm2_67'][::sample_rate]
            framerate = data['mocap_framerate']
            joints = data['joints'][::sample_rate].reshape([-1,22,3])
            transf_rotmat = data['transf_rotmat']
            transf_transl = data['transf_transl']

            ## normalized walking path and unnormalized marker to target
            vec_to_target, wpath, marker2tarloc_n = self._get_target_feature(joints, body_ssm2_67)

        if self.body_repr == 'smpl_params':
            body_feature = np.concatenate([transl, pose],axis=-1)
        elif self.body_repr == 'joints':
            body_feature = joints.reshape([-1,22*3])
        elif self.body_repr == 'cmu_41':
            body_feature = body_cmu_41.reshape([-1,41*3])
        elif self.body_repr == 'ssm2_67':
            body_feature = body_ssm2_67.reshape([-1,67*3])
        elif self.body_repr == 'ssm2_67_marker2tarloc':
                    body_feature = np.concatenate([body_ssm2_67.reshape([-1,67*3]),
                                                   marker2tarloc_n.reshape([-1,67*3])],
                                                axis=-1)
        elif self.body_repr == 'bone_transform':
            joint_loc = joints
            joint_rot_aa = pose.reshape([-1, 22, 3])
            body_feature = np.concatenate([joint_loc, joint_rot_aa], axis=-1)
        else:
            raise NameError('[ERROR] not valid body representation. Terminate')

        ## pack output data
        output = {}
        output['betas'] = betas
        output['gender'] = gender
        output['transl'] = transl
        output['glorot'] = pose[:,:3]
        output['poses'] = pose[:,3:]
        output['body_feature'] = body_feature
        output['transf_rotmat'] = transf_rotmat
        output['transf_transl'] = transf_transl
        output['pelvis_loc'] = joints[:,0,:]
        self.index_rec += 1
        return output




    def next_batch_genderselection(self, batch_size=64, gender='male',
                                    batch_first=True, noise=None):
        '''
        - this function is to select a batch of data with the same gender
        - it not only outputs body features, but also body parameters, and genders
        - note here the "batch_size" indicates the number of sequences
        '''
        batch_betas = []
        batch_transl = []
        batch_glorot = []
        batch_thetas = []
        batch_body_feature = []
        batch_jts = []
        stack_dim = 0 if batch_first else 1

        bb = 0
        while self.has_next_rec():
            rec = self.rec_list[self.index_rec]
            if bb == batch_size:
                break
            else:
                if str(np.load(rec)['gender']) != gender:
                    self.index_rec += 1
                    continue
            with np.load(rec) as data:
                framerate = data['mocap_framerate']
                # sample_rate = int(framerate//self.fps)
                sample_rate = self.sample_rate
                transl = data['trans'][::sample_rate]
                pose = data['poses'][::sample_rate,:66] # 156d = 66d+hand
                betas  = np.tile(data['betas'][:10], (transl.shape[0],1) )
                body_cmu_41 =  data['marker_cmu_41'][::sample_rate]
                body_ssm2_67 = data['marker_ssm2_67'][::sample_rate]
                joints = data['joints'][::sample_rate].reshape([-1,22,3])

            ## normalized walking path and unnormalized marker to target
            vec_to_target, wpath, marker2tarloc_n = self._get_target_feature(joints, body_ssm2_67)

            if self.body_repr == 'smpl_params':
                body_feature = np.concatenate([transl, pose],axis=-1)
            elif self.body_repr == 'joints':
                body_feature = joints.reshape([-1,22*3])
            elif self.body_repr == 'cmu_41':
                body_feature = body_cmu_41.reshape([-1,41*3])
            elif self.body_repr == 'ssm2_67':
                body_feature = body_ssm2_67.reshape([-1,67*3])
            elif self.body_repr == 'ssm2_67_marker2tarloc':
                        body_feature = np.concatenate([body_ssm2_67.reshape([-1,67*3]),
                                                    marker2tarloc_n.reshape([-1,67*3])],
                                                    axis=-1)
            elif self.body_repr == 'bone_transform':
                joint_loc = joints
                joint_rot_aa = pose.reshape([-1, 22, 3])
                body_feature = np.concatenate([joint_loc, joint_rot_aa], axis=-1)
            else:
                raise NameError('[ERROR] not valid body representation. Terminate')

            batch_betas.append(betas)
            batch_transl.append(transl)
            batch_glorot.append(pose[:,:3])
            batch_thetas.append(pose[:,3:])
            batch_jts.append(joints.reshape([-1,22*3]))
            batch_body_feature.append(body_feature)
            self.index_rec += 1
            bb += 1

            if self.index_rec == len(self.data_list):
                break
        if len(batch_betas) < batch_size:
            return None
        else:
            batch_betas = torch.cuda.FloatTensor(np.stack(batch_betas,axis=stack_dim)) #[b, t, d]
            batch_transl = torch.cuda.FloatTensor(np.stack(batch_transl,axis=stack_dim)) #[b, t, d]
            batch_glorot = torch.cuda.FloatTensor(np.stack(batch_glorot,axis=stack_dim)) #[b, t, d]
            batch_thetas = torch.cuda.FloatTensor(np.stack(batch_thetas,axis=stack_dim)) #[b, t, d]
            batch_jts = torch.cuda.FloatTensor(np.stack(batch_jts,axis=stack_dim)) #[b, t, d]
            batch_body_feature = torch.cuda.FloatTensor(np.stack(batch_body_feature,axis=stack_dim))
            return [batch_betas, batch_body_feature,
                    batch_transl, batch_glorot,batch_thetas, batch_jts]

    def get_all_data(self):
        return torch.FloatTensor(self.data_all).permute(1,0,2) #[t,b,d]





class BatchGeneratorFollowPathInCubes(object):
    def __init__(self,
                dataset_path,
                body_model_path='/home/yzhang/body_models/VPoser',
                scene_ori='ZupYf', # Z-up Y-forward, this is the default setting in our work. Otherwise we need to transform it before and after
                body_repr='ssm2_67', #['smpl_params', 'cmu_41', 'ssm2_67', 'joints']
                data_list=None,
                ):
        self.rec_list = list()
        self.index_rec = 0
        self.dataset_path = dataset_path
        self.data_list = [] if data_list is None else data_list
        self.body_repr = body_repr
        self.scene_ori = scene_ori
        

        self.bm_male = smplx.create(body_model_path, model_type='smplx',
                                    gender='male', ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=1
                                    ).eval()
        self.bm_female = smplx.create(body_model_path, model_type='smplx',
                                    gender='female', ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=1
                                    ).eval()
        self.vposer, _ = load_vposer(body_model_path+'/vposer_v1_0', vp_model='snapshot')
        self.vposer.eval()

    def params2torch(self, params, dtype = torch.float32):
        return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}

    def params2numpy(self, params):
        return {k: v.detach().cpu().numpy() for k, v in params.items() if type(v)==torch.Tensor}

    def reset(self):
        self.index_rec = 0
        random.shuffle(self.rec_list)

    def has_next_rec(self):
        if self.index_rec < len(self.rec_list):
            return True
        return False

    def get_rec_list(self, shuffle_seed=None):
        self.rec_list = sorted(glob.glob(os.path.join(self.dataset_path, 'traj_*.pkl')))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.rec_list)



    def xbo_to_bodydict(self, body_params):
        '''
        from what siwei gives me, to what we need to get the smplx mesh
        '''
        body_params_dict = {}
        body_params_dict['transl'] = body_params[:, :3]
        body_params_dict['global_orient'] = body_params[:, 3:6]
        body_params_dict['betas'] = body_params[:, 6:16]
        body_params_dict['body_pose_vp'] = body_params[:, 16:48]
        body_params_dict['left_hand_pose'] = body_params[:, 48:60]
        body_params_dict['right_hand_pose'] = body_params[:, 60:]
        body_params_dict['body_pose'] = self.vposer.decode(body_params[:, 16:48],
                                        output_type='aa').view(1, -1)  # tensor, [1, 63]
        return body_params_dict


    def snap_to_ground(self, xbo_dict, bm, height=0):
        ## get the body mesh and (optionally) vis in open3d
        verts = bm(**xbo_dict).vertices[0].detach().cpu().numpy()#[P, 3]
        ## snap the body mesh to the ground, and move it to a new place
        delta_z = np.array([[np.min(verts[:,-1])]])-height
        delta_xy = np.zeros((1,2))
        delta = np.concatenate([delta_xy, delta_z],axis=-1)
        xbo_dict['transl'] -= torch.FloatTensor(delta)
        return xbo_dict


    def snap_to_ground_cuda(self, xbo_dict, bm, height=0):
        ## get the body mesh and (optionally) vis in open3d
        verts = bm(**xbo_dict).vertices[0].detach()#[P, 3]
        ## snap the body mesh to the ground, and move it to a new place
        delta_z = torch.amin(verts[:,-1:],keepdim=True)-height
        delta_xy = torch.cuda.FloatTensor(1,2).zero_()
        delta = torch.cat([delta_xy, delta_z],dim=-1)
        xbo_dict['transl'] -= delta
        return xbo_dict



    def get_body_keypoints(self, xbo_dict, bm):
        bmout = bm(**xbo_dict)
        ## snap the body mesh to the ground, and move it to a new place
        markers = bmout.vertices[:,self.marker_ssm_67,:].detach().cpu().numpy()
        jts = bmout.joints[:,:22,:].detach().cpu().numpy()
        return markers, jts

    def get_bodyori_from_wpath(self, a, b):
        '''
        input: a,b #(3,) denoting starting and ending location
        '''
        z_axis = (b-a)/np.linalg.norm(b-a)
        y_axis = np.array([0,0,1])
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis/np.linalg.norm(x_axis)
        glorot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
        glorot_aa = R.from_matrix(glorot_mat).as_rotvec()
        return glorot_aa


    def next_body(self, character_file=None, visualize=False):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        """read walking path"""
        rec_path = self.rec_list[self.index_rec]
        wpath0 = np.load(rec_path, allow_pickle=True)
        if self.scene_ori == 'YupZf':
            #rotate around x by 90, and rotate it back at the very end.
            rotmat = np.array([[1,0,0],[0,0,-1], [0,1,0]]) # rotate about x by 90deg
            wpath = [np.einsum('ij,j->i', rotmat, x) for x in wpath0]
        elif self.scene_ori == 'ZupYf':
            wpath = wpath0
        """generate a body"""
        xbo_dict = {}
        if character_file is not None:
            character_data = np.load(character_file, allow_pickle=True)
            gender = character_data['gender']
            xbo_dict['betas'] = character_data['betas']
        else:
            gender = random.choice(['female', 'male'])
            xbo_dict['betas'] = np.random.randn(1,10)

        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[1])[None,...]
        xbo_dict['body_pose'] = self.vposer.decode(torch.randn(1,32), # prone to self-interpenetration
                                           output_type='aa').view(1, -1).detach().numpy()
        # TODO: accurately translate to wpath[0]
        xbo_dict['transl'] = wpath[0][None, ...]  # [1,3]

        """snap to the ground"""
        floor_height = wpath[0][-1]
        xbo_dict = self.params2torch(xbo_dict)
        bm = self.bm_male if gender=='male' else self.bm_female
        xbo_dict = self.snap_to_ground(xbo_dict, bm, height=wpath[0][-1])

        # # lift waypoints to pelvis height
        # pelvis_height = bm(**xbo_dict).joints[0, 0, 2].item()
        # wpath[:, 2] = pelvis_height

        """specify output"""
        out_dict = self.params2numpy(xbo_dict)
        out_dict['betas'] = out_dict['betas']
        out_dict['gender']=gender
        out_dict['wpath']=wpath
        out_dict['wpath_filename']=os.path.basename(rec_path)
        self.index_rec += 1

        return out_dict


    def next_body_cuda(self, character_file=None):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        """read walking path"""
        rec_path = self.rec_list[self.index_rec]
        wpath = np.load(rec_path, allow_pickle=True)

        """generate a body"""
        xbo_dict = {}
        if character_file is not None:
            character_data = np.load(character_file, allow_pickle=True)
            gender = character_data['gender']
            xbo_dict['betas'] = torch.cuda.FloatTensor(character_data['betas'])
        else:
            gender = random.choice(['female'])
            xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()

        xbo_dict['transl'] = torch.cuda.FloatTensor(wpath[0][None,...]) #[1,3]
        xbo_dict['global_orient'] = torch.cuda.FloatTensor(self.get_bodyori_from_wpath(wpath[0], wpath[1])[None,...])
        xbo_dict['body_pose'] = self.vposer.decode(torch.randn(1,32), # prone to self-interpenetration
                                           output_type='aa').view(1, -1).cuda()
        """snap to the ground"""
        # xbo_dict = self.params2torch(xbo_dict)
        bm = self.bm_male if gender=='male' else self.bm_female
        bm = bm.cuda()
        xbo_dict = self.snap_to_ground_cuda(xbo_dict, bm, height=wpath[0][-1])

        """specify output"""
        out_dict = xbo_dict
        out_dict['betas']
        out_dict['gender']=gender
        out_dict['wpath']= torch.cuda.FloatTensor(np.stack(wpath))
        out_dict['wpath_filename']=os.path.basename(rec_path)
        self.index_rec += 1

        return out_dict





class BatchGeneratorReachingTarget(object):
    def __init__(self,
                dataset_path,
                body_model_path='/home/yzhang/body_models/VPoser',
                body_repr='ssm2_67' #['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                ):
        self.rec_list = list()
        self.index_rec = 0
        self.dataset_path = dataset_path
        self.data_list = []
        self.body_repr = body_repr

        self.bm_male = smplx.create(body_model_path, model_type='smplx',
                                    gender='male', ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=1
                                    ).eval().cuda()
        self.bm_female = smplx.create(body_model_path, model_type='smplx',
                                    gender='female', ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=1
                                    ).eval().cuda()
        self.vposer, _ = load_vposer(body_model_path+'/vposer_v1_0', vp_model='snapshot')
        self.vposer.eval()
        self.vposer.cuda()

    def params2torch(self, params, dtype = torch.float32):
        return {k: torch.from_numpy(v).type(dtype) if type(v)==np.ndarray else v for k, v in params.items()}

    def params2numpy(self, params):
        return {k: v.detach().cpu().numpy() if type(v)==torch.Tensor else v for k, v in params.items() }

    def reset(self):
        self.index_rec = 0


    def has_next_rec(self):
        pass

    def get_rec_list(self, shuffle_seed=None):
        pass

    def snap_to_ground(self, xbo_dict, bm, floor_height=0):
        ## get the body mesh and (optionally) vis in open3d
        verts = bm(**xbo_dict).vertices[0].detach()#[P, 3]
        ## snap the body mesh to the ground, and move it to a new place
        delta_z = torch.cuda.FloatTensor([[torch.amin(verts[:,-1]) - floor_height]])
        delta_xy = torch.cuda.FloatTensor(1,2).zero_()
        delta = torch.cat([delta_xy, delta_z],dim=-1)
        xbo_dict['transl'] -= delta
        return xbo_dict

    def snap_to_ground_recenter_origin(self, xbo_dict, bm):
        ## get the body mesh and (optionally) vis in open3d
        verts = bm(**xbo_dict).vertices[0].detach()#[P, 3]
        joints = bm(**xbo_dict).joints[0].detach()#[P, 3]
        ## snap the body mesh to the ground, and move it to a new place
        delta_z = torch.cuda.FloatTensor([[torch.amin(verts[:,-1])]])
        delta_xy = joints[[0], :2]
        delta = torch.cat([delta_xy, delta_z],dim=-1)
        return xbo_dict['transl'] - delta, joints[[0]] - delta


    def get_bodyori_from_wpath(self, a, b):
        '''
        input: a,b #(3,) denoting starting and orientating location
        '''
        z_axis = (b-a) / torch.norm(b-a).clip(min=1e-12)
        y_axis = torch.cuda.FloatTensor([0,0,1])
        x_axis = torch.cross(y_axis, z_axis)
        x_axis = x_axis / torch.norm(x_axis).clip(min=1e-12)
        glorot_mat = torch.stack([x_axis, y_axis, z_axis], dim=1)
        glorot_aa = RotConverter.rotmat2aa(glorot_mat)[0]
        return glorot_aa


    def next_body(self, sigma=10, character_file=None, use_zero_pose=False,
                  visualize=False, to_numpy=False,
                  hard_code=False
                  ):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        """randomly specify a 3D path"""
        wpath = np.zeros((3,3))
        # wpath = torch.cuda.FloatTensor(3, 3).normal_() #starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(wpath) #starting point, ending point, another point to initialize the body orientation
        wpath[0] = 0 #starting point
        r = torch.cuda.FloatTensor(1).uniform_() * 0.5 + 0.25
        theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi
        wpath[1, :2] = torch.cat([r * torch.cos(theta), r * torch.sin(theta)])
        # wpath[1, :2] = sigma*(2*torch.cuda.FloatTensor(2).uniform_()-1) #ending point xy
        wpath[1, 2] = torch.rand(1) * 0.2 + 0.5
        wpath[2, :2] = sigma * torch.randn(2) #point to initialize the body orientation, not returned

        # hard code initial and target
        if hard_code:
            wpath[1, 0] = 0.5
            wpath[1, 1] = 0
            wpath[2, :2] = wpath[1, :2]

        # wpath[:,-1] = 0 # proj to ground
        # wpath = np.array([[  0.        ,   0.        ,   0.        ],
        #                 [-27.91866343/2,   3.70924324/2,   0.        ],
        #                 [-13.72670315/3, -33.74804827/3,   0.        ]])

        """generate a body"""
        xbo_dict = {}
        if character_file is not None:
            # character_data = np.load(character_file, allow_pickle=True)
            character_data = character_file[torch.randint(len(character_file), (1,))]
            gender = 'male'
            xbo_dict['betas'] = torch.cuda.FloatTensor(character_data['betas'][:10])
            xbo_dict['body_pose'] = torch.cuda.FloatTensor(character_data['body_pose'])
            xbo_dict['global_orient'] = torch.cuda.FloatTensor(character_data['global_orient'])
        else:
            # gender = random.choice(['male', 'female'])
            gender = random.choice(['male'])
            xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()
            xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                           output_type='aa').view(1, -1)
            xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None,...]
            # gender = random.choice(['male'])
            # xbo_dict['betas'] = np.zeros([1,10])
        xbo_dict['transl'] = wpath[:1] #[1,3]



        """snap to the ground"""
        # xbo_dict = self.params2torch(xbo_dict)
        bm = self.bm_male if gender=='male' else self.bm_female
        # xbo_dict = self.snap_to_ground(xbo_dict, bm)
        # wpath[0] = bm(**xbo_dict).joints[0, 0].detach()  # recenter starting point at pelvis
        xbo_dict['transl'], wpath[0] = self.snap_to_ground_recenter_origin(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis

        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                              transform=np.array([[1.0, 0.0, 0.0, 0],
                                                                  [0.0, 1.0, 0.0, 0],
                                                                  [0.0, 0.0, 1.0, -0.005],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            vis_mesh = [floor_mesh, init_body_mesh,
                        trimesh.creation.axis()
                        ]
            for point_idx, pelvis in enumerate(wpath[:2, :].reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)

            print(wpath)
            import pyrender
            scene = pyrender.Scene()
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
            for mesh in vis_mesh:
                viewer.render_lock.acquire()
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
                viewer.render_lock.release()

        """specify output"""
        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        if to_numpy:
            xbo_dict = self.params2numpy(xbo_dict)
        self.index_rec += 1

        return xbo_dict

class BatchGeneratorReachingMarker(BatchGeneratorReachingTarget):
    def snap_to_ground(self, xbo_dict, bm):
        ## get the body mesh and (optionally) vis in open3d
        bm_output = bm(**xbo_dict)
        verts = bm_output.vertices[0].detach()#[P, 3]
        joints = bm_output.joints[0].detach()
        ## snap the body mesh to the ground, and move it to a new place
        delta_z = torch.cuda.FloatTensor([[torch.amin(verts[:,-1])]])
        delta_xy = torch.cuda.FloatTensor(1,2).zero_()
        delta = torch.cat([delta_xy, delta_z],dim=-1)
        return xbo_dict['transl'] - delta, joints[0] - delta

    def next_body(self, sigma=10, character_file=None, use_zero_pose=False):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        """randomly specify a 3D path"""
        wpath = np.zeros((3,3))
        # wpath = torch.cuda.FloatTensor(3, 3).normal_() #starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(wpath) #starting point, ending point, another point to initialize the body orientation
        wpath[0] = 0 #starting point
        wpath[1, :2] = sigma*(2*torch.cuda.FloatTensor(2).uniform_()-1) #ending point xy
        # wpath[1, 2] = torch.rand(1) * 0.2 + 0.5
        wpath[2, :2] = sigma * torch.randn(2) #point to initialize the body orientation, not returned
        # wpath[:,-1] = 0 # proj to ground
        # wpath = np.array([[  0.        ,   0.        ,   0.        ],
        #                 [-27.91866343/2,   3.70924324/2,   0.        ],
        #                 [-13.72670315/3, -33.74804827/3,   0.        ]])

        """generate init body"""
        xbo_dict = {}
        gender = 'male'
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).normal_()
        xbo_dict['body_pose'] = self.vposer.decode(
            torch.cuda.FloatTensor(1, 32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1, 32).normal_(),
            # prone to self-interpenetration
            output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None, ...]
        # gender = random.choice(['male'])
        # xbo_dict['betas'] = np.zeros([1,10])
        xbo_dict['transl'] = wpath[:1]  # [1,3]
        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict['transl'], wpath[0] = self.snap_to_ground_recenter_origin(xbo_dict,
                                                                           bm)  # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        start_markers = bm(**xbo_dict).vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        # generate target markers
        character_data = character_file[torch.randint(len(character_file), (1,))]
        gender = 'male'
        bm = self.bm_male if gender == 'male' else self.bm_female
        smplx_params = {}
        smplx_params['betas'] = torch.cuda.FloatTensor(character_data['betas'][:10])
        smplx_params['body_pose'] = torch.cuda.FloatTensor(character_data['body_pose'])
        smplx_params['global_orient'] = torch.cuda.FloatTensor(character_data['global_orient'])
        smplx_params['transl'] = wpath[[1]]
        smplx_params['transl'], wpath[1] = self.snap_to_ground(smplx_params, bm)
        target_markers = bm(**smplx_params).vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """specify output"""
        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        xbo_dict['markers'] = torch.cat([start_markers, target_markers], dim=0)
        self.index_rec += 1

        return xbo_dict

"""
R: Bx3x3
T: Bx1x3
local to world transformation
local map of walkability, 1 walkable, 0 not walkable
"""
def get_map(navmesh, R, T, res=32, extent=1.6, return_type='torch'):
    batch_size = R.shape[0]
    x = torch.linspace(-extent, extent, res)
    y = torch.linspace(-extent, extent, res)
    xv, yv = torch.meshgrid(x, y)
    points = torch.stack([xv, yv, torch.zeros_like(xv)], axis=2).to(device='cuda')  # [res, res, 3]
    points = points.reshape(1, -1, 3).repeat(batch_size, 1, 1)
    points_scene = torch.einsum('bij,bpj->bpi', R, points) + T  # [b, r*r, 3]
    floor_height = navmesh.vertices[0, 2]
    points_scene[:, :, 2] = floor_height

    # https://stackoverflow.com/a/2049593/14532053
    points_2d = points_scene[:, :, :2].reshape(batch_size * res * res, 1, 2)  # [P=b*r*r, 1, 2]
    triangles = torch.cuda.FloatTensor(np.stack([navmesh.vertices[navmesh.faces[:, 0], :2],
                                        navmesh.vertices[navmesh.faces[:, 1], :2],
                                        navmesh.vertices[navmesh.faces[:, 2], :2]], axis=-1)).permute(0, 2, 1)[None, ...]  # [1, F, 3, 2]
    def sign(p1, p2, p3):
        return (p1[:, :, 0] - p3[:, :, 0]) * (p2[:, :, 1] - p3[:, :, 1]) - (p2[:, :, 0] - p3[:, :, 0]) * (p1[:, :, 1] - p3[:, :, 1])

    d1 = sign(points_2d, triangles[:, :, 0, :], triangles[:, :, 1, :])
    d2 = sign(points_2d, triangles[:, :, 1, :], triangles[:, :, 2, :])
    d3 = sign(points_2d, triangles[:, :, 2, :], triangles[:, :, 0, :])

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    inside_triangle = ~(has_neg & has_pos) #[P, F]
    inside_mesh = inside_triangle.any(-1)

    map = inside_mesh.reshape((batch_size, res * res))
    if return_type == 'numpy':
        map = map.detach().cpu().numpy()
        points_scene = points_scene.detach().cpu().numpy()
        points = points.detach().cpu().numpy()
    return points, points_scene, map

def get_map_trimesh(navmesh, navmesh_query, R, T, res=32, extent=1.6, return_type='torch'):
    batch_size = R.shape[0]
    x = torch.linspace(-extent, extent, res)
    y = torch.linspace(-extent, extent, res)
    xv, yv = torch.meshgrid(x, y)
    points = torch.stack([xv, yv, torch.zeros_like(xv)], axis=2).to(device='cuda')  # [res, res, 3]
    points = points.reshape(1, -1, 3).repeat(batch_size, 1, 1)
    points_scene = torch.einsum('bij,bpj->bpi', R, points) + T  # [b, r*r, 3]
    floor_height = navmesh.vertices[0, 2]
    points_scene[:, :, 2] = floor_height

    closest, distance, triangle_id = navmesh_query.on_surface(points_scene.reshape(-1, 3).detach().cpu().numpy())
    print('distance:', distance.max())
    map = (distance < 0.001).reshape((batch_size, res * res))
    if return_type == 'torch':
        map = torch.cuda.FloatTensor(map)
    if return_type == 'numpy':
        points_scene = points_scene.detach().cpu().numpy()
        points = points.detach().cpu().numpy()
    return points, points_scene, map

"""Get the transformation to scale shapenet object to real size, and put its bottom on floor"""
def transform_real_size_on_floor(obj_path, json_path):
    with open(json_path, 'r') as f:
        meta_info = json.load(f)
    obj_mesh = trimesh.load(obj_path, force='mesh')
    scale = (np.array(meta_info['max']) - np.array(meta_info['min'])) / (obj_mesh.bounds[1] - obj_mesh.bounds[0])
    transform = np.diag([scale[0], scale[1], scale[2], 1])
    transform[1, 3] = -obj_mesh.bounds[0, 1] * scale[1]
    return transform

class BatchGeneratorCollision(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67',  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 scene_list=None,
                 shapenet_dir=None,
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        self.scene_list= scene_list
        self.shapenet_dir = shapenet_dir
        self.scene_idx = 0
        # with open(os.path.join(dataset_path, 'orient.json')) as f:
        #     self.orient = np.array(json.load(f)).reshape(1, 3)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=True, scene_idx=0, res=32, extent=1.6):
        if scene_idx is None:
            scene_idx = torch.randint(len(self.scene_list), size=(1,)).item()
        scene_name = self.scene_list[scene_idx]
        mesh_path = Path(self.shapenet_dir).joinpath(*(scene_name.split('_'))).joinpath('models', 'model_normalized.obj')
        json_path = Path(self.shapenet_dir).joinpath(*(scene_name.split('_'))).joinpath('models',
                                                                                        'model_normalized.json')
        navmesh_path = Path(self.dataset_path) /  (scene_name + '.obj')
        obj_mesh = trimesh.load(mesh_path, force='mesh')
        obj_transform = np.matmul(shapenet_to_zup, transform_real_size_on_floor(mesh_path, json_path))
        obj_mesh.apply_transform(obj_transform)
        navmesh = trimesh.load(navmesh_path)
        navmesh.apply_transform(unity_to_zup)
        # print('navmesh bounds:', navmesh.bounds)
        navmesh.vertices[:, 2] = 0
        navmesh.visual.vertex_colors = np.array([0, 0, 200, 200])
        navmesh_query = trimesh.proximity.ProximityQuery(navmesh)
        navmesh_torch = pytorch3d.structures.Meshes(
            verts=[torch.cuda.FloatTensor(navmesh.vertices)],
            faces=[torch.cuda.LongTensor(navmesh.faces)]
        )
        crop_box = trimesh.creation.box(extents=[sigma + obj_mesh.extents[0], sigma + obj_mesh.extents[1],  2])
        navmesh_crop = navmesh.slice_plane(crop_box.facets_origin, -crop_box.facets_normal)
        navmesh_crop.visual.vertex_colors = np.array([0, 200, 0, 200])
        # import pyrender
        # scene = pyrender.Scene()
        # scene.add(pyrender.Mesh.from_trimesh(obj_mesh, smooth=False))
        # # scene.add(pyrender.Mesh.from_trimesh(navmesh, smooth=False))
        # scene.add(pyrender.Mesh.from_trimesh(navmesh_crop, smooth=False))
        # pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        """randomly specify a 3D path"""

        wpath = np.zeros((3,3))
        # wpath = torch.cuda.FloatTensor(3, 3).normal_() #starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(wpath) #starting point, ending point, another point to initialize the body orientation
        start_target = np.zeros((2, 3))  # pairs of start and target positions
        for try_idx in range(32):
            start_target = trimesh.sample.sample_surface_even(navmesh_crop, 2)[0]
            if len(start_target) < 2:  # sample_surface_even can return less points
                continue
            if np.linalg.norm(start_target[0] - start_target[1]) > 0.3:
                break
        wpath[0] = torch.cuda.FloatTensor(start_target[0]) #starting point
        # wpath[1] = torch.cuda.FloatTensor(start_target[1])  # ending point xy
        wpath[1] = -wpath[0]
        # wpath[2, :2] = wpath[0, :2] + torch.randn(2).to(device=wpath.device) #point to initialize the body orientation, not returned
        theta = torch.pi * (2 * torch.cuda.FloatTensor(1).uniform_() - 1) * 0.5
        random_rotz = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]).reshape(1, 3),
                                                                  convention="XYZ")
        wpath[2] = torch.einsum('ij, j->i', random_rotz[0], wpath[1] - wpath[0]) + wpath[0]  # face the target with [-90, 90] disturbance
        # hard code
        # wpath[0] = torch.cuda.FloatTensor([-1, 0, 0])
        # wpath[1] = torch.cuda.FloatTensor([0.5, 0, 0])
        # wpath[2] = wpath[1]

        """generate a body"""
        xbo_dict = {}
        # gender = random.choice(['male', 'female'])
        gender = random.choice(['male'])
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None,...]

        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :]  # [1,3]
        xbo_dict = self.snap_to_ground(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        wpath[0] = bm(**xbo_dict).joints[0, 0, :]
        wpath[1, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        xbo_dict['obj_id'] = self.scene_list[scene_idx]
        xbo_dict['obj_transform'] = obj_transform
        xbo_dict['navmesh'] = navmesh
        xbo_dict['navmesh_torch'] = navmesh_torch
        xbo_dict['navmesh_path'] = navmesh_path
        self.index_rec += 1

        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                              transform=np.array([[1.0, 0.0, 0.0, 0],
                                                                  [0.0, 1.0, 0.0, 0],
                                                                  [0.0, 0.0, 1.0, -0.005],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            vis_mesh = [
                # floor_mesh,
                        init_body_mesh,
                        obj_mesh, navmesh,
                        trimesh.creation.axis()
                        ]

            for point_idx, pelvis in enumerate(wpath[:2, :].reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)

            # visualize map
            joints = bm(**xbo_dict).joints  # [b,p,3]
            x_axis = joints[:, 2, :] - joints[:, 1, :]
            x_axis[:, -1] = 0
            x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
            z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
            y_axis = torch.cross(z_axis, x_axis)
            gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
            gamma_transl = joints[0, 0, :].reshape(1, 1, 3)
            _, points, map = get_map(navmesh, gamma_orient, gamma_transl,
                                    res=res, extent=extent,
                                  return_type='numpy')
            # _, points, map = get_map_trimesh(navmesh, navmesh_query, gamma_orient, gamma_transl,
            #                          res=32, extent=1.6,
            #                          return_type='numpy')
            points = points[0]  # [p, 3]
            map = map[0]  #[p]
            cells = []
            for point_idx in range(points.shape[0]):
                color = np.array([0, 0, 200, 100]) if map[point_idx] else np.array([200, 0, 0, 100])
                transform = np.eye(4)
                transform[:3, 3] = points[point_idx]
                cell = trimesh.creation.box(extents=(0.05, 0.05, 1), vertex_colors=color, transform=transform)
                # cell = trimesh.creation.cylinder(radius=0.02,
                #                           segment=np.stack([points[point_idx], points[point_idx] + np.array([0, 0, 0.5])], axis=0),
                #                           vertex_colors=color)
                cells.append(cell)
            vis_mesh.append(trimesh.util.concatenate(cells))

            print(xbo_dict['wpath'])
            import pyrender
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        return xbo_dict

class BatchGeneratorSitting(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path, shapenet_dir, sdf_dir,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67'  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        with open(dataset_path, 'rb') as f:
            self.interaction_data = pickle.load(f)
        self.shapenet_dir = shapenet_dir
        self.sdf_dir = sdf_dir
        if not os.path.exists(sdf_dir):
            os.mkdir(sdf_dir)
        import logging
        logger = logging.getLogger("trimesh")
        logger.setLevel(logging.ERROR)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=False,
                  interaction_id=None, hard_code=None, reverse=False):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''

        """read interaction"""
        # interaction_id = 373
        if interaction_id is None:
            # interaction_id = torch.randint(len(self.interaction_data), size=(1,)).item()
            interaction_id = torch.randint(32, size=(1,)).item()
        interaction = self.interaction_data[interaction_id]
        gender = 'male'
        bm = self.bm_male if gender == 'male' else self.bm_female
        smplx_params = deepcopy(interaction['smplx'])  # will change smplx_param inplace
        smplx_params = {k: v.cpu().cuda() if type(v)==torch.Tensor else v for k, v in smplx_params.items() }  # change cuda device to current device
        objects = deepcopy(interaction['objects'])
        assert objects['obj_num'] == 1
        shapenet_id = objects['shapenet_id'][0]
        transform = deepcopy(objects['transform'][0])
        # print("load interaction:", shapenet_id + '-id' + str(interaction_id))
        # load or calc sdf grid
        sdf_path = os.path.join(self.sdf_dir, shapenet_id + '-id' + str(interaction_id) + '.pkl')
        if not os.path.exists(sdf_path):
            object_mesh = trimesh.load(
                os.path.join(*([self.shapenet_dir] + shapenet_id.split('-') + ['models', 'model_normalized.obj'])),
                force='mesh'
            )
            object_mesh.apply_transform(transform)
            object_sdf = {
                'grid': mesh_to_voxels(object_mesh, 128, pad=False, surface_point_method='scan', sign_method='depth'),
                'dim': 128,
                'extent': np.max(object_mesh.bounding_box.extents),
                'centroid': object_mesh.bounding_box.centroid,
            }
            with open(sdf_path, 'wb') as f:
                pickle.dump(object_sdf, f)
        else:
            with open(sdf_path, 'rb') as f:
                object_sdf = pickle.load(f)
            # print(type(object_sdf))
            # print(object_sdf['grid'])
        object_mesh = trimesh.load(
            os.path.join(*([self.shapenet_dir] + shapenet_id.split('-') + ['models', 'model_normalized.obj'])),
            force='mesh'
        )

        """randomly specify a 3D path"""
        wpath = np.zeros((3, 3))
        # starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(
            wpath)  # starting point, ending point, another point to initialize the body orientation
        wpath[0] = 0  # starting point
        r = torch.cuda.FloatTensor(1).uniform_() * 0.4 + 0.6
        theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
        forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
        forward_dir[2] = 0
        forward_dir = forward_dir / torch.norm(forward_dir)
        random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]), convention="XYZ")
        forward_dir = torch.matmul(random_rot, forward_dir)
        # wpath[1, :2] = sigma*(2*torch.cuda.FloatTensor(2).uniform_()-1) #ending point xy
        wpath[1] = -forward_dir * r
        # # wpath[2] for inital body orientation
        # theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        # random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
        #                                                          convention="XYZ")
        # forward_dir = torch.matmul(random_rot, forward_dir)
        # wpath[2, :2] = forward_dir[:2]  # point to initialize the body orientation, not returned
        wpath[2, :2] = torch.randn(2)  # point to initialize the body orientation, not returned

        # left
        # wpath[1, 0] = 0.5
        # wpath[1, 1] = -0.2
        # wpath[2, :2] = - wpath[1, :2]  # point to initialize the body orientation, not returned
        # back
        # wpath[1, 0] = 0
        # wpath[1, 1] = 0.75
        # wpath[2, :2] = wpath[1, :2] #point to initialize the body orientation, not returned
        # front
        # wpath[1, 0] = 0
        # wpath[1, 1] = -0.75
        # wpath[2, :2] = wpath[1, :2] #point to initialize the body orientation, not returned
        # lie
        wpath[1, 0] = 0.75
        wpath[1, 1] = -0.75

        """ translate object and target body to the sampled location and make floor to be the plane z=0"""
        transl_xy = wpath[1, :2] - bm(**smplx_params).joints[0, 0, :2]
        transl_z = -torch.cuda.FloatTensor([interaction['floor_height']])
        transl = torch.cat([transl_xy, transl_z])
        smplx_params['transl'] = smplx_params['transl'] + transl
        transform[:3, 3] = transform[:3, 3] + transl.cpu().numpy()
        object_sdf['centroid'] = object_sdf['centroid'] + transl.cpu().numpy()
        # get transformed mesh
        object_mesh.apply_transform(transform)
        output = bm(**smplx_params)
        wpath[1] = output.joints[:, 0, :].detach()
        target_markers = output.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """generate init body"""
        xbo_dict = {}
        gender = 'male'
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None, ...]
        xbo_dict['transl'] = wpath[:1]  # [1,3]
        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict['transl'], wpath[0] = self.snap_to_ground_recenter_origin(xbo_dict,
                                                                           bm)  # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        start_markers = bm(**xbo_dict).vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """specify output"""
        # xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        xbo_dict['markers'] = torch.cat([start_markers, target_markers], dim=0)
        # xbo_dict['obj_mesh'] = object_mesh
        xbo_dict['obj_id'] = shapenet_id
        xbo_dict['obj_transform'] = torch.cuda.FloatTensor(transform)
        object_sdf['grid'] = torch.cuda.FloatTensor(object_sdf['grid'])
        object_sdf['centroid'] = torch.cuda.FloatTensor(object_sdf['centroid'])
        xbo_dict['obj_sdf'] = object_sdf
        obj_points, _ = trimesh.sample.sample_surface_even(object_mesh, 1024)
        xbo_dict['obj_points'] = torch.cuda.FloatTensor(obj_points)
        xbo_dict['target_body'] = deepcopy(smplx_params)

        """" reverse start and target body"""
        target_orient = R.from_rotvec(smplx_params['global_orient'].detach().cpu().numpy() if not reverse else xbo_dict[
            'global_orient'].detach().cpu().numpy())
        joints = bm(**(smplx_params if not reverse else xbo_dict)).joints  # [b,p,3]
        if reverse:
            for key in smplx_params:
                if key in xbo_dict:
                    xbo_dict['target_body'][key] = xbo_dict[key]
                xbo_dict[key] = smplx_params[key]
            xbo_dict['wpath'] = torch.flip(xbo_dict['wpath'], [0])
            xbo_dict['markers'] = torch.flip(xbo_dict['markers'], [0])

        """target orientation"""
        xbo_dict['target_orient'] = torch.cuda.FloatTensor(target_orient.as_rotvec())  # [1, 3]
        xbo_dict['target_orient_matrix'] = torch.cuda.FloatTensor(target_orient.as_matrix())  # [1, 3, 3]
        xbo_dict['wpath_orients'] = torch.cat([xbo_dict['global_orient'], xbo_dict['target_orient']], dim=0)
        # target_forward_dir = xbo_dict['target_orient_matrix'][:, :3, 2]  # [1, 3]
        # target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        # target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        # xbo_dict['target_forward_dir'] = target_forward_dir
        x_axis = joints[:, 2, :] - joints[:, 1, :]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
        z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
        y_axis = torch.cross(z_axis, x_axis)
        gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
        target_forward_dir = y_axis  # [1, 3], y-axis
        target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        xbo_dict['target_forward_dir'] = target_forward_dir

        if visualize:
            target_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict['target_body']).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([200, 100, 100])
            )
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                         transform=np.array([[1.0, 0.0, 0.0, 0],
                                                             [0.0, 1.0, 0.0, 0],
                                                             [0.0, 0.0, 1.0, -0.005],
                                                             [0.0, 0.0, 0.0, 1.0],
                                                             ]),
                                         )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            forward_dir_segment = torch.cat([joints[:, 0, :], joints[:, 0, :] + target_forward_dir], dim=0).detach().cpu().numpy()
            forward_dir_segment = trimesh.creation.annulus(0.01, 0.03, segment=forward_dir_segment)
            forward_dir_segment.visual.vertex_colors = np.array([0, 0, 255, 255])
            # forward_dir_segment.visual.vertex_colors = np.array([255, 0, 0, 255])
            vis_mesh = [floor_mesh,
                        target_body_mesh,
                        init_body_mesh,
                        object_mesh,
                        forward_dir_segment,
                        trimesh.creation.axis(),
                        ]
            for point_idx, pelvis in enumerate(xbo_dict['wpath']):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.cpu().numpy()
                trans_mat[:3, :3] = R.from_rotvec(xbo_dict['wpath_orients'][point_idx].detach().cpu().numpy()).as_matrix()
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)
            for marker in xbo_dict['markers'].reshape(-1, 3):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = marker.cpu().numpy()
                sm = trimesh.creation.uv_sphere(radius=0.02)
                sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                sm.apply_transform(trans_mat)
                vis_mesh.append(sm)
            print(xbo_dict['wpath'])
            # trimesh.util.concatenate(vis_mesh).show()
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        xbo_dict['betas'] = xbo_dict['betas'][0]
        # xbo_dict = self.params2numpy(xbo_dict)

        self.index_rec += 1

        return xbo_dict

class BatchGeneratorSittingTest(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67'  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        # pass

    def next_body(self, sigma=10, visualize=False, use_zero_pose=False,
                  mesh_path=None, sdf_path=None, target_pelvis=None, floor_height=0,
                  initial_body=None, to_numpy=False,
                  ):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        gender = 'male'
        bm = self.bm_male if gender == 'male' else self.bm_female
        if not os.path.exists(sdf_path):
            object_mesh = trimesh.load(
                mesh_path,
                force='mesh'
            )
            object_sdf = {
                'grid': mesh_to_voxels(object_mesh, 128, pad=False, surface_point_method='scan', sign_method='depth'),
                'dim': 128,
                'extent': np.max(object_mesh.bounding_box.extents),
                'centroid': object_mesh.bounding_box.centroid,
            }
            with open(sdf_path, 'wb') as f:
                pickle.dump(object_sdf, f)
        else:
            with open(sdf_path, 'rb') as f:
                object_sdf = pickle.load(f)

        object_mesh = trimesh.load(
            mesh_path,
            force='mesh'
        )

        """randomly specify a 3D path"""
        wpath = np.zeros((3, 3))
        # starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(
            wpath)  # starting point, ending point, another point to initialize the body orientation
        wpath[1] = torch.cuda.FloatTensor(target_pelvis[:3, 3])
        if initial_body is None:
            # r = torch.cuda.FloatTensor(1).uniform_() * 0.2 + 0.75
            r = 0.7
            forward_dir = torch.cuda.FloatTensor(target_pelvis[:3, 2])
            forward_dir[2] = 0
            forward_dir = forward_dir / torch.norm(forward_dir)
            wpath[0] = wpath[1] + forward_dir * r
            # wpath[2] for inital body orientation
            wpath[2, :2] = forward_dir[:2]  # point to initialize the body orientation, not returned

            """generate init body"""
            xbo_dict = {}
            gender = 'male'
            xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).normal_()
            xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                           output_type='aa').view(1, -1)
            xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None, ...]
            xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[:, 0, :]  # [1,3]
            """snap to the ground"""
            bm = self.bm_male if gender == 'male' else self.bm_female
            xbo_dict = self.snap_to_ground(xbo_dict, bm, floor_height=floor_height)  # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
            wpath[0] = bm(**xbo_dict).joints[:, 0, :]
        else:
            xbo_dict = {}
            gender = initial_body['gender']
            xbo_dict['betas'] = initial_body['betas']
            xbo_dict['transl'] = initial_body['smplx_params'][:, :3]
            xbo_dict['global_orient'] = initial_body['smplx_params'][:, 3:6]
            xbo_dict['body_pose'] = initial_body['smplx_params'][:, 6:69]
            bm = self.bm_male if gender == 'male' else self.bm_female
            wpath[0] = bm(**xbo_dict).joints[:, 0, :]

        """specify output"""
        # xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        # xbo_dict['obj_mesh'] = object_mesh
        xbo_dict['scene_path'] = mesh_path
        xbo_dict['floor_height'] = floor_height
        object_sdf['grid'] = torch.cuda.FloatTensor(object_sdf['grid'])
        object_sdf['centroid'] = torch.cuda.FloatTensor(object_sdf['centroid'])
        xbo_dict['obj_sdf'] = object_sdf
        obj_points, _ = trimesh.sample.sample_surface_even(object_mesh, 1024)
        xbo_dict['obj_points'] = torch.cuda.FloatTensor(obj_points)

        # target orientation
        target_orient = R.from_matrix(target_pelvis[:3, :3][None, ...])
        xbo_dict['target_orient'] = torch.cuda.FloatTensor(target_orient.as_rotvec())  # [1, 3]
        xbo_dict['target_orient_matrix'] = torch.cuda.FloatTensor(target_orient.as_matrix())  # [1, 3, 3]
        target_forward_dir = xbo_dict['target_orient_matrix'][:, :3, 2]  # [1, 3]
        target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        xbo_dict['target_forward_dir'] = target_forward_dir
        xbo_dict['wpath_orients'] = torch.cat([xbo_dict['global_orient'], xbo_dict['target_orient']], dim=0)
        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                         transform=np.array([[1.0, 0.0, 0.0, 0],
                                                             [0.0, 1.0, 0.0, 0],
                                                             [0.0, 0.0, 1.0, floor_height-0.005],
                                                             [0.0, 0.0, 0.0, 1.0],
                                                             ]),
                                         )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            vis_mesh = [floor_mesh, init_body_mesh,
                        object_mesh,
                        trimesh.creation.axis(),
                        trimesh.creation.axis(transform=target_pelvis)
                        ]
            print(xbo_dict['wpath'])
            # trimesh.util.concatenate(vis_mesh).show()
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        xbo_dict['betas'] = xbo_dict['betas'][0]
        if to_numpy:
            xbo_dict = self.params2numpy(xbo_dict)

        self.index_rec += 1

        return xbo_dict

class BatchGeneratorSceneNav(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67'  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        y_up_to_z_up = np.eye(4)
        y_up_to_z_up[:3, :3] = np.array(
            [[1, 0, 0],
             [0, 0, 1],
             [0, 1, 0]]
        )
        floor = trimesh.load(os.path.join(dataset_path, 'floor.obj'))
        obj = trimesh.load(os.path.join(dataset_path, 'object.obj'))
        self.obj_path = os.path.join(dataset_path, 'object.obj')
        floor.apply_transform(y_up_to_z_up)
        obj.apply_transform(y_up_to_z_up)
        self.floor = floor
        self.object = obj
        with open(os.path.join(dataset_path, 'path.json')) as f:
            way_points = json.load(f)
        x = [point['x'] * -1 for point in way_points]  # needs to negate x, not sure whether this is related to unity coords transform
        y = [point['y'] for point in way_points]
        z = [point['z'] for point in way_points]
        way_points = np.array([x, z, y]).T  # [p, 3]
        way_points[:, 2] = 0  # snap z to ground
        with open(os.path.join(dataset_path, 'target.json')) as f:
            self.target = np.array(json.load(f)).reshape(1, 3)
        with open(os.path.join(dataset_path, 'orient.json')) as f:
            self.orient = np.array(json.load(f)).reshape(1, 3)
        self.way_points = np.concatenate([way_points, self.target], axis=0)
        print(self.way_points)
    def next_body(self, sigma=10, visualize=False, use_zero_pose=True):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        """load path from navigation mesh"""
        wpath = torch.cuda.FloatTensor(self.way_points) #starting point, ending point, another point to initialize the body orientation
        gender = 'male'
        """generate init body"""
        xbo_dict = {}
        gender = 'male'
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).normal_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[1])[None, ...]
        xbo_dict['transl'] = wpath[:1].clone()  # [1,3]
        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        self.snap_to_ground(xbo_dict, bm)  # snap foot to ground
        wpath[0] = bm(**xbo_dict).joints.detach()[:, 0, :]
        wpath[:-1, 2] = wpath[0, 2]  #  assign the pelvis height to all way points except last one, which is sitting pose
        # start_markers = bm(**xbo_dict).vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """specify output"""
        object_mesh = self.object
        # xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath
        wpath_orients = []
        for point_idx in range(len(wpath) - 2):
            wpath_orients.append(self.get_bodyori_from_wpath(wpath[point_idx], wpath[point_idx + 1])[None, ...].detach().cpu().numpy())

        z_axis = (wpath[-2] - wpath[-1])
        z_axis[2] = 0
        z_axis = z_axis / torch.norm(z_axis)
        y_axis = torch.cuda.FloatTensor([0, 0, 1])
        x_axis = torch.cross(y_axis, z_axis)
        x_axis = x_axis / torch.norm(x_axis)
        glorot_mat = torch.stack([x_axis, y_axis, z_axis], dim=1)
        # wpath_orients += [RotConverter.rotmat2aa(glorot_mat).detach().cpu().numpy()]  # second last point
        wpath_orients += [self.orient]
        wpath_orients += [self.orient]  # last sitting point
        wpath_orients = np.concatenate(wpath_orients)
        xbo_dict['wpath_orients'] = wpath_orients
        # xbo_dict['markers'] = torch.cat([start_markers, target_markers], dim=0)
        # xbo_dict['obj_mesh'] = object_mesh
        # xbo_dict['obj_id'] = shapenet_id
        # xbo_dict['obj_transform'] = torch.cuda.FloatTensor(transform)
        # xbo_dict['obj_sdf'] = object_sdf
        obj_points, _ = trimesh.sample.sample_surface_even(object_mesh, 1024)
        xbo_dict['obj_points'] = obj_points
        xbo_dict['obj_path'] = self.obj_path
        if visualize:
            # target_body_mesh = trimesh.Trimesh(
            #     vertices=output.vertices[0].detach().cpu().numpy(),
            #     faces=bm.faces,
            #     vertex_colors=np.array([100, 100, 100])
            # )
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = self.floor
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            vis_mesh = [floor_mesh, init_body_mesh,
                        object_mesh,
                        trimesh.creation.axis()
                        ]
            from scipy.spatial.transform import Rotation as R
            for point_idx, pelvis in enumerate(wpath.reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)

                trans_mat[:3, :3] = R.from_rotvec(wpath_orients[point_idx]).as_matrix()
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)

            print(xbo_dict['wpath'])
            import pyrender
            scene = pyrender.Scene()
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
            for mesh in vis_mesh:
                viewer.render_lock.acquire()
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
                viewer.render_lock.release()
        xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict = self.params2numpy(xbo_dict)

        self.index_rec += 1

        return xbo_dict

class BatchGeneratorReplicaSceneNav(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67',  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 path_name_list=None,
                 target_pelvis_file_list=None,
                 scene=None,
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        if path_name_list is None:
            path_name_list = []
            print('no path file specified!')
        self.dataset_path = dataset_path = scene.replica_folder
        self.path_name_list = path_name_list
        self.target_pelvis_list = []
        self.scene = scene
        self.scene_name = scene_name = scene.name
        self.scene_path = dataset_path / self.scene_name / 'mesh.ply'
        self.scene_mesh = scene.mesh
        self.navmesh_path = dataset_path / self.scene_name / 'navmesh_tight.ply'
        self.navmesh = trimesh.load(self.navmesh_path, force='mesh')
        self.floor_height = scene.floor_height
        self.scene_mesh.vertices[:, 2] -= self.floor_height
        waypoints_folder = dataset_path / self.scene_name / 'waypoints'
        path_list = []
        for path_name in path_name_list:
            waypoints_path = waypoints_folder / (path_name + '.json')
            with open(waypoints_path) as f:
                way_points = json.load(f)
            x = [point['x'] for point in way_points]
            y = [point['y'] for point in way_points]
            z = [point['z'] for point in way_points]
            waypoints = np.array([x, y, z])  # [3, p]
            waypoints = np.matmul(unity_to_zup[:3, :3], waypoints).T  # [p, 3]
            waypoints[:, 2] -= self.floor_height
            path_list.append(waypoints)
        self.path_list = path_list
        self.path_idx = 0

        for pelvis_file in target_pelvis_file_list:
            with open(waypoints_folder / pelvis_file) as f:
                pelvis_frame = np.array(json.load(f)['pelvis_frame'])
                pelvis_frame[3, 2] -= self.floor_height
            self.target_pelvis_list.append(pelvis_frame)

    def interpolate_path(self, wpath):
        interpolated_path = [wpath[0]]
        last_point = wpath[0]
        for point_idx in range(1, wpath.shape[0]):
            while torch.norm(wpath[point_idx] - last_point) > 1:
                last_point = last_point + (wpath[point_idx] - last_point) / torch.norm(wpath[point_idx] - last_point)
                interpolated_path.append(last_point)
            last_point = wpath[point_idx]
            interpolated_path.append(last_point)
        return torch.stack(interpolated_path, dim=0)

    def next_body(self, visualize=False, use_zero_pose=True, wpath_on_floor=False,
                  single_wpath=False, interpolate_path=False,
                  to_numpy=True):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        """load path from navigation mesh"""
        wpath = torch.cuda.FloatTensor(self.path_list[self.path_idx]) #starting point, ending point, another point to initialize the body orientation
        if interpolate_path:
            wpath = self.interpolate_path(wpath)
        if single_wpath:
            wpath = torch.cat([wpath[:1, ...], wpath[-1:, ...]], dim=0)
        target_pelvis = torch.cuda.FloatTensor(self.target_pelvis_list[self.path_idx])
        self.path_idx = (self.path_idx + 1) % len(self.path_list)
        """generate init body"""
        xbo_dict = {}
        gender = 'male'
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[1])[None, ...]
        """move to init waypoint, snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict['transl'] = wpath[:1].clone() - bm(**xbo_dict).joints.detach()[:, 0, :]  # [1,3]
        xbo_dict = self.snap_to_ground(xbo_dict, bm)  # snap foot to ground
        wpath[0] = bm(**xbo_dict).joints.detach()[:, 0, :]
        if wpath_on_floor:
            wpath[1:, 2] = 0
        else:
            wpath[1:, 2] = wpath[0, 2]  #  assign the pelvis height to all way points
        # start_markers = bm(**xbo_dict).vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """specify output"""
        object_mesh = self.scene_mesh
        # xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath
        xbo_dict['last_orient'] = target_pelvis[:3, 2]  # body forward direction

        wpath_orients = []
        for point_idx in range(len(wpath) - 1):
            wpath_orients.append(RotConverter.aa2rotmat(self.get_bodyori_from_wpath(wpath[point_idx], wpath[point_idx + 1])[None, ...]).reshape(1, 3, 3))
        # orientation goal at each waypoint
        z_axis = target_pelvis[:3, 2]
        z_axis[2] = 0
        z_axis = z_axis / torch.norm(z_axis)
        y_axis = torch.cuda.FloatTensor([0, 0, 1])
        x_axis = torch.cross(y_axis, z_axis)
        x_axis = x_axis / torch.norm(x_axis)
        glorot_mat = torch.stack([x_axis, y_axis, z_axis], dim=1)
        wpath_orients += [glorot_mat[None, ...]]  # second last point
        wpath_orients = torch.cat(wpath_orients, dim=0)
        xbo_dict['wpath_orients_matrix'] = wpath_orients
        xbo_dict['wpath_orients_vec'] = RotConverter.rotmat2aa(wpath_orients)

        # xbo_dict['markers'] = torch.cat([start_markers, target_markers], dim=0)
        # xbo_dict['obj_mesh'] = object_mesh
        # xbo_dict['obj_id'] = shapenet_id
        # xbo_dict['obj_transform'] = torch.cuda.FloatTensor(transform)
        # xbo_dict['obj_sdf'] = object_sdf
        # obj_points, _ = trimesh.sample.sample_surface_even(object_mesh, 1024)
        # xbo_dict['obj_points'] = obj_points
        xbo_dict['scene_path'] = self.scene_path
        xbo_dict['navmesh_path'] = self.navmesh_path
        xbo_dict['navmesh'] = self.navmesh
        xbo_dict['floor_height'] = self.floor_height
        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            vis_mesh = [init_body_mesh,
                        object_mesh,
                        trimesh.creation.axis()
                        ]
            from scipy.spatial.transform import Rotation as R
            for point_idx, pelvis in enumerate(wpath.reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.cpu().numpy()
                trans_mat[:3, :3] = wpath_orients[point_idx].detach().cpu().numpy()
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)

            print(xbo_dict['wpath'])
            import pyrender
            scene = pyrender.Scene()
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
            for mesh in vis_mesh:
                viewer.render_lock.acquire()
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
                viewer.render_lock.release()
        xbo_dict['betas'] = xbo_dict['betas'][0]
        if to_numpy:
            xbo_dict = self.params2numpy(xbo_dict)

        self.index_rec += 1

        return xbo_dict

class BatchGeneratorReachOrient(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67'  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        # with open(os.path.join(dataset_path, 'orient.json')) as f:
        #     self.orient = np.array(json.load(f)).reshape(1, 3)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=True):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        """randomly specify a 3D path"""
        wpath = np.zeros((3,3))
        # wpath = torch.cuda.FloatTensor(3, 3).normal_() #starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(wpath) #starting point, ending point, another point to initialize the body orientation
        wpath[0] = 0 #starting point
        # wpath[1, :2] = sigma*(2*torch.cuda.FloatTensor(2).uniform_()-1) #ending point xy
        wpath[1, :2] = sigma * torch.randn(2)  # ending point xy
        # wpath[1, 2] = torch.rand(1) * 0.2 + 0.5
        theta = torch.pi * (2 * torch.cuda.FloatTensor(1).uniform_() - 1) * 0.5
        random_rotz = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]).reshape(1, 3), convention="XYZ")
        wpath[2] = torch.einsum('ij, j->i', random_rotz[0], wpath[1])
        # wpath[2, :2] = torch.cuda.FloatTensor(2).normal_() #point to initialize the body orientation, not returned


        """generate a body"""
        xbo_dict = {}
        # gender = random.choice(['male', 'female'])
        gender = random.choice(['male'])
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None,...]
        # gender = random.choice(['male'])
        # xbo_dict['betas'] = np.zeros([1,10])
        xbo_dict['transl'] = wpath[:1] #[1,3]

        """snap to the ground"""
        # xbo_dict = self.params2torch(xbo_dict)
        bm = self.bm_male if gender=='male' else self.bm_female
        # xbo_dict = self.snap_to_ground(xbo_dict, bm)
        # wpath[0] = bm(**xbo_dict).joints[0, 0].detach()  # recenter starting point at pelvis
        xbo_dict['transl'], wpath[0] = self.snap_to_ground_recenter_origin(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        wpath[1, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        from scipy.spatial.transform import Rotation as R
        rotation_z = R.from_euler('z', np.random.rand() * 360, degrees=True)
        init_orient = R.from_rotvec(xbo_dict['global_orient'].detach().cpu().numpy())
        xbo_dict['target_orient'] = torch.cuda.FloatTensor((rotation_z * init_orient).as_rotvec())  # [1, 3]
        xbo_dict['target_orient_matrix'] = torch.cuda.FloatTensor((rotation_z * init_orient).as_matrix())  # [1, 3, 3]
        target_forward_dir = xbo_dict['target_orient_matrix'][:, :3, 2]  # [1, 3]
        target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        xbo_dict['target_forward_dir'] = target_forward_dir
        self.index_rec += 1

        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                              transform=np.array([[1.0, 0.0, 0.0, 0],
                                                                  [0.0, 1.0, 0.0, 0],
                                                                  [0.0, 0.0, 1.0, -0.005],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            vis_mesh = [floor_mesh, init_body_mesh,
                        trimesh.creation.axis()
                        ]
            from scipy.spatial.transform import Rotation as R
            wpath_orients = torch.cat([xbo_dict['global_orient'], xbo_dict['target_orient']]).detach().cpu().numpy()
            for point_idx, pelvis in enumerate(wpath[:2, :].reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)

                trans_mat[:3, :3] = R.from_rotvec(wpath_orients[point_idx]).as_matrix()
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)

            print(xbo_dict['wpath'])
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        return xbo_dict

def vis_body(bm, bparam_dict, meshgp, wpath):
    body = o3d.geometry.TriangleMesh()
    smplxout = bm(**bparam_dict)
    verts = smplxout.vertices.detach().cpu().numpy().squeeze()
    body.vertices = o3d.utility.Vector3dVector(verts)
    body.triangles = o3d.utility.Vector3iVector(bm.faces)
    body.vertex_normals = o3d.utility.Vector3dVector([])
    body.triangle_normals = o3d.utility.Vector3dVector([])
    body.compute_vertex_normals()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    ball_list = []
    for i in range(len(wpath)):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        ball.paint_uniform_color(np.zeros(3))
        ball.translate(wpath[i], relative=False)
        ball_list.append(ball)

    o3d.visualization.draw_geometries([body, coord, meshgp]+ball_list)




### an example of how to use it
if __name__=='__main__':
    sys.path.append(os.getcwd())
    from exp_GAMMAPrimitive.utils import config_env
    import open3d as o3d
    import trimesh
    bm_path = config_env.get_body_model_path()
    host_name = config_env.get_host_name()
    batch_gen = BatchGeneratorSitting(dataset_path='data/interaction.pkl',
                                      shapenet_dir='/mnt/atlas_root/vlg-data/ShapeNetCore.v2/' if host_name == 'dalcowks' else '/vlg-data/ShapeNetCore.v2/',
                                      sdf_dir='data/shapenet_sdf',
                                      body_model_path=bm_path)
    for idx in range(32):
        data = batch_gen.next_body(sigma=None, use_zero_pose=False, visualize=True,
                               interaction_id=idx)

    # batch_gen = BatchGeneratorFollowPathInCubes(dataset_path='/mnt/hdd/tmp/slabs_with_navigation/slab008/slab_navimesh.obj_traj',
    #                                             body_model_path=bm_path)
    # batch_gen.get_rec_list()
    # for _ in range(10):
    #     data = batch_gen.next_body()
    #         # print(data['global_orient'])
    #         # print(data['wpath'][1])
    #         # print(data.keys())
    #     # data['transl'][:,:2]=0
    #     gender = data.pop('gender')
    #     wpath = data.pop('wpath')
    #     wpath_filename = data.pop('wpath_filename')
    #     bparam_dict =batch_gen.params2torch(data)
    #     bm = batch_gen.bm_male if gender=='male' else batch_gen.bm_female
    #     meshgp = trimesh.load("/mnt/hdd/tmp/slabs_with_navigation/slab008/slab.obj", force='mesh')
    #     vis_body(bm, bparam_dict, meshgp.as_open3d, wpath)


















