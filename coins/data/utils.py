import numpy as np
import torch
from torch.nn import functional as F

from configuration.config import *
# from human_body_prior.tools import tgm_conversion as tgm
import pytorch3d.transforms
def matrot2aa(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: Nx3
    '''
    # bs = pose_matrot.size(0)
    # homogen_matrot = F.pad(pose_matrot, [0,1])
    # pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot)
    pose = pytorch3d.transforms.matrix_to_axis_angle(pose_matrot)
    return pose

def aa2matrot(pose):
    '''
    :param Nx3
    :return: pose_matrot: Nx3x3
    '''
    bs = pose.size(0)
    num_joints = pose.size(1)//3
    # pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose)[:, :3, :3].contiguous()#.view(bs, num_joints*9)
    pose_body_matrot = pytorch3d.transforms.axis_angle_to_matrix(pose)[:, :3, :3].contiguous()  # .view(bs, num_joints*9)
    return pose_body_matrot

def smplx_dict_to_tensor(smplx_dict):
    # ['transl', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'betas']
    smplx_params = [smplx_dict[param] for param in used_smplx_param_names]
    smplx_tensor = torch.cat(smplx_params, dim=1)
    return smplx_tensor

def smplx_dict_to_rotmat(smplx_dict):
    rotvec = torch.cat([smplx_dict['global_orient'], smplx_dict['body_pose']], dim=1)
    rotmat = aa2matrot(rotvec.view(-1, 3)).view(-1, 3, 3)
    return rotmat  # batch*22 x 3 x 3

def smplx_dict_to_nonrot(smplx_dict, include_transl=True):
    nonrot = [smplx_dict['transl']] if include_transl else []
    nonrot += [
              smplx_dict['left_hand_pose'],
              smplx_dict['right_hand_pose'],
              smplx_dict['betas']
              ]
    return torch.cat(nonrot, dim=1)  # batch x (3 + num_pca *2 + 10)

