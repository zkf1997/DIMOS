import sys
sys.path.append('..')
from configuration.config import *

import smplx
import torch
import numpy as np

body_model_dict = {
        'male': smplx.create(smplx_model_folder, model_type='smplx',
                             gender='male', ext='npz',
                             num_pca_comps=num_pca_comps),
        'female': smplx.create(smplx_model_folder, model_type='smplx',
                               gender='female', ext='npz',
                               num_pca_comps=num_pca_comps),
        'neutral': smplx.create(smplx_model_folder, model_type='smplx',
                               gender='neutral', ext='npz',
                               num_pca_comps=num_pca_comps)
    }

# pca components in MANO seems not have 1 norm
def pose_to_pca(left_hand_pose, right_hand_pose, gender='neutral'):
    left_hand_pose = torch.tensor(left_hand_pose, dtype=torch.float32)
    right_hand_pose = torch.tensor(right_hand_pose, dtype=torch.float32)

    body_model = body_model_dict[gender]
    left_hand_mean = body_model.pose_mean[75:120].reshape(1, 45)
    right_hand_mean = body_model.pose_mean[120:165].reshape(1, 45)
    left_hand_comps = body_model.left_hand_components / (torch.norm(body_model.left_hand_components, dim=1, keepdim=True) ** 2)
    right_hand_comps = body_model.right_hand_components / (torch.norm(body_model.right_hand_components, dim=1, keepdim=True) ** 2)

    left_hand_pca = (left_hand_pose - left_hand_mean).matmul(left_hand_comps.T)
    right_hand_pca = (right_hand_pose - right_hand_mean).matmul(right_hand_comps.T)
    # print(left_hand_pca, right_hand_pca)
    return left_hand_pca, right_hand_pca

def pca_to_pose(left_hand_pca, right_hand_pca, gender='neutral'):
    left_hand_pca = torch.tensor(left_hand_pca, dtype=torch.float32)
    right_hand_pca = torch.tensor(right_hand_pca, dtype=torch.float32)

    body_model = body_model_dict[gender]
    left_hand_comps = body_model.left_hand_components
    right_hand_comps = body_model.right_hand_components
    left_hand_mean = body_model.pose_mean[75:120].reshape(1, 45)
    right_hand_mean = body_model.pose_mean[120:165].reshape(1, 45)

    left_hand_pose = left_hand_pca.matmul(left_hand_comps) + left_hand_mean
    right_hand_pose = right_hand_pca.matmul(right_hand_comps) + right_hand_mean
    # print(left_hand_pose, right_hand_pose)
    return left_hand_pose, right_hand_pose

if __name__ == '__main__':
    num_pca_comps = 45
    batch_size = 4
    body_model_dict = {
        'male': smplx.create(smplx_model_folder, model_type='smplx',
                             gender='male', ext='npz',
                             num_pca_comps=num_pca_comps, batch_size=batch_size),
        'female': smplx.create(smplx_model_folder, model_type='smplx',
                               gender='female', ext='npz',
                               num_pca_comps=num_pca_comps, batch_size=batch_size)
    }

    body_model = body_model_dict['male']
    left_hand_comps = body_model.left_hand_components
    # right_hand_comps = body_model.right_hand_components
    # print(left_hand_comps, right_hand_comps)
    # for idx in range(45):
    #     print(torch.norm(left_hand_comps[idx]))
    #     print(left_hand_comps[idx].dot(left_hand_comps[0]))

    left_hand_pose = torch.ones((1, 45), dtype=torch.float32)
    right_hand_pose = torch.ones((1, 45), dtype=torch.float32)
    left_hand_pca, right_hand_pca = pose_to_pca(left_hand_pose, right_hand_pose, 'male')
    left_hand_pose_rec, right_hand_pose_rec = pca_to_pose(left_hand_pca, right_hand_pca, gender='male')
    print(left_hand_pose_rec - left_hand_pose, right_hand_pose_rec - right_hand_pose)

    left_hand_pose = torch.rand((batch_size, num_pca_comps), dtype=torch.float32)
    right_hand_pose = torch.rand((batch_size, num_pca_comps), dtype=torch.float32)
    full_pose = body_model(left_hand_pose=left_hand_pose,
                           right_hand_pose=right_hand_pose,
                           return_full_pose=True).full_pose.detach()
    # print(full_pose.shape, full_pose)
    pose_hand = full_pose[:, 75:165]
    left, right = pca_to_pose(left_hand_pose, right_hand_pose)
    print(pose_hand - torch.cat([left, right], dim=1))
