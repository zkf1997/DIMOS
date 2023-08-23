import os
import sys
import numpy as np
import open3d as o3d
import torch
import smplx

def update_render_cam(cam_param,  trans):
    ### NOTE: trans is the trans relative to the world coordinate!!!

    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T) #!!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0,0,0,1]])
    mat = np.concatenate([cam_R, cam_T],axis=-1)
    mat = np.concatenate([mat, cam_aux],axis=0)
    cam_param.extrinsic = mat
    return cam_param


def create_lineset(x_range, y_range, z_range):
    gp_lines = o3d.geometry.LineSet()
    gp_pcd = o3d.geometry.PointCloud()
    points = np.stack(np.meshgrid(x_range, y_range, z_range), axis=-1)

    lines = []
    for ii in range( x_range.shape[0]-1):
        for jj in range(y_range.shape[0]-1):
            lines.append(np.array([ii*x_range.shape[0]+jj, ii*x_range.shape[0]+jj+1]))
            lines.append(np.array([ii*x_range.shape[0]+jj, ii*x_range.shape[0]+jj+y_range.shape[0]]))

    points = np.reshape(points, [-1,3])
    colors = np.random.rand(len(lines), 3)*0.5+0.5

    gp_lines.points = o3d.utility.Vector3dVector(points)
    gp_lines.colors = o3d.utility.Vector3dVector(colors)
    gp_lines.lines = o3d.utility.Vector2iVector(np.stack(lines,axis=0))
    gp_pcd.points = o3d.utility.Vector3dVector(points)

    return gp_lines, gp_pcd


def color_hex2rgb(hex):
    h = hex.lstrip('#')
    return np.array(  tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) )/255


def get_body_model(type, gender, batch_size,device='cpu'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model_path = '/home/kaizhao/dataset/models_smplx_v1_1/models/'
    body_model = smplx.create(body_model_path, model_type=type,
                                    gender=gender, ext='npz',
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
                                    batch_size=batch_size
                                    )
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model
