"""
generate sequences of sitting to  close chair using policy, can be combined with search
"""

import numpy as np
import random
import argparse
import os, sys, glob
import copy
import pickle
import pdb
import torch
from torch import optim
from tqdm import tqdm
import smplx
import time
import warnings

from pathlib import Path
import json
sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorSitting, BatchGeneratorSittingTest
from exp_GAMMAPrimitive.utils.environments import BatchGeneratorInteractionReplicaTest
from exp_GAMMAPrimitive.utils import config_env

from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP
from models.models_policy import GAMMAPolicy
from models.baseops import SMPLXParser
from models.searchop import MPTNode, MinHeap, calc_sdf
from models.baseops import get_logger
from models.chamfer_distance import chamfer_dists

from logging import log
import warnings
import numpy as np
import random
import argparse
import os, sys, glob
import torchgeometry as tgm
import copy
import pickle
import pdb
import json
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from human_body_prior.tools.model_loader import load_vposer

import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.environments import BatchGeneratorInteractionTrain, BatchGeneratorInteractionShapenetTest, BatchGeneratorInteractionTest
from exp_GAMMAPrimitive.utils import config_env

from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP
from models.models_policy import GAMMAPolicy
from models.baseops import SMPLXParser
from models.baseops import get_logger
from models.chamfer_distance import chamfer_dists
from models.searchop import *

def calc_sdf(vertices, sdf_dict, return_gradient=False):
    sdf_centroid = sdf_dict['centroid']
    sdf_scale = sdf_dict['scale']
    sdf_grids = sdf_dict['grid']


    batch_size, num_vertices, _ = vertices.shape
    vertices = vertices.reshape(1, -1, 3)  # [B, V, 3]
    vertices = (vertices - sdf_centroid) / sdf_scale  # convert to [-1, 1]
    sdf_values = F.grid_sample(sdf_grids,
                                   vertices[:, :, [2, 1, 0]].view(1, batch_size * num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                   padding_mode='border',
                                   align_corners=True
                                   # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                   ).reshape(batch_size, num_vertices)
    if return_gradient:
        sdf_gradients = sdf_dict['gradient_grid']
        gradient_values = F.grid_sample(sdf_gradients,
                                   vertices[:, :, [2, 1, 0]].view(1, batch_size * num_vertices, 1, 1, 3),
                                   # [2,1,0] permute because of grid_sample assumes different dimension order, see below
                                   padding_mode='border',
                                   align_corners=True
                                   # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                   ).permute(2, 1, 0, 3, 4).reshape(batch_size, num_vertices, 3)
        gradient_values = gradient_values / torch.norm(gradient_values, dim=-1, keepdim=True).clip(min=1e-12)
        return sdf_values, gradient_values
    '''
    # illustration of grid_sample dimension order, assume first dimension to be innermost
    # import torch
    # import torch.nn.functional as F
    # import numpy as np
    # sz = 5
    # input_arr = torch.from_numpy(np.arange(sz * sz).reshape(1, 1, sz, sz)).float()
    # indices = torch.from_numpy(np.array([-1, -1, -0.5, -0.5, 0, 0, 0.5, 0.5, 1, 1, -1, 0.5, 0.5, -1]).reshape(1, 1, 7, 2)).float()
    # out = F.grid_sample(input_arr, indices, align_corners=True)
    # print(input_arr)
    # print(out)
    '''
    return sdf_values


sys.setrecursionlimit(10000)  # if too small, deepcopy will reach the maximal depth limit.

def configure_model(cfg, gpu_index, seed):
    cfgall = ConfigCreator(cfg)
    modelcfg = cfgall.modelconfig
    traincfg = cfgall.trainconfig
    predictorcfg = ConfigCreator(modelcfg['predictor_config'])
    regressorcfg = ConfigCreator(modelcfg['regressor_config'])

    testcfg = {}
    testcfg['gpu_index'] = gpu_index
    testcfg['ckpt_dir'] = traincfg['save_dir']
    testcfg['result_dir'] = cfgall.cfg_result_dir
    testcfg['seed'] = seed
    testcfg['log_dir'] = cfgall.cfg_log_dir
    testop = GAMMAPrimitiveComboGenOP(predictorcfg, regressorcfg, testcfg)
    testop.build_model(load_pretrained_model=True)

    return testop


def gen_motion_one_step(motion_model, policy_model,
                        states, bparam_seed, prev_betas, t_his, local_map=None):
    # print(states.shape, bparam_seed.shape, prev_betas.shape, local_map.shape)
    if t_his == 1:
        n_gens = n_gens_1frame
    elif t_his == 2:
        n_gens = n_gens_2frame
    [pred_markers, pred_params,
     act, act_log_prob,
     value] = motion_model.generate_ppo(policy_model,
                                        states.permute([1, 0, 2]),
                                        bparam_seed.permute([1, 0, 2]),
                                        prev_betas,
                                        to_numpy=False,
                                        n_gens=-1,
                                        param_blending=True,
                                        local_map=local_map,
                                        use_policy_mean=USE_POLICY_MEAN,
                                        use_policy=USE_POLICY
                                        )
    pred_markers = pred_markers.reshape(pred_markers.shape[0],
                                             pred_markers.shape[1], -1, 3)  # [t, n_gens, V, 3]
    return pred_markers.permute([1, 0, 2, 3]), pred_params.permute([1, 0, 2]), act

def canonicalize_static_pose(data):
    smplx_transl = data['transl']
    smplx_glorot = data['global_orient']
    smplx_poses = data['body_pose']
    gender = data['gender']
    smplx_handposes = torch.cuda.FloatTensor(smplx_transl.shape[0], 24).zero_()
    prev_params = torch.cat([smplx_transl, smplx_glorot,
                             smplx_poses, smplx_handposes],
                            dim=-1)  # [t,d]
    prev_params = prev_params.repeat(n_gens_1frame, 1, 1)
    prev_betas = data['betas']
    nb, nt = prev_params.shape[:2]
    ## move frame to the body's pelvis
    R0, T0 = smplxparser_1frame_batch.get_new_coordinate(
        betas=prev_betas,
        gender=gender,
        xb=prev_params[:, 0],
        to_numpy=False)

    ## get the last body param and marker in the new coordinate
    body_param_seed = smplxparser_1frame_batch.update_transl_glorot(R0, T0,
                                                              betas=prev_betas,
                                                              gender=gender,
                                                              xb=prev_params.reshape(nb * nt, -1),
                                                              to_numpy=False
                                                              ).reshape(nb, nt, -1)
    marker_seed = smplxparser_1frame_batch.get_markers(
        betas=prev_betas,
        gender=gender,
        xb=body_param_seed.reshape(nb * nt, -1),
        to_numpy=False).reshape(nb, nt, -1, 3)
    pelvis_loc = smplxparser_1frame_batch.get_jts(betas=prev_betas,
                                            gender=gender,
                                            xb=body_param_seed.reshape(nb * nt, -1),
                                            to_numpy=False)[:, 0].reshape(nb, nt, 3)  # [b, t, 3]

    return marker_seed, body_param_seed, R0, T0, pelvis_loc


def get_feature(Y_l, pel, R0, T0, pt_wpath, scene):
    '''
    --Y_l = [b,t,d] local marker
    --pel = [b,t,d]
    --pt_wpath = [1,d]
    '''
    nb, nt = pel.shape[:2]
    Y_l = Y_l.reshape(nb, nt, -1, 3)
    pt_markers_l_3d = torch.einsum('bij,btpj->btpi', R0.permute(0, 2, 1), scene['markers'][None, -1:, :, :] - T0[:, None, :, :])

    '''extract marker feature'''
    fea_marker = pt_markers_l_3d - Y_l
    dist_marker_target = torch.norm(fea_marker, dim=-1, keepdim=True).clip(min=1e-12)
    dir_marker_target = (fea_marker / dist_marker_target).reshape(nb, nt, -1)
    dist_last = dist_marker_target[:, -1, :, 0].mean(dim=-1)

    target_ori_world = scene['target_forward_dir']  # [1, 3]
    target_ori_local = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1), target_ori_world[None, ...])  # [b, 1, 3]
    target_ori_local = target_ori_local.squeeze(1)  # [b, 3]
    target_ori_local = target_ori_local[:, :2] / torch.norm(target_ori_local[:, :2], dim=-1,
                                                            keepdim=True).clip(min=1e-12)  # [b, 2], only care about xy projection on floor

    """distance from marker to target object"""
    # obj_points = scene['obj_points']
    # obj_points_l = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1), obj_points[None, ...] - T0)  # b, p, 3
    # obj_points_l = obj_points_l[:, None, :, :].expand(nb, nt, -1, 3)
    # fea_marker_dist = chamfer_dists(Y_l.reshape(nb * nt, -1, 3), obj_points_l.reshape(nb * nt, -1, 3)).reshape(nb, nt, -1)  # b, t, 67
    """use sdf to calc dists"""
    Y_w = torch.einsum('bij,btpj->btpi', R0, Y_l) + T0[:, None, :, :]  # [b, t, p, 3]
    sdf_values, sdf_gradients = calc_sdf(Y_w.reshape(nb * nt, -1, 3), scene['obj_sdf'], return_gradient=True)
    sdf_values = sdf_values.reshape(nb, nt, -1)
    sdf_gradients = sdf_gradients.reshape(nb, nt, -1, 3)
    # print(torch.sqrt(fea_marker_dist[0, 0]))
    # print(sdf_values[0, 0])
    # print(torch.sqrt(fea_marker_dist[0, 0]) - sdf_values[0, 0])
    # print((torch.sqrt(fea_marker_dist) - sdf_values).max(), (torch.sqrt(fea_marker_dist) - sdf_values).min())

    if cfg.modelconfig.truncate:
        dist_marker_target = dist_marker_target.clip(min=-cfg.modelconfig.truncate_dist, max=cfg.modelconfig.truncate_dist)
        sdf_values = sdf_values.clip(min=-cfg.modelconfig.truncate_dist, max=cfg.modelconfig.truncate_dist)
        clip_mask = (sdf_values == cfg.modelconfig.truncate_dist)[:, :, :, None].repeat(1, 1, 1, 3)
        # print(sdf_values.shape, sdf_gradients.shape, clip_mask.shape)
        sdf_gradients[clip_mask] = 0.0
    return dist_last, dist_marker_target, dir_marker_target, sdf_values, sdf_gradients, target_ori_local

def get_rewards(bparams, vertices, joints, Y_l, R0, T0, wpath, scene, target_ori_local, last_step=False):
    t1 = time.time()
    pel_loc = joints[:, :, 0]
    Y_w = torch.einsum('bij,btpj->btpi', R0, Y_l) + T0[:, None, :, :]  # [b, t, p, 3]
    nb, nt = Y_l.shape[:2]
    h = 1 / 40
    Y_l_speed = torch.norm(Y_l[:, 2:, :, :2] - Y_l[:, :-2, :, :2], dim=-1) / (2 * h)  # [b, t=9,P=67]
    # Y_l_speed = Y_l_speed.reshape(nb, -1)
    pt_markers_l_3d = torch.einsum('bij,btpj->btpi', R0.permute(0, 2, 1),
                                   scene['markers'][None, -1:, :, :] - T0[:, None, :, :])
    '''evaluate contact soft'''
    dist2gp = torch.abs(Y_w[:, :, feet_marker_idx, 2].amin(dim=-1) - 0.02).mean(dim=-1)  # feet on floor
    # dist2skat = Y_l_speed.amin(dim=-1)
    dist2skat = (Y_l_speed[:, :, feet_marker_idx].amin(dim=-1) - 0.075).clamp(min=0).mean(dim=-1)
    r_floor = torch.exp(-dist2gp)
    r_skate = torch.exp(-dist2skat)
    r_contact_feet = r_floor * r_skate

    '''evaluate body pose naturalness via vposer'''
    body_pose = bparams[:, :, 6:69].reshape(nt * nb, -1)
    vp_embedding = vposer.encode(body_pose).loc
    latent_dim = vp_embedding.shape[-1]
    vp_norm = torch.norm(vp_embedding.reshape(nb, nt, -1), dim=-1).mean(dim=1)
    r_vp = torch.exp(-vp_norm / (latent_dim ** 0.5))

    '''evaluate body facing orientation'''
    joints_end = joints[:, -1]  # [b,p,3]
    x_axis = joints_end[:, 2, :] - joints_end[:, 1, :]
    x_axis[:, -1] = 0
    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
    z_axis = torch.cuda.FloatTensor([[0, 0, 1]], device=device).repeat(x_axis.shape[0], 1)
    y_axis = torch.cross(z_axis, x_axis)
    b_ori = y_axis[:, :2]
    t_ori = target_ori_local  # body forward dir of SMPL is z axis
    r_target_ori = (torch.einsum('bi,bi->b', t_ori, b_ori) + 1) / 2.0
    """dist to target"""
    dist2target = torch.norm(pt_markers_l_3d[:, :, :, :] - Y_l[:, :, :, :], dim=-1, keepdim=False).mean(dim=-1).min(dim=-1)[0]  # [b]
    r_target_dist = (1 - (dist2target).clip(min=1e-10) ** 0.5)

    # penetration reward
    t1 = time.time()
    # sdf_values = calc_sdf(Y_w.reshape(nb * nt, -1, 3), scene['obj_sdf']).reshape(nb, -1)  #[b, t*p]
    vertices_w = torch.einsum('bij,btpj->btpi', R0, vertices) + T0[:, None, :, :]  # [b, t, p, 3]
    sdf_values = calc_sdf(vertices_w.reshape(nb * nt, -1, 3), scene['obj_sdf']).reshape(nb, -1)  # [b, t*p]
    """also consider penetration with floor"""
    # sdf_values = torch.minimum(sdf_values, vertices_w[:, :, :, 2].reshape(nb, -1))
    # percent_inside = sdf_values.lt(0.0).sum(dim=-1) / sdf_values.shape[-1] # [b, ]
    num_inside = sdf_values.lt(0.0).sum(dim=-1)  # [b, ]
    negative_values = sdf_values * (sdf_values < 0)
    # r_penetration = torch.exp(-num_inside / nt / 512)
    r_penetration = torch.exp((negative_values.sum(dim=-1) / nt / 512).clip(min=-100))
    # r_penetration = negative_values.sum(dim=-1) / nt
    if args.profile:
        print(vertices_w.shape)
        print('sdf query time:', time.time() - t1)

    interaction_marker_sdf = sdf_values.reshape(nb, nt, -1)[:, :, [5487, 8391, 5697, 3336, 6127, 3617, 6378, 3464, 6225]]  # [b, t, 9], sit_lie
    # interaction_marker_sdf = sdf_values.reshape(nb, nt, -1)[:, :, [3617, 6378, 3464, 6225]]  # [b, t, 4], sit
    # interaction_marker_sdf = sdf_values.reshape(nb, nt, -1)[:, :, [5487, 8391, 5697, 3336, 6127]]  # [b, t, 4], lie
    r_interaction = torch.exp(-interaction_marker_sdf.abs().mean(dim=[1, 2]) / 0.1)

    # interaction reward, hip markers have zero sdf
    if not last_step and cfg.lossconfig.sparse_reward:
        # if False:
        r_interaction = 0.0
        r_target_dist = 0.0
        r_target_ori = 0.0

    # reward = r_contact+r_target+BODY_ORI_WEIGHT*r_ori+r_vp+r_slow
    reward = r_contact_feet * cfg.lossconfig.weight_contact_feet + \
             r_vp * cfg.lossconfig.weight_vp + \
             r_penetration * cfg.lossconfig.weight_pene + \
             r_interaction * cfg.lossconfig.weight_interaction + \
             r_target_dist * cfg.lossconfig.weight_target_dist + \
             r_target_ori * cfg.lossconfig.weight_target_ori
    # reward = reward + r_floor + r_pelvis_vel

    info = {
        'reward': reward,
        'r_floor': r_floor,
        'r_skate': r_skate,
        'r_vp': r_vp,
        'r_interaction': r_interaction,
        'r_pene': r_penetration,
        'r_target_dist': r_target_dist,
        'dist2target': dist2target,
    }
    return reward, info

def gen_tree_roots(start_node, wpath, scene):
    mp_heap = MinHeap()

    # canonicalize the starting pose
    gender = str(start_node.data['gender'])
    marker_seed = start_node.data['markers']
    body_param_seed = start_node.data['smplx_params']
    R0 = start_node.data['transf_rotmat']
    T0 = start_node.data['transf_transl']
    motion_model = genop_1frame_male if gender == 'male' else genop_1frame_female
    prev_betas = start_node.data['betas'].unsqueeze(0)

    pelvis_loc = start_node.data['pelvis_loc']

    ## retrieve current target and update it
    dist_last, dist_marker_target, dir_marker_target, sdf_values, sdf_gradients, target_ori_local = get_feature(marker_seed, pelvis_loc,
                                                                                                                R0, T0, wpath[-1:], scene)

    nb, nt = body_param_seed.shape[:2]
    states = torch.cat([marker_seed.reshape(nb, nt, -1), dist_marker_target.reshape(nb, nt, -1),
                        dir_marker_target.reshape(nb, nt, -1), sdf_values.reshape(nb, nt, -1),
                        sdf_gradients.reshape(nb, nt, -1)], dim=-1)

    # generate markers and regress to body params
    pred_markers, pred_params, pred_latent = gen_motion_one_step(motion_model, policy_model,
                                                                 states, body_param_seed,
                                                                 prev_betas,
                                                                 t_his=1,
                                                                 )
    nb, nt = pred_params.shape[:2]
    for ii in range(pred_markers.shape[0]):
        pred_joints = smplxparser_mp.get_jts(betas=start_node.data['betas'],
                                        gender=gender,
                                        xb=pred_params[ii],
                                        to_numpy=False).unsqueeze(0)  # [1, t, p, 3]
        pelvis_loc = pred_joints[:, :, 0]  # [1, t, 3]
        pred_markers_proj = smplxparser_mp.get_markers(betas=start_node.data['betas'],
                                                       gender=gender,
                                                       xb=pred_params[ii],
                                        to_numpy=False).unsqueeze(0)  # [1, t, p, 3]
        pred_vertices = smplxparser_mp.forward_smplx(betas=start_node.data['betas'],
                                                     gender=gender,
                                                     xb=pred_params[ii],
                                                     output_type='vertices',
                                                     to_numpy=False).reshape([1, nt, -1, 3])


        pred_marker_b = REPROJ_FACTOR * pred_markers_proj + (1 - REPROJ_FACTOR) * pred_markers[[ii]]  # [1, t, p=67, 3]
        idx_target_curr = 1
        rootnode = MPTNodeTorch(gender, start_node.data['betas'], R0[[ii]], T0[[ii]], pelvis_loc, pred_joints,
                           pred_markers[ii:ii + 1], pred_markers_proj, pred_params[ii:ii + 1], pred_latent[ii:ii + 1],
                           '1-frame',
                           timestamp=0, curr_target_wpath=(idx_target_curr, wpath[idx_target_curr]))
        rewards, info = get_rewards(pred_params[[ii]], pred_vertices, pred_joints,
                                              pred_marker_b, R0[[ii]], T0[[ii]], wpath[[idx_target_curr]], scene, target_ori_local)
        rootnode.quality = rewards[0].item() # reward for achiveing waypoints
        rootnode.info = info
        mp_heap.push(rootnode)
    return mp_heap


def expand_tree(mp_heap_prev, wpath, scene=None, max_depth=10):
    mp_heap_curr = MinHeap()
    # generate child treenodes
    for iop in range(0, max_depth):
        print('[INFO] at level {}'.format(iop))
        idx_node = 0
        while (not mp_heap_prev.is_empty()) and (idx_node < max_nodes_to_expand):
            mp_prev = mp_heap_prev.pop()
            idx_node += 1

            '''produce marker seed'''
            t_his = 2
            prev_params = copy.deepcopy(mp_prev.data['smplx_params'])
            prev_markers = copy.deepcopy(mp_prev.data['markers'])
            prev_markers_proj = copy.deepcopy(mp_prev.data['markers_proj'])
            prev_pel_loc = copy.deepcopy(mp_prev.data['pelvis_loc'])
            prev_betas = mp_prev.data['betas']
            prev_gender = mp_prev.data['gender']
            prev_rotmat = copy.deepcopy(mp_prev.data['transf_rotmat'])
            prev_transl = copy.deepcopy(mp_prev.data['transf_transl'])
            body_param_seed = prev_params[:, -t_his:]  #[b=1, t_his, d]
            nb, nt = body_param_seed.shape[:2]
            ## move frame to the second last body's pelvis
            R_, T_ = smplxparser_1frame.get_new_coordinate(
                betas=prev_betas,
                gender=prev_gender,
                xb=body_param_seed[:, 0],
            to_numpy=False)
            T0 = torch.einsum('bij,bpj->bpi', prev_rotmat, T_) + prev_transl
            R0 = torch.einsum('bij,bjk->bik', prev_rotmat, R_)
            ## get the last body param and marker in the new coordinate
            body_param_seed = smplxparser_2frame.update_transl_glorot(
                R_.repeat(t_his, 1, 1),
                T_.repeat(t_his, 1, 1),
                betas=prev_betas,
                gender=prev_gender,
                xb=body_param_seed.reshape(nb * nt, -1),
            to_numpy=False,
            inplace=False).reshape(nb, nt, -1)

            ## blend predicted markers and the reprojected markers to eliminated jitering
            marker_seed = REPROJ_FACTOR * prev_markers_proj[:, -t_his:] + (1 - REPROJ_FACTOR) * prev_markers[:, -t_his:]
            marker_seed = torch.einsum('bij,btpj->btpi', R_.permute(0, 2, 1), marker_seed - T_[..., None, :])
            pel_loc_seed = torch.einsum('bij,btj->bti', R_.permute(0, 2, 1), prev_pel_loc[:, -t_his:] - T_)

            """get states"""
            dist_last, dist_marker_target, dir_marker_target, sdf_values, sdf_gradients, target_ori_local = get_feature(
                marker_seed, pel_loc_seed,
                R0, T0, wpath[-1:], scene)
            nb, nt = body_param_seed.shape[:2]
            states = torch.cat([marker_seed.reshape(nb, nt, -1), dist_marker_target.reshape(nb, nt, -1),
                                dir_marker_target.reshape(nb, nt, -1), sdf_values.reshape(nb, nt, -1),
                                sdf_gradients.reshape(nb, nt, -1)], dim=-1)

            '''generate future motions'''
            motion_model = genop_2frame_male if prev_gender == 'male' else genop_2frame_female

            prev_betas_torch = prev_betas.unsqueeze(0)

            pred_markers, pred_params, pred_latent = gen_motion_one_step(motion_model, policy_model,
                                states, body_param_seed,
                                prev_betas_torch,
                                t_his=t_his,
                                )  # smplx [n_gens, n_frames, d]


            '''sort generated primitives'''
            idx_target_curr = 1
            for ii in range(pred_markers.shape[0]):
                pred_joints = smplxparser_mp.get_jts(betas=prev_betas,
                                                gender=prev_gender,
                                                xb=pred_params[ii],
                                                to_numpy=False,).unsqueeze(0)  # [1, t, p, 3]
                pelvis_loc = pred_joints[:, :, 0]  # [1, t, 3]
                pred_markers_proj = smplxparser_mp.get_markers(betas=prev_betas,
                                                               gender=prev_gender,
                                                               xb=pred_params[ii],
                                                to_numpy=False,).unsqueeze(0)  # [1, t, p, 3]
                pred_vertices = smplxparser_mp.forward_smplx(betas=prev_betas,
                                                             gender=prev_gender,
                                                             xb=pred_params[ii],
                                                             output_type='vertices',
                                                             to_numpy=False).reshape([1, nt, -1, 3])
                mp_curr = MPTNodeTorch(prev_gender, prev_betas, R0, T0, pelvis_loc, pred_joints,
                                  pred_markers[ii:ii + 1], pred_markers_proj, pred_params[ii:ii + 1],
                                  pred_latent[ii:ii + 1], '2-frame',
                                  timestamp=iop, curr_target_wpath=(idx_target_curr, wpath[idx_target_curr]))
                pred_marker_b = REPROJ_FACTOR * pred_markers_proj + (1 - REPROJ_FACTOR) * pred_markers[
                    [ii]]  # [1, t, p=67, 3]
                rewards, info = get_rewards(pred_params[[ii]], pred_vertices, pred_joints,
                                                 pred_marker_b, R0, T0, wpath[[idx_target_curr]], scene, target_ori_local)
                mp_curr.quality = rewards[0].item()  # reward for achiveing waypoints
                # print(info, mp_curr.quality)
                mp_curr.info = info
                mp_curr.set_parent(mp_prev)
                mp_heap_curr.push(mp_curr)


        if mp_heap_curr.len() == 0:
            print('[INFO] |--no movements searched. Program terminates.')
            return None
            # sys.exit()
        print(
            '[INFO] |--valid MPs={}, path_finished={}/{}'.format(
                mp_heap_curr.len(),
                mp_heap_curr.data[0].data['curr_target_wpath'][0], len(wpath),
            ))
        info_str = ''
        for key in mp_heap_curr.data[0].info:
            info_str += ' {}:{:.2f},'.format(key, mp_heap_curr.data[0].info[key][0].item())
        print(info_str)
        mp_heap_prev.clear()
        mp_heap_prev = copy.deepcopy(mp_heap_curr)
        mp_heap_curr.clear()


        # early stop, if the best node is close enough to the target, search stops and return
        # TODO: add check for orientation foe early stop
        if mp_heap_prev.data[0].data['curr_target_wpath'][0] == len(wpath) - 1:
            # and mp_heap_prev.data[0].dist2ori < args.ori_thresh
            if mp_heap_prev.data[0].info['dist2target'][0] < args.goal_thresh_final:
                print('[INFO] |--find satisfactory solutions. Search finishes.')
                return mp_heap_prev

    return mp_heap_prev

def gen_motion(body_s, max_depth=10, outfoldername=None):
    '''
    idx: the sequence seed index, which is to name the output file actually.
    max_depth: this determines how long the generated motion is (0.25sec per motion prmitive)
    '''
    wpath = body_s['wpath']
    # specify the start node
    if body_s['motion_history'] is None or args.history_mode == 1:
    # if True:
        [marker_start, body_param_start,
         R_start, T_start, pelvis_loc_start] = canonicalize_static_pose(body_s)
        start_node = MPTNodeTorch(str(body_s['gender']), body_s['betas'], R_start, T_start, pelvis_loc_start, None,
                             marker_start, marker_start, body_param_start, None, 'start-frame',
                             timestamp=-1,
                             curr_target_wpath=(1, wpath[1])
                             )
        # depending on a static pose, generate a list of tree roots
        print('[INFO] generate roots in a heap')
        mp_heap_prev = gen_tree_roots(start_node, wpath, body_s)
        print('[INFO] |--valid MPs={}'.format(mp_heap_prev.len()))
        if mp_heap_prev.len() == 0:
            print('[INFO] |--no movements searched. Program terminates.')
            return
    else:
        mp_heap_prev = MinHeap()
        def params2torch(params, dtype=torch.float32):
            return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}
        last_mp = params2torch(body_s['motion_history']['motion'][-1])
        start_node = MPTNodeTorch(last_mp['gender'], last_mp['betas'],
                             last_mp['transf_rotmat'], last_mp['transf_transl'], last_mp['pelvis_loc'], last_mp['joints'],
                             last_mp['markers'], last_mp['markers_proj'], last_mp['smplx_params'], last_mp['mp_latent'], '2-frame',
                             timestamp=-1,
                             curr_target_wpath=(1, wpath[1])
                             )
        mp_heap_prev.push(start_node)

    # generate tree leaves
    mp_heap_prev = expand_tree(mp_heap_prev, wpath, body_s, max_depth=max_depth)
    if mp_heap_prev is None:
        return
    output = {'motion': [], 'wpath': wpath.detach().cpu().numpy()}
    if body_s['motion_history'] is not None:
        if args.history_mode == 1:
            output['motion'] = body_s['motion_history']['motion']
        else:
            output['motion'] = body_s['motion_history']['motion'][:-1]  # if use last primitive last 2 frames to build start node, need to avoid repeating
        output['wpath'] = np.concatenate([body_s['motion_history']['wpath'], output['wpath']], axis=0)
    if 'wpath_orients_vec' in body_s:
        output['wpath_orients'] = body_s['wpath_orients_vec'].detach().cpu().numpy()
    if 'scene_path' in body_s:
        output['scene_path'] = body_s['scene_path']
        output['floor_height'] = body_s['floor_height']
    if 'obj_transform' in body_s:
        output['obj_transform'] = body_s['obj_transform']
    print('[INFO] save results...')
    mp_leaves = mp_heap_prev
    motion_idx = 0
    while not mp_leaves.is_empty():
        if motion_idx >= 1:
            break
        gen_results = []
        mp_leaf = mp_leaves.pop()
        gen_results.append(mp_leaf.data)
        while mp_leaf.parent is not None:
            gen_results.append(mp_leaf.parent.data)
            mp_leaf = mp_leaf.parent
        gen_results.reverse()
        for primitive in gen_results:
            for key in primitive:
                if type(primitive[key]) == torch.Tensor:
                    primitive[key] = primitive[key].detach().cpu().numpy()
                elif key == 'curr_target_wpath':
                    primitive[key] = (primitive[key][0], primitive[key][1].detach().cpu().numpy())
        output['motion'] += gen_results
        ### save to file
        outfilename = 'results_{}_{}.pkl'.format(body_repr, motion_idx)
        outfilename_f = os.path.join(outfoldername, outfilename)
        print("save at:", outfilename_f)
        with open(outfilename_f, 'wb') as f:
            pickle.dump(output, f)
        motion_idx += 1

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
# create dirs for saving, then add to cfg
def create_dirs(cfg):
    with open_dict(cfg):
        # create dirs
        cfg.cfg_exp_dir = os.path.join(args.project_dir, 'results', 'exp_GAMMAPrimitive', cfg.cfg_name, cfg.wandb.name)
        cfg.cfg_result_dir = os.path.join(cfg.cfg_exp_dir, 'results')
        cfg.cfg_ckpt_dir = os.path.join(cfg.cfg_exp_dir, 'checkpoints')
        cfg.cfg_log_dir = os.path.join(cfg.cfg_exp_dir, 'logs')
        os.makedirs(cfg.cfg_result_dir, exist_ok=True)
        os.makedirs(cfg.cfg_ckpt_dir, exist_ok=True)
        os.makedirs(cfg.cfg_log_dir, exist_ok=True)

        # specify missed experiment settings
        cfg['trainconfig']['save_dir'] = cfg.cfg_ckpt_dir
        cfg['trainconfig']['log_dir'] = cfg.cfg_log_dir


parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str, default='orient_state')
parser.add_argument('--checkpoint', type=str, default='last.ckp')
parser.add_argument('--max_depth', type=int, default=60,
                    help='the maximal number of (0.25-second) motion primitives in each motion.')
parser.add_argument('--num_sequence', type=int, default=4)
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--num_gen1', type=int, default=16)
parser.add_argument('--num_gen2', type=int, default=4)
parser.add_argument('--num_expand', type=int, default=4)
parser.add_argument('--weight_target', type=float, default=0.1)
parser.add_argument('--weight_ori', type=float, default=0.1)

parser.add_argument('--switch_stage', type=int, default=0, help='opt to switch off policy when close to target than switch thresh')
parser.add_argument('--switch_thresh', type=float, default=0.75)
parser.add_argument('--goal_thresh', type=float, default=0.05)
parser.add_argument('--goal_thresh_final', type=float, default=0.25)
parser.add_argument('--ori_thresh', type=float, default=0.3)
parser.add_argument('--goal_sigma', type=float, default=0.25)
parser.add_argument('--use_policy', type=int, default=1)
parser.add_argument('--hard_code', type=int, default=0)
parser.add_argument('--use_orient', type=int, default=1)
parser.add_argument('--env_cfg', type=str, default='0')
parser.add_argument('--gen_name', type=str, default='test')

parser.add_argument('--ground_euler', nargs=3, type=float, default=[0, 0, 0],
                    help='the gorund plan rotation. Normally we set it to flat with Z-up Y-forward.') # which dataset to evaluate? choose only one
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--profile', type=int, default=0)

parser.add_argument('--cfg_policy', default='MPVAEPolicy_v0',
                    help='specify the motion model and the policy config.')
parser.add_argument('--project_dir', default='.',
                    help='specify the motion model and the policy config.')
parser.add_argument('--visualize', type=int, default=1)
parser.add_argument('--scene_path', type=str, default='')
parser.add_argument('--sdf_path', type=str, default='')
parser.add_argument('--mesh_path', type=str, default='')
parser.add_argument('--floor_height', type=float, default=0)
parser.add_argument('--use_zero_pose', type=int, default=1)
parser.add_argument('--use_zero_shape', type=int, default=1)
parser.add_argument('--target_body_path', type=str, default=None)
parser.add_argument('--target_point_path', type=str, default='')
parser.add_argument('--start_point_path', type=str, default='')
parser.add_argument('--scene_name', type=str, default='')
parser.add_argument('--interaction_name', type=str, default='')
parser.add_argument('--last_motion_path', type=str, default=None)
parser.add_argument('--history_mode', type=int, default=1)

parser.add_argument('--weight_pene', type=float, default=1)
parser.add_argument('--weight_floor', type=float, default=1)
parser.add_argument('--weight_target_dist', type=float, default=1)
args = parser.parse_args()

"""setup"""
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
torch.set_grad_enabled(False)

"""global parameter"""
random_seed = args.random_seed
# default
# n_gens_1frame = 16  # the number of primitives to generate from a single-frame motion seed
# n_gens_2frame = 4  # the nunber of primitives to generate from a two-frame motion seed
# max_nodes_to_expand = 4  # in the tree search, how many nodes to expand at the same level.
# sample more
# n_gens_1frame = 32     # the number of primitives to generate from a single-frame motion seed
# n_gens_2frame = 8     # the nunber of primitives to generate from a two-frame motion seed
# max_nodes_to_expand = 32 # in the tree search, how many nodes to expand at the same level.
# rely simply on policy
n_gens_1frame = args.num_gen1  # the number of primitives to generate from a single-frame motion seed
n_gens_2frame = args.num_gen2  # the nunber of primitives to generate from a two-frame motion seed
max_nodes_to_expand = args.num_expand  # in the tree search, how many nodes to expand at the same level.
GOAL_THRESH = args.goal_thresh  # the threshold to reach the goal.
HARD_CONTACT=False  # for ranking the primitives in the tree search. If True, then motion primitives with implausible foot-ground contact are discarded.
USE_POLICY_MEAN = False # only use the mean of the policy. If False, random samples are drawn from the policy.
USE_POLICY = args.use_policy # If False, random motion generation will be performed.
SCENE_ORI='ZupYf' # the coordinate setting of the scene.
max_depth = args.max_depth
NUM_SEQ = args.num_sequence # the number of sequences to produce


# cfg_policy = ConfigCreator(args.cfg_policy)
"""get cfg"""
hydra.initialize(version_base=None, config_path=args.cfg_policy)
cfg = cfg_policy = hydra.compose(config_name='config')
# cfg_policy.args.exp_name = args.exp_name
cfg.lossconfig.weight_pene = args.weight_pene
cfg.lossconfig.weight_contact_feet = args.weight_floor
cfg.lossconfig.weight_target_dist = args.weight_target_dist
create_dirs(cfg_policy)
cfg_1frame_male = cfg_policy.trainconfig['cfg_1frame_male']
cfg_2frame_male = cfg_policy.trainconfig['cfg_2frame_male']
cfg_1frame_female = cfg_policy.trainconfig['cfg_1frame_female']
cfg_2frame_female = cfg_policy.trainconfig['cfg_2frame_female']
body_repr = cfg_policy.modelconfig['body_repr']
REPROJ_FACTOR = cfg_policy.modelconfig.get('reproj_factor', 1.0)

"""set GAMMA primitive networks"""
genop_1frame_male = configure_model(cfg_1frame_male, args.gpu_index, args.random_seed)
genop_1frame_female = configure_model(cfg_1frame_female, args.gpu_index, args.random_seed)
genop_2frame_male = configure_model(cfg_2frame_male, args.gpu_index, args.random_seed)
genop_2frame_female = configure_model(cfg_2frame_female, args.gpu_index, args.random_seed)

if USE_POLICY:
    policy_model = GAMMAPolicy(cfg_policy.modelconfig)
    policy_model.eval()
    policy_model.to(device)
    print('policy dir:', cfg_policy.trainconfig['save_dir'])
    ckp_list = sorted(glob.glob(os.path.join(cfg_policy.trainconfig['save_dir'], args.checkpoint)),
                        key=os.path.getmtime)
    if len(ckp_list)>0:
        ckptfile = os.path.join(ckp_list[-1])
        checkpoint = torch.load(ckptfile, map_location=device)
        model_dict = policy_model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        policy_model.load_state_dict(model_dict)
        # policy_model.load_state_dict(checkpoint['model_state_dict'])
        print('[INFO] --load checkpoint from {}'.format(ckptfile))
else:
    policy_model = None


"""body model parsers"""
pconfig_mp = {
    'n_batch':10,
    'device': device,
    'marker_placement': 'ssm2_67'
}
smplxparser_mp = SMPLXParser(pconfig_mp)

pconfig_2frame = {
    'n_batch':2,
    'device': device,
    'marker_placement': 'ssm2_67'
}
smplxparser_2frame = SMPLXParser(pconfig_2frame)

pconfig_1frame = {
    'n_batch':1,
    'device': device,
    'marker_placement': 'ssm2_67'
}
smplxparser_1frame = SMPLXParser(pconfig_1frame)

pconfig_1frame_batch = {
    'n_batch':1 * n_gens_1frame,
    'device': device,
    'marker_placement': 'ssm2_67'
}
smplxparser_1frame_batch = SMPLXParser(pconfig_1frame_batch)

bm_path = config_env.get_body_model_path()
host_name = config_env.get_host_name()
vposer, _ = load_vposer(bm_path + '/vposer_v1_0', vp_model='snapshot')
vposer.eval()
vposer.to(device)
"""data"""
with open(config_env.get_body_marker_path() + '/SSM2.json') as f:
    marker_ssm_67 = json.load(f)['markersets'][0]['indices']
feet_markers = ['RHEE', 'RTOE', 'RRSTBEEF', 'LHEE', 'LTOE', 'LRSTBEEF']
feet_marker_idx = [list(marker_ssm_67.keys()).index(marker_name) for marker_name in feet_markers]

# scene data
Rg = R.from_euler('xyz', np.array([0, 0, 0]), degrees=True)
rotmat_g = Rg.as_matrix()
transl_g = np.array([0, 0, 0])


batch_gen = BatchGeneratorInteractionTest(dataset_path='', body_model_path=bm_path)
# print('floor:', args.floor_height)
data = batch_gen.next_body(visualize=args.visualize, use_zero_pose=args.use_zero_pose, use_zero_shape=args.use_zero_shape,
                            scene_path=Path(args.scene_path), floor_height=args.floor_height,
                           sdf_path=Path(args.sdf_path), mesh_path=Path(args.mesh_path),
                            target_body_path=None if args.target_body_path is None else (args.target_body_path),
                            last_motion_path=None if args.last_motion_path is None else Path(args.last_motion_path),
                            target_point_path=Path(args.target_point_path), start_point_path=Path(args.start_point_path),
                            )
resultdir = 'results/interaction/{}/{}/{}/{}/{}'.format(args.scene_name, args.interaction_name, cfg_policy.cfg_name, cfg_policy.wandb.name, args.gen_name)
idx_seq = 0
while idx_seq < NUM_SEQ:
    outfoldername = '{}/seq{:03d}/'.format(resultdir, idx_seq)
    if not os.path.exists(outfoldername):
        os.makedirs(outfoldername)
    # logger = get_logger(outfoldername, mode='eval')
    print('[INFO] generate sequence {:d}'.format(idx_seq))
    gen_motion(data, max_depth=max_depth, outfoldername=outfoldername)
    idx_seq += 1