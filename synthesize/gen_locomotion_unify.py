import json
import os
import sys
import numpy as np
import torch
import pytorch3d.transforms
import trimesh
from copy import deepcopy
from pathlib import Path
from human_body_prior.tools.model_loader import load_vposer

sys.path.append(os.getcwd())
from synthesize.get_scene import ReplicaScene, replica_folder

unity_to_zup = np.array(
        [[-1, 0, 0, 0],
         [0, 0, -1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]
    )

# use pretrained GAMMA for locomotion
"""
generate sequences of sitting to  close target location using policy, can be combined with search
"""

import numpy as np
import random
import argparse
import os, sys, glob
import copy
import pickle
import pdb
import torch

sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorReachingTarget
from exp_GAMMAPrimitive.utils.environments import BatchGeneratorSceneTrain, BatchGeneratorSceneTest, BatchGeneratorSceneRandomTest
from exp_GAMMAPrimitive.utils import config_env

from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP
from models.models_policy import GAMMAPolicy
from models.baseops import SMPLXParser
from models.searchop import MPTNode, MinHeap, MPTNodeTorch
from models.baseops import get_logger

from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorReplicaSceneNav, get_map

sys.setrecursionlimit(10000)  # if too small, deepcopy will reach the maximal depth limit.



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
    pt_wpath_l_3d = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1), pt_wpath[None, ...] - T0)

    '''extract path feature = normalized direction + unnormalized height'''
    fea_wpathxy = pt_wpath_l_3d[:, :, :2] - pel[:, :, :2]
    fea_wpathxyz = pt_wpath_l_3d[:, :, :3] - pel[:, :, :3]
    dist_xy = torch.norm(fea_wpathxy, dim=-1, keepdim=True).clip(min=1e-12)
    dist_xyz = torch.norm(fea_wpathxyz, dim=-1, keepdim=True).clip(min=1e-12)
    fea_wpathxy = fea_wpathxy / dist_xy
    fea_wpathz = pt_wpath_l_3d[:, :, -1:] - pel[:, :, -1:]
    fea_wpath = torch.cat([fea_wpathxy, fea_wpathz], dim=-1)

    '''extract marker feature'''
    fea_marker = pt_wpath_l_3d[:, :, None, :] - Y_l
    dist_m_3d = torch.norm(fea_marker, dim=-1, keepdim=True).clip(min=1e-12)
    fea_marker_3d_n = (fea_marker / dist_m_3d).reshape(nb, nt, -1)

    '''extract marker feature with depth'''
    dist_m_2d = torch.norm(fea_marker[:, :, :, :2], dim=-1, keepdim=True).clip(min=1e-12)
    fea_marker_xyn = fea_marker[:, :, :, :2] / dist_m_2d
    fea_marker_h = torch.cat([fea_marker_xyn, fea_marker[:, :, :, -1:]], dim=-1).reshape(nb, nt, -1)

    """local map"""
    # t1 = time.time()
    points_local, points_scene, map = get_map(scene['navmesh'],
                          R0, T0,
                          res=cfg_policy.modelconfig.map_res, extent=cfg_policy.modelconfig.map_extent,
                          return_type='torch')
    local_map = map.float()  # [b, res*res]
    local_map[map == False] = -1  #  reassign 0(non-walkable) to -1, https://stats.stackexchange.com/a/138370
    # local_map = torch.zeros_like(local_map)  # test dump map
    # print(time.time() - t1)

    for idx, feature in enumerate((dist_xy, dist_xyz, fea_wpath, fea_marker_3d_n, fea_marker_h, points_local, local_map)):
        if torch.isnan(feature).any() or torch.isinf(feature).any():
            print('feature ', idx, 'is not valid')
            print(feature)

    return dist_xy, dist_xyz, fea_wpath, fea_marker_3d_n, fea_marker_h, points_local, local_map

def update_local_target(marker, pelvis_loc, curr_target_wpath, R0, T0, wpath, scene):
    idx_target_curr, pt_target_curr = curr_target_wpath
    # wpath_ori = scene['wpath_orients_matrix'][idx_target_curr][:3, 2]
    dist_xy, dist_xyz, fea_wpath, fea_marker_3dn, fea_marker_h, points_local, local_map = get_feature(marker, pelvis_loc, R0, T0,
                                                                  wpath[idx_target_curr][None, ...], scene)
    while (dist_xyz < GOAL_THRESH).any() and idx_target_curr < len(wpath) - 1:
        idx_target_curr = idx_target_curr + 1
        # wpath_ori = scene['wpath_orients_matrix'][idx_target_curr][:3, 2]
        dist_xy, dist_xyz, fea_wpath, fea_marker_3dn, fea_marker_h, points_local, local_map = get_feature(marker, pelvis_loc, R0, T0,
                                                                      wpath[idx_target_curr][None, ...], scene)
    return idx_target_curr, fea_wpath, fea_marker_3dn, fea_marker_h, points_local, local_map

def get_rewards(bparams, vertices, joints, Y_l, R0, T0, wpath, scene, points_local, local_map, last_step=False):
    pel_loc = joints[:, :, 0]
    Y_w = torch.einsum('bij,btpj->btpi', R0, Y_l) + T0[:, None, :, :]  # [b, t, p, 3]
    nb, nt = Y_l.shape[:2]
    h = 1 / 40
    Y_l_speed = torch.norm(Y_l[:, 2:] - Y_l[:, :-2], dim=-1) / (2 * h)  # [b, t=9,P=67]
    # Y_l_speed = Y_l_speed.reshape(nb, -1)
    '''evaluate contact soft'''
    # dist2gp = torch.abs(pel_loc[:,-1,-1]-pel_loc[:,0, -1])  # pelvis height not change, not suitable for sitting

    dist2gp = torch.abs(Y_w[:, :, feet_marker_idx, 2].amin(dim=-1) - 0.02).mean(dim=-1)  # feet on floor
    # dist2skat = Y_l_speed.amin(dim=-1)
    dist2skat = (Y_l_speed[:, :, feet_marker_idx].amin(dim=-1) - 0.075).clamp(min=0).mean(dim=-1)
    r_floor = torch.exp(-dist2gp)
    r_skate = torch.exp(-dist2skat)
    # r_contact_friction = r_skate * r_floor

    """penalize static motion"""
    r_nonstatic = (Y_l_speed[:, :, feet_marker_idx].amax(dim=(1, 2)) > 0.1).float()

    """penalize when feet are are too far away from pelvis projection on floor"""
    # feet_markers_xy = Y_l[:, :, feet_marker_idx, :2]
    # box_min = feet_markers_xy.amin(dim=[2])  # [b, t, 2]
    # box_max = feet_markers_xy.amax(dim=[2])
    # extents = box_max - box_min
    # # not too far horizontally and vertically
    # dist2big_stride = (extents[:, :, 0] - 0.4).clamp(min=0).mean(dim=-1) + (extents[:, :, 1] - 1).clamp(min=0).mean(dim=-1)
    # r_stride = torch.exp(-dist2big_stride)

    """penalize too fast motion"""
    # dist2run = (Y_l_speed[:, :, feet_marker_idx].amax(dim=-1) - 2).clamp(min=0).mean(dim=-1)
    # r_not_run =torch.exp(dist2run)

    '''evaluate moving direction'''
    pt_wpath = wpath[-1:]
    target_wpath_l = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1), pt_wpath[None, ...] - T0)[:, :, :3]
    b_dir = pel_loc[:, -1, :3] - pel_loc[:, 0, :3]  # (b,3)
    w_dir = target_wpath_l[:, 0] - pel_loc[:, 0, :3]
    w_dir_n = w_dir / torch.norm(w_dir, dim=-1, keepdim=True).clip(min=1e-12)
    b_dir = b_dir / torch.norm(b_dir,dim=-1, keepdim=True).clip(min=1e-12)
    r_move_toward = (1+torch.einsum('bi,bi->b', w_dir_n, b_dir))/2

    """pose prior, reduce unnatural poses"""
    body_pose = bparams[:, :, 6:69].reshape(nt * nb, -1)
    vp_embedding = vposer.encode(body_pose).loc
    latent_dim = vp_embedding.shape[-1]
    vp_norm = torch.norm(vp_embedding.reshape(nb, nt, -1), dim=-1).mean(dim=1)
    r_vp = torch.exp(-vp_norm / (latent_dim ** 0.5))

    """encourage to face the target"""
    joints_end = joints[:, -1]  # [b,p,3]
    x_axis = joints_end[:, 2, :] - joints_end[:, 1, :]
    x_axis[:, -1] = 0
    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True).clip(min=1e-12)
    z_axis = torch.cuda.FloatTensor([[0, 0, 1]], device=device).repeat(x_axis.shape[0], 1)
    y_axis = torch.cross(z_axis, x_axis)
    b_ori = y_axis[:, :2]  # body forward dir of GAMMA is y axis
    face_target_ori = target_wpath_l[:, 0, :2] - pel_loc[:, -1, :2]
    face_target_ori = face_target_ori / torch.norm(face_target_ori, dim=-1, keepdim=True).clip(min=1e-12)
    r_face_target = (torch.einsum('bi,bi->b', face_target_ori, b_ori) + 1) / 2.0

    """distance to target"""
    dist2targetall = torch.norm(target_wpath_l[:, :, :] - pel_loc[:, :, :], dim=-1, keepdim=False).clip(
        min=1e-12)  # [b, t]
    dist2target = torch.norm(target_wpath_l[:, :, :] - pel_loc[:, :, :], dim=-1, keepdim=False).clip(min=1e-12).amin(
        dim=-1)  # [b,]
    r_target_dist = (1 - ((dist2target - cfg.trainconfig.goal_thresh).clip(min=1e-10)) ** 0.5)
    # r_target_dist = torch.exp(-(dist2target - cfg.trainconfig.goal_thresh).clip(min=0))
    # r_target_dist = dist2targetall[:, 0] - dist2targetall[:, -1]
    r_target = r_target_dist * r_face_target

    """penetration according to map"""
    if args.pene_type == 'foot':
        markers_local_xy = Y_l[:, :, feet_marker_idx, :2]  # [b, t, p, 2]
    elif args.pene_type == 'body':
        markers_local_xy = Y_l[:, :, :, :2]  # [b, t, p, 2]
    # get body bbox on xy plane
    box_min = markers_local_xy.amin(dim=[1, 2]).reshape(nb, 1, 2)
    box_max = markers_local_xy.amax(dim=[1, 2]).reshape(nb, 1, 2)
    inside_feet_box = ((points_local[:, :, :2] >= box_min).all(-1) & (points_local[:, :, :2] <= box_max).all(-1)).float()
    num_pene = inside_feet_box * (1 - local_map) * 0.5 # [b, D]
    num_pene = num_pene.sum(dim=[1]) # [b]
    r_pene = torch.exp(-num_pene)

    """check whether reach the goal within distance and orientation threshold"""
    use_early_stop = cfg.trainconfig.use_early_stop
    terminate = True if use_early_stop and (dist2target < cfg.trainconfig.goal_thresh).any() else False
    r_success = (dist2target < cfg.trainconfig.goal_thresh).float()
    if not last_step and not terminate and cfg.lossconfig.sparse_reward:
        r_target_dist = 0

    reward = r_floor * cfg.lossconfig.weight_floor + \
             r_skate * cfg.lossconfig.weight_skate + \
             r_vp * cfg.lossconfig.weight_vp + \
             r_pene * cfg.lossconfig.weight_pene + \
             r_target_dist * cfg.lossconfig.weight_target_dist + \
             r_face_target * cfg.lossconfig.weight_face_target + \
             r_nonstatic * cfg.lossconfig.weight_nonstatic + \
             r_success * cfg.lossconfig.weight_success

    info = {
        'reward': reward,
        'r_floor': r_floor,
        'r_skate': r_skate,
        'r_vp': r_vp,
        'r_pene': r_pene,
        'r_target_dist': r_target_dist,
        'r_face_target': r_face_target,
        'r_nonstatic': r_nonstatic,
        'r_success': r_success,
        'dist2target': dist2target,
            }

    return reward, terminate, info

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
    idx_target_curr, fea_wpath, fea_marker_3dn, fea_marker_h, points_local, local_map = update_local_target(marker_seed, pelvis_loc,
                                                                               start_node.data['curr_target_wpath'],
                                                                               R0, T0, wpath, scene)

    nb, nt = body_param_seed.shape[:2]
    states = torch.cat([marker_seed.reshape(nb, nt, -1), fea_marker_3dn], dim=-1) if 'condi' in cfg_policy.modelconfig['body_repr'] else marker_seed


    # generate markers and regress to body params
    pred_markers, pred_params, pred_latent = gen_motion_one_step(motion_model, policy_model,
                                                                 states, body_param_seed,
                                                                 prev_betas,
                                                                 t_his=1,
                                                                 local_map=local_map if 'map' in cfg_policy.modelconfig[
                                                                     'body_repr'] else None,
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


        pred_marker_b = REPROJ_FACTOR * pred_markers_proj + (1 - REPROJ_FACTOR) * pred_markers[[ii]]  # [1, t, p=67, 3]
        rootnode = MPTNodeTorch(gender, start_node.data['betas'], R0[[ii]], T0[[ii]], pelvis_loc, pred_joints,
                           pred_markers[ii:ii + 1], pred_markers_proj, pred_params[ii:ii + 1], pred_latent[ii:ii + 1],
                           '1-frame',
                           timestamp=0, curr_target_wpath=(idx_target_curr, wpath[idx_target_curr]))
        rewards, terminate, info = get_rewards(pred_params[[ii]], None, pred_joints,
                                              pred_marker_b, R0[[ii]], T0[[ii]], wpath[[idx_target_curr]], scene, points_local, local_map)
        rootnode.quality = rewards[0].item() + idx_target_curr * 1  # reward for achiveing waypoints
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
            # idx_target_curr, fea_wpath, fea_marker, fea_marker_h, target_ori_local = update_local_target(marker_seed, pel_loc_seed,
            #                                                                            mp_prev.data[
            #                                                                                'curr_target_wpath'],
            #                                                                            R0, T0, wpath, scene)
            # states = torch.cuda.FloatTensor(np.concatenate([marker_seed, fea_marker], axis=-1),
            #                                     device=device)[None, ...]
            idx_target_curr, fea_wpath, fea_marker_3dn, fea_marker_h, points_local, local_map = update_local_target(
                marker_seed, pel_loc_seed,
                mp_prev.data['curr_target_wpath'],
                R0, T0, wpath, scene)

            states = torch.cat([marker_seed.reshape(nb, nt, -1), fea_marker_3dn], dim=-1) if 'condi' in \
                                                                                             cfg_policy.modelconfig[
                                                                                                 'body_repr'] else marker_seed

            '''generate future motions'''
            motion_model = genop_2frame_male if prev_gender == 'male' else genop_2frame_female

            prev_betas_torch = prev_betas.unsqueeze(0)

            pred_markers, pred_params, pred_latent = gen_motion_one_step(motion_model, policy_model,
                                states, body_param_seed,
                                prev_betas_torch,
                                t_his=t_his,
                                local_map=local_map if 'map' in cfg_policy.modelconfig[
                                    'body_repr'] else None,
                                )  # smplx [n_gens, n_frames, d]
            if args.project_floor:
                lowest = pred_markers[:, :, :, 2].amin(dim=[1, 2]) + T0[0, 0, 2]
                pred_markers = pred_markers - lowest.reshape(-1, 1, 1, 1)
                pred_params[:, :, 2] = pred_params[:, :, 2] - lowest.reshape(-1, 1)

            '''sort generated primitives'''
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
                mp_curr = MPTNodeTorch(prev_gender, prev_betas, R0, T0, pelvis_loc, pred_joints,
                                  pred_markers[ii:ii + 1], pred_markers_proj, pred_params[ii:ii + 1],
                                  pred_latent[ii:ii + 1], '2-frame',
                                  timestamp=iop, curr_target_wpath=(idx_target_curr, wpath[idx_target_curr]))
                pred_marker_b = REPROJ_FACTOR * pred_markers_proj + (1 - REPROJ_FACTOR) * pred_markers[
                    [ii]]  # [1, t, p=67, 3]
                rewards, terminate, info = get_rewards(pred_params[[ii]], None, pred_joints,
                                                 pred_marker_b, R0, T0, wpath[[idx_target_curr]], scene, points_local,
                                                 local_map)
                mp_curr.quality = rewards[0].item() + idx_target_curr * 1  # reward for achiveing waypoints
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
                                  last_mp['transf_rotmat'], last_mp['transf_transl'], last_mp['pelvis_loc'],
                                  last_mp['joints'],
                                  last_mp['markers'], last_mp['markers_proj'], last_mp['smplx_params'],
                                  last_mp['mp_latent'], '2-frame',
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
        output['navmesh_path'] = body_s['navmesh_path']
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

parser.add_argument('--cfg_policy', default='MPVAEPolicy_v0',
                    help='specify the motion model and the policy config.')
parser.add_argument('--project_dir', default='.',
                    help='specify the motion model and the policy config.')
parser.add_argument('--visualize', type=int, default=1, help='switch on/off of visualization')
parser.add_argument('--random_orient', type=int, default=0, help='1: initialize body orientation randomly, 0: make initial body face the first waypoint')
parser.add_argument('--use_zero_pose', type=int, default=1, help='0: initialize body pose randomly, 1: initialize body pose with zero pose of VPoser')
parser.add_argument('--use_zero_shape', type=int, default=1, help='0: initialize body shape randomly, 1: use zero body shape of SMPL-X')
parser.add_argument('--clip_far', type=int, default=0, help='if set 1, linearly interpolate two waypoints that are too far away')
parser.add_argument('--scene_path', type=str, default='', help='scene mesh path')
parser.add_argument('--navmesh_path', type=str, default='', help='(tight) navigation mesh path')
parser.add_argument('--floor_height', type=float, default=0)
parser.add_argument('--wpath_path', type=str, default='', help='waypoints file path')
parser.add_argument('--scene_name', type=str, default='')
parser.add_argument('--path_name', type=str, default='')
parser.add_argument('--last_motion_path', type=str, default=None, help='file path of the generated last stage motion')
parser.add_argument('--history_mode', type=int, default=1, help='use 1 frame or 2 frames of last stage motion to initialize current motion generation')
parser.add_argument('--project_floor', type=int, default=0)
parser.add_argument('--pene_type', type=str, default='foot', help='foot or body, detect human-scene penetration considering only feet or full body')

parser.add_argument('--weight_pene', type=float, default=1)
parser.add_argument('--weight_floor', type=float, default=1)

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
cfg.lossconfig.weight_floor = args.weight_floor
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

# scene_path = replica_folder / 'room_0' / 'mesh.ply'
# scene = ReplicaScene('room_0', replica_folder, build=False)
# visualize
# scene_mesh = scene.mesh
# for way_point in way_points:
#     sphere = trimesh.primitives.Sphere(center=way_point, radius=0.1)
#     sphere.visual.vertex_colors = np.array([255, 0, 0, 255])
#     scene_mesh = scene_mesh + sphere
# scene_mesh.show()
"""test in replica"""
# path_name_list = ['path_behind']
# pelvis_file_list = ['pelvis.json']
# bm_path = config_env.get_body_model_path()
# batch_gen = BatchGeneratorReplicaSceneNav(dataset_path=None,
#                                           scene=ReplicaScene(scene_name='room_0', replica_folder=replica_folder, zero_floor=False),
#                                           path_name_list=path_name_list,
#                                           target_pelvis_file_list=pelvis_file_list,
#                                          body_model_path=bm_path)
"""main block for motion generation"""
# for path_name in path_name_list:
#     data = batch_gen.next_body(use_zero_pose=True, interpolate_path=True, to_numpy=False)
#     resultdir = 'results/locomotion/room_0/{}/{}/{}/{}'.format(cfg_policy.cfg_name, cfg_policy.wandb.name, path_name, args.gen_name)
#     idx_seq = 0
#     while idx_seq < NUM_SEQ:
#         outfoldername = '{}/randseed{:03d}_seq{:03d}/'.format(resultdir, random_seed, idx_seq)
#         if not os.path.exists(outfoldername):
#             os.makedirs(outfoldername)
#         # logger = get_logger(outfoldername, mode='eval')
#         print('[INFO] generate sequence {:d}'.format(idx_seq))
#         gen_motion(data, max_depth=max_depth)
#         idx_seq += 1


# scene_dir = 'data/scenes'
# scene_list_path = Path('data/scenes/random_scene_names.pkl')+
# with open(scene_list_path, 'rb') as f:
#     scene_list = pickle.load(f)
# batch_gen = BatchGeneratorSceneTrain(dataset_path='',
#                                         scene_dir=scene_dir,
#                                         scene_list=scene_list,
#                                         scene_type='random',
#                                         body_model_path=bm_path)
# start_target = np.array([
#     [1.05, -0.58, 0],
#     [2.97, -2.35, 0]
# ])
# path_name = 'along_sofa'
# data = batch_gen.next_body(use_zero_pose=True, visualize=False, scene_idx=0, start_target=start_target,
#                          res=cfg.modelconfig.map_res, extent=cfg.modelconfig.map_extent,)
# gen_locomotion()
#
# start_target = np.array([
#     [-0.42, -4.33, 0],
#     [0.02, -1.02, 0]
# ])
# path_name = 'narrow'
# data = batch_gen.next_body(use_zero_pose=True, visualize=False, scene_idx=0, start_target=start_target,
#                          res=cfg.modelconfig.map_res, extent=cfg.modelconfig.map_extent,)
# gen_locomotion()
#
# start_target = np.array([
#     [2.68, -4.91, 0],
#     [-0.28, -5.00, 0]
# ])
# path_name = 'corner'
# data = batch_gen.next_body(use_zero_pose=True, visualize=False, scene_idx=0, start_target=start_target,
#                          res=cfg.modelconfig.map_res, extent=cfg.modelconfig.map_extent,)
# gen_locomotion()


"""random scene test"""
# scene_dir = 'data/scenes'
# scene_list_path = Path('data/scenes/random_scene_names.pkl')
# with open(scene_list_path, 'rb') as f:
#     scene_list = pickle.load(f)
# batch_gen = BatchGeneratorSceneRandomTest(dataset_path='',
#                                         scene_dir=scene_dir,
#                                         scene_list=scene_list,
#                                         scene_type='random',
#                                         body_model_path=bm_path)
# for path_idx in range(10):
#     data = batch_gen.next_body(use_zero_pose=True, visualize=False, scene_idx=0, path_idx=path_idx,
#                              res=cfg.modelconfig.map_res, extent=cfg.modelconfig.map_extent,)
#     path_name = data['path_name']
#     gen_locomotion()

def gen_locomotion():
    resultdir = 'results/locomotion/{}/{}/{}/{}/{}'.format(args.scene_name, args.path_name, cfg_policy.cfg_name, cfg_policy.wandb.name, args.gen_name)
    idx_seq = 0
    while idx_seq < NUM_SEQ:
        outfoldername = '{}/seq{:03d}/'.format(resultdir, idx_seq)
        if not os.path.exists(outfoldername):
            os.makedirs(outfoldername)
        # logger = get_logger(outfoldername, mode='eval')
        print('[INFO] generate sequence {:d}'.format(idx_seq))
        gen_motion(data, max_depth=max_depth, outfoldername=outfoldername)
        idx_seq += 1

batch_gen = BatchGeneratorSceneTest(dataset_path='', body_model_path=bm_path)
# print('floor:', args.floor_height)
data = batch_gen.next_body(visualize=args.visualize, use_zero_pose=args.use_zero_pose, use_zero_shape=args.use_zero_shape,
                            scene_path=Path(args.scene_path), floor_height=args.floor_height, navmesh_path=Path(args.navmesh_path),
                            wpath_path=Path(args.wpath_path), path_name=args.path_name,
                            last_motion_path=None if args.last_motion_path is None else Path(args.last_motion_path),
                            clip_far=args.clip_far, random_orient=args.random_orient,
                            res=cfg.modelconfig.map_res, extent=cfg.modelconfig.map_extent,)
gen_locomotion()