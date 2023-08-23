"""Information
- This script is to generate long-term motions in the scene with cubes.
- By setting the flags, motion is generated either by the policy or randomly.
- The tree-based search is performed to rank all generated motion primitives.
    It does not run in parallel. See the *_parallel.py version
- Variables are mainly in numpy.array
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
from coap import attach_coap
from utils.config_env import get_body_model_path

sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorSceneNav
from exp_GAMMAPrimitive.utils import config_env

from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP
from models.models_policy import GAMMAPolicy
from models.baseops import SMPLXParser
from models.searchop import MPTNode, MinHeap, calc_sdf
from models.baseops import get_logger

sys.setrecursionlimit(10000)  # if too small, deepcopy will reach the maximal depth limit.


def log_and_print(logstr):
    logger.info(logstr)
    if args.verbose:
        print(logstr)


def gen_motion_one_step(motion_model, states, bparam_seed, prev_betas, t_his):
    if t_his == 1:
        n_gens = n_gens_1frame
    elif t_his == 2:
        n_gens = n_gens_2frame
    [pred_markers, pred_params,
     act, act_log_prob,
     value] = motion_model.generate_ppo(policy_model,
                                        states.permute([1, 0, 2]),
                                        bparam_seed.transpose([1, 0, 2]),
                                        prev_betas,
                                        n_gens=n_gens,
                                        param_blending=True,
                                        use_policy_mean=USE_POLICY_MEAN,
                                        use_policy=USE_POLICY
                                        )
    pred_markers = np.reshape(pred_markers, (pred_markers.shape[0],
                                             pred_markers.shape[1], -1, 3))  # [t, n_gens, V, 3]
    return pred_markers.transpose([1, 0, 2, 3]), pred_params.transpose([1, 0, 2]), act

"""generate on motion primitive with given latent z"""
def gen_motion_one_step_optim(motion_model, marker_seed, bparam_seed, prev_betas, t_his=-1, n_gens=1, z=None):
    [pred_markers,
     pred_params] = motion_model.generate(
                                        marker_seed.permute([1, 0, 2]),
                                        bparam_seed.permute([1, 0, 2]),
                                        prev_betas,
                                        n_gens=n_gens,
                                        to_numpy=False,
                                        param_blending=True,
                                        t_his=t_his,
                                        z=z,
                                        )
    pred_markers = pred_markers.reshape(pred_markers.shape[0],
                                        pred_markers.shape[1], -1, 3)  # [n_gens, t, V, 3]
    # pred_markers, pred_params = pred_markers.permute([1, 0, 2, 3]), pred_params.permute([1, 0, 2])
    return pred_markers, pred_params

def canonicalize_static_pose(data):
    smplx_transl = data['transl']
    smplx_glorot = data['global_orient']
    smplx_poses = data['body_pose']
    gender = data['gender']
    betas = data['betas']
    smplx_handposes = np.zeros([smplx_transl.shape[0], 24])
    prev_params = np.concatenate([smplx_transl, smplx_glorot,
                                  smplx_poses, smplx_handposes],
                                 axis=-1)  # [t,d]
    body_param_seed = prev_params[:1]

    ## move frame to the body's pelvis
    R0, T0 = smplxparser_1frame.get_new_coordinate(
        betas=betas,
        gender=gender,
        xb=body_param_seed)

    ## get the last body param and marker in the new coordinate
    body_param_seed = smplxparser_1frame.update_transl_glorot(R0, T0,
                                                              betas=betas,
                                                              gender=gender,
                                                              xb=body_param_seed)
    marker_seed = smplxparser_1frame.get_markers(
        betas=betas,
        gender=gender,
        xb=body_param_seed)
    pelvis_loc = smplxparser_1frame.get_jts(betas=betas,
                                            gender=gender,
                                            xb=body_param_seed)[:, 0]  # [t, 3]
    return marker_seed, body_param_seed[None, ...], betas, R0, T0, pelvis_loc


def get_wpath_feature(Y_l, pel, R0, T0, pt_wpath):
    '''
    --Y_l = [t,d] local marker
    --pel = [t,d]
    --R0 = [1,3,3] rotation matrix
    --T0 = [1,1,3] translation
    --pt_wpath = [1,d]
    '''
    nt = pel.shape[0]
    Y_l = Y_l.reshape(nt, -1, 3)
    # print(R0.shape, pt_wpath.shape, T0.shape)
    pt_wpath_l_3d = np.einsum('ij,tj->ti', R0[0].T, pt_wpath - T0[0])

    '''extract path feature = normalized direction + unnormalized height'''
    fea_wpathxy = pt_wpath_l_3d[:, :2] - pel[:, :2]
    dist_xy = np.linalg.norm(fea_wpathxy, axis=-1, keepdims=True)
    fea_wpathxy = fea_wpathxy / dist_xy
    fea_wpathz = pt_wpath_l_3d[:, -1:] - pel[:, -1:]
    fea_wpath = np.concatenate([fea_wpathxy, fea_wpathz], axis=-1)

    '''extract marker feature'''
    fea_marker = pt_wpath_l_3d[:, None, :] - Y_l
    dist_m_3d = np.linalg.norm(fea_marker, axis=-1, keepdims=True)
    fea_marker_3dn = (fea_marker / dist_m_3d).reshape(nt, -1)

    '''extract marker feature with depth'''
    fea_marker_xy = fea_marker[:, :, :2]
    dist_m_2d = np.linalg.norm(fea_marker_xy, axis=-1, keepdims=True)
    fea_marker_xy = fea_marker_xy / dist_m_2d
    fea_marker_h = np.concatenate([fea_marker_xy, fea_marker[:, :, -1:]], axis=-1).reshape(nt, -1)

    return dist_xy, fea_wpath, fea_marker_3dn, fea_marker_h


def update_local_target(marker, pelvis_loc, curr_target_wpath, R0, T0, wpath):
    idx_target_curr, pt_target_curr = curr_target_wpath
    dist, fea_wpath, fea_marker, fea_marker_h = get_wpath_feature(marker, pelvis_loc, R0, T0,
                                                                  wpath[idx_target_curr][None, ...])
    while np.any(dist < GOAL_THRESH) and idx_target_curr < len(wpath) - 1:
        idx_target_curr = idx_target_curr + 1
        # print('reached one point')
        dist, fea_wpath, fea_marker, fea_marker_h = get_wpath_feature(marker, pelvis_loc, R0, T0,
                                                                      wpath[idx_target_curr][None, ...])
    return idx_target_curr, fea_wpath, fea_marker, fea_marker_h


def gen_tree_roots(start_node, wpath, scene=None):
    mp_heap = MinHeap()

    # canonicalize the starting pose
    marker_seed = start_node.data['markers']
    body_param_seed = start_node.data['smplx_params']
    R0 = start_node.data['transf_rotmat']
    T0 = start_node.data['transf_transl']
    motion_model = genop_1frame_male
    # print(start_node.data['betas'])
    prev_betas = torch.FloatTensor(start_node.data['betas']).unsqueeze(0).to(device)
    gender = str(start_node.data['gender'])
    pelvis_loc = start_node.data['pelvis_loc']

    ## retrieve current target and update it
    idx_target_curr, fea_wpath, fea_marker, fea_marker_h = update_local_target(marker_seed, pelvis_loc,
                                                                               start_node.data['curr_target_wpath'],
                                                                               R0, T0, wpath)
    if body_repr == 'ssm2_67_condi_marker':
        states = torch.cuda.FloatTensor(np.concatenate([marker_seed.reshape([1, -1]), fea_marker], axis=-1),
                                        device=device)[None, ...]
    else:
        raise NotImplementedError

    # generate markers and regress to body params
    pred_markers, pred_params, pred_latent = gen_motion_one_step(motion_model,
                                                                 states, body_param_seed, prev_betas, t_his=1)
    for ii in range(pred_markers.shape[0]):
        joints = smplxparser_mp.get_jts(betas=start_node.data['betas'],
                                        gender=gender,
                                        xb=pred_params[ii])
        pelvis_loc = joints[:, 0]  # [t, 3]
        pred_markers_proj = smplxparser_mp.get_markers(betas=start_node.data['betas'],
                                                       gender=gender,
                                                       xb=pred_params[ii]).reshape(
            (pred_params.shape[1], -1, 3))  # [t, p, 3]
        rootnode = MPTNode(gender, start_node.data['betas'], R0, T0, pelvis_loc, joints,
                           pred_markers[ii:ii + 1], pred_markers_proj, pred_params[ii:ii + 1], pred_latent[ii:ii + 1],
                           '1-frame',
                           timestamp=0, curr_target_wpath=(idx_target_curr, wpath[idx_target_curr]))
        rootnode.evaluate_quality_collision(terrian_rotmat=rotmat_g, wpath=wpath,
                                            # obj_transform=scene['obj_transform'],
                                            # obj_sdf=scene['obj_sdf'],
                                            obj_points=scene['obj_points'],
                                            scene=scene,
                                            collision_mode=args.use_collision,
                                            coap_model=coap_model_10,
                                            weight_target=args.weight_target,
                                            weight_ori=args.weight_ori,
                                            weight_pene=args.weight_pene)
        if rootnode.quality != 0:
            mp_heap.push(rootnode)
    return mp_heap

from tqdm import tqdm
def expand_tree(mp_heap_prev, wpath, max_depth=10, scene=None):
    mp_heap_curr = MinHeap()
    # generate child treenodes
    for iop in tqdm(range(0, max_depth)):
        log_and_print('[INFO] at level {}'.format(iop))
        idx_node = 0
        while (not mp_heap_prev.is_empty()) and (idx_node < max_nodes_to_expand):
            mp_prev = mp_heap_prev.pop()
            if mp_prev.quality == 0:
                continue
            idx_node += 1

            '''produce marker seed'''
            t_his = 2
            prev_params = copy.deepcopy(mp_prev.data['smplx_params'])
            prev_markers = copy.deepcopy(mp_prev.data['markers'])
            prev_pel_loc = copy.deepcopy(mp_prev.data['pelvis_loc'])
            prev_betas = mp_prev.data['betas']
            prev_gender = mp_prev.data['gender']
            prev_rotmat = copy.deepcopy(mp_prev.data['transf_rotmat'])
            prev_transl = copy.deepcopy(mp_prev.data['transf_transl'])
            body_param_seed = prev_params[0, -t_his:]
            ## move frame to the second last body's pelvis
            R_, T_ = smplxparser_1frame.get_new_coordinate(
                betas=prev_betas,
                gender=prev_gender,
                xb=body_param_seed[:1])
            T0 = np.einsum('bij,bpj->bpi', prev_rotmat, T_) + prev_transl
            R0 = np.einsum('bij,bjk->bik', prev_rotmat, R_)
            ## get the last body param and marker in the new coordinate
            body_param_seed = smplxparser_2frame.update_transl_glorot(
                np.tile(R_, (t_his, 1, 1)),
                np.tile(T_, (t_his, 1, 1)),
                betas=prev_betas,
                gender=prev_gender,
                xb=body_param_seed)

            ## blend predicted markers and the reprojected markers to eliminated jitering
            marker_seed_rproj = smplxparser_2frame.get_markers(
                betas=prev_betas,
                gender=prev_gender,
                xb=body_param_seed).reshape([t_his, -1])
            pred_markers = np.einsum('ij, tnj->tni', R_[0].T, prev_markers[0, -t_his:] - T_).reshape([t_his, -1])
            marker_seed = REPROJ_FACTOR * marker_seed_rproj + (1 - REPROJ_FACTOR) * pred_markers
            pel_loc_seed = np.einsum('ij,tj->ti', R_[0].T, prev_pel_loc[-t_his:] - T_[0])
            idx_target_curr, fea_wpath, fea_marker, fea_marker_h = update_local_target(marker_seed, pel_loc_seed,
                                                                                       mp_prev.data[
                                                                                           'curr_target_wpath'],
                                                                                       R0, T0, wpath)
            if body_repr == 'ssm2_67_condi_marker':
                states = torch.cuda.FloatTensor(np.concatenate([marker_seed, fea_marker], axis=-1),
                                                device=device)[None, ...]
            else:
                raise NotImplementedError

            '''generate future motions'''
            motion_model = genop_2frame_male if prev_gender == 'male' else genop_2frame_female
            body_param_seed = body_param_seed[None, ...]
            prev_betas_torch = torch.FloatTensor(prev_betas).unsqueeze(0).to(device)

            pred_markers, pred_params, pred_latent = gen_motion_one_step(motion_model, states,
                                                                         body_param_seed,
                                                                         prev_betas_torch,
                                                                         t_his)  # smplx [n_gens, n_frames, d]

            '''sort generated primitives'''
            for ii in range(pred_markers.shape[0]):
                joints = smplxparser_mp.get_jts(betas=prev_betas,
                                                gender=prev_gender,
                                                xb=pred_params[ii])
                pelvis_loc = joints[:, 0]  # [t, 3]
                pred_markers_proj = smplxparser_mp.get_markers(betas=prev_betas,
                                                               gender=prev_gender,
                                                               xb=pred_params[ii]).reshape(
                    (pred_params.shape[1], -1, 3))  # [t, p, 3]
                mp_curr = MPTNode(prev_gender, prev_betas, R0, T0, pelvis_loc, joints,
                                  pred_markers[ii:ii + 1], pred_markers_proj, pred_params[ii:ii + 1],
                                  pred_latent[ii:ii + 1], '2-frame',
                                  timestamp=iop, curr_target_wpath=(idx_target_curr, wpath[idx_target_curr]))
                mp_curr.evaluate_quality_collision(terrian_rotmat=rotmat_g, wpath=wpath,
                                                    # obj_transform=scene['obj_transform'],
                                                    # obj_sdf=scene['obj_sdf'],
                                                   obj_points=scene['obj_points'],
                                                   scene=scene,
                                                   collision_mode=args.use_collision,
                                                   coap_model=coap_model_10,
                                                   weight_target=args.weight_target,
                                                   weight_ori=args.weight_ori,
                                                   weight_pene=args.weight_pene
                                                   )
                if mp_curr.quality != 0:
                    mp_curr.set_parent(mp_prev)
                    mp_heap_curr.push(mp_curr)

            ## if all children is 0 quality with 2frame model, switch to 1 frame model and continue
            if mp_heap_curr.len() == 0:
                motion_model = genop_1frame_male if prev_gender == 'male' else genop_1frame_female
                states = states[:, :1]
                body_param_seed = body_param_seed[:, :1]
                pred_markers, pred_params, pred_latent = gen_motion_one_step(motion_model, states,
                                                                             body_param_seed,
                                                                             prev_betas_torch,
                                                                             t_his - 1)  # smplx [n_gens, n_frames, d]
                for ii in range(pred_markers.shape[0]):
                    pelvis_loc = smplxparser_mp.get_jts(betas=prev_betas,
                                                        gender=prev_gender,
                                                        data=pred_params[ii])[:, 0]  # [t, 3]
                    pred_markers_proj = smplxparser_mp.get_markers(betas=prev_betas,
                                                                   gender=prev_gender,
                                                                   params=pred_params[ii]).reshape(
                        (pred_params.shape[1], -1, 3))  # [t, p, 3]
                    mp_curr = MPTNode(prev_gender, prev_betas, R0, T0, pelvis_loc,
                                      pred_markers[ii:ii + 1], pred_markers_proj, pred_params[ii:ii + 1],
                                      pred_latent[ii:ii + 1], '1-frame',
                                      timestamp=iop, curr_target_wpath=(idx_target_curr, wpath[idx_target_curr]))
                    mp_curr.evaluate_quality_collision(terrian_rotmat=rotmat_g, wpath=wpath,
                                                       # obj_transform=scene['obj_transform'],
                                                       # obj_sdf=scene['obj_sdf'],
                                                       obj_points=scene['obj_points'],
                                                       scene=scene,
                                                       collision_mode=args.use_collision,
                                                       coap_model=coap_model_10,
                                                       weight_target=args.weight_target,
                                                       weight_ori=args.weight_ori,
                                                       weight_pene=args.weight_pene
                                                       )
                    if mp_curr.quality != 0:
                        mp_curr.set_parent(mp_prev)
                        mp_heap_curr.push(mp_curr)

        if mp_heap_curr.len() == 0:
            log_and_print('[INFO] |--no movements searched. Program terminates.')
            return None
            # sys.exit()
        log_and_print(
            '[INFO] |--valid MPs={}, dist_to_target={:.2f}, path_finished={}/{}, dist_to_target_curr={:.2f}, dist_to_ori_curr={:.2f}, dist_to_gp={:.2f}, dist_to_skat={:.2f}, pene_value={:.4f}'.format(
                mp_heap_curr.len(),
                mp_heap_curr.data[0].dist2target,
                mp_heap_curr.data[0].data['curr_target_wpath'][0], len(wpath),
                mp_heap_curr.data[0].dist2target_curr,
                mp_heap_curr.data[0].dist2ori_curr,
                mp_heap_curr.data[0].dist2g, mp_heap_curr.data[0].dist2skat,
                mp_heap_curr.data[0].pene_value))
        mp_heap_prev.clear()
        mp_heap_prev = copy.deepcopy(mp_heap_curr)
        mp_heap_curr.clear()
        ## if the best node is close enough to the target, search stops and return
        # if np.abs(mp_heap_prev.data[0].dist2target) < GOAL_THRESH:
        #     log_and_print('[INFO] |--find satisfactory solutions. Search finishes.')
        #     return mp_heap_prev

    return mp_heap_prev

def gen_tree_roots_optim(data_mp, latent, wpath, scene=None):
    # canonicalize the starting pose
    marker_seed = torch.FloatTensor(data_mp['markers']).to(device)
    body_param_seed = torch.FloatTensor(data_mp['smplx_params']).to(device)
    R0 = torch.FloatTensor(data_mp['transf_rotmat']).to(device)
    T0 = torch.FloatTensor(data_mp['transf_transl']).to(device)
    # print(data_mp['betas'])
    prev_betas = torch.FloatTensor(data_mp['betas']).unsqueeze(0).to(device)
    prev_gender = str(data_mp['gender'])
    motion_model = genop_1frame_male
    pelvis_loc = data_mp['pelvis_loc']

    # generate markers and regress to body params
    nb = 1
    t_his = 1
    # print(marker_seed.shape, body_param_seed.shape, latent.shape)
    marker_seed = marker_seed.reshape(nb, t_his, -1)  # [b,t,d]
    body_param_seed = body_param_seed.reshape(nb, t_his, -1)
    pred_markers, pred_params = gen_motion_one_step_optim(motion_model,
                                                          marker_seed, body_param_seed,
                                                          prev_betas, t_his=1,
                                                          z=latent)
    nb, nt = pred_params.shape[:2]
    pred_joints = smplxparser_mp.get_jts(betas=prev_betas,
                                         gender=prev_gender,
                                         xb=pred_params.reshape([nb * nt, -1]),
                                         to_numpy=False).reshape([nb, nt, -1, 3])  # [t*b, 3]
    pred_pelvis_loc = pred_joints[:, :, 0]
    pred_markers_proj = smplxparser_mp.get_markers(betas=prev_betas,
                                                   gender=prev_gender,
                                                   xb=pred_params.reshape([nb * nt, -1]),
                                                   to_numpy=False).reshape((nb, nt, -1, 3))  # [b, t, p, 3]

    """trim the tree and keep the best K=1 motion"""
    pred_marker_b = REPROJ_FACTOR * pred_markers_proj + (1 - REPROJ_FACTOR) * pred_markers
    return [pred_marker_b, pred_params, prev_betas, prev_gender, R0, T0, pred_pelvis_loc, pred_joints, '1-frame'], 0


def expand_tree_optim(data_mp, latent, wpath, max_depth=10, scene=None):
    prev_markers_b, prev_params, prev_betas, prev_gender, prev_rotmat, prev_transl, prev_pelvis_loc, prev_joints, _ = data_mp

    '''produce marker seed'''
    t_his = 2
    body_param_seed = prev_params[:, -t_his:]  # [b,t,d], need deepcopy, body_param will be changed inplace
    nb, nt = body_param_seed.shape[:2]
    ## move frame to the second last body's pelvis
    assert nb == 1
    R_, T_ = smplxparser_1frame.get_new_coordinate(
        betas=prev_betas,
        gender=prev_gender,
        xb=body_param_seed[:, 0],
        to_numpy=False)  # [b,3,3][b,1,3]
    T0 = torch.einsum('bij,btj->bti', prev_rotmat, T_) + prev_transl
    R0 = torch.einsum('bij,bjk->bik', prev_rotmat, R_)

    marker_seed = torch.einsum('bij,btpj->btpi', R_.permute(0, 2, 1), prev_markers_b[:, -t_his:] - T_[..., None, :])
    # pel_loc_seed = torch.einsum('bij,btj->bti', R_.permute(0, 2, 1), prev_pelvis_loc[:, -t_his:] - T_)

    # original update_transl_glorot changes xb inplace !!
    body_param_seed = smplxparser_2frame.update_transl_glorot(
        R_.expand(t_his, -1, -1),
        T_.expand(t_his, -1, -1),
        betas=prev_betas,
        gender=prev_gender,
        to_numpy=False,
        xb=body_param_seed.reshape(nb * nt, -1),
        inplace=False
    ).reshape(nb, nt, -1)

    '''generate future motions'''
    # motion_model = genop_2frame_male
    marker_seed = marker_seed.reshape(nb, t_his, -1)  # [b,t,d]

    [pred_markers, pred_params] = gen_motion_one_step_optim(genop_2frame_male,
                                                      marker_seed, body_param_seed, prev_betas,
                                                      t_his=2, n_gens=1,
                                                      z=latent)

    nb, nt = pred_params.shape[:2]
    pred_joints = smplxparser_mp.get_jts(betas=prev_betas,
                                         gender=prev_gender,
                                         xb=pred_params.reshape([nb * nt, -1]),
                                         to_numpy=False).reshape([nb, nt, -1, 3])  # [t*b, 3]
    pred_pelvis_loc = pred_joints[:, :, 0]
    pred_markers_proj = smplxparser_mp.get_markers(betas=prev_betas,
                                                   gender=prev_gender,
                                                   xb=pred_params.reshape([nb * nt, -1]),
                                                   to_numpy=False).reshape((nb, nt, -1, 3))  # [b, t, p, 3]

    """trim the tree and keep the best K=1 motion"""
    pred_marker_b = REPROJ_FACTOR * pred_markers_proj + (1 - REPROJ_FACTOR) * pred_markers
    return [pred_marker_b, pred_params, prev_betas, prev_gender, R0, T0, pred_pelvis_loc, pred_joints, '2-frame'], 0

def get_cost_search(bparams, betas, joints, Y_l, R0, T0, body_scene):
    R0 = R0.repeat(joints.shape[0], 1, 1)
    T0 = T0.repeat(joints.shape[0], 1, 1)
    pel_loc = joints[:, :, 0]
    nb, nt = Y_l.shape[:2]
    Y_w = torch.einsum('bij,btpj->btpi', R0, Y_l) + T0[:, None, ...]
    # ----select motion index of proper contact with the ground
    Y_wz = Y_w[:, :, :, -1]  # [b,t, P]
    Y_wz = Y_wz.reshape(nb, -1)
    h = 1 / 40
    Y_w_speed = torch.norm(Y_w[:, 2:] - Y_w[:, :-2], dim=-1) / (2 * h)  # [b, t=9,P=67]
    Y_w_speed = Y_w_speed.reshape(nb, -1)
    '''evaluate contact soft'''
    dist2gp = (torch.abs(Y_wz.min(dim=-1)[0]) - 0.05).clamp(min=0)
    dist2skat = (torch.abs(Y_w_speed.min(dim=-1)[0]) - 0.075).clamp(min=0)

    # dist2target = torch.mean(torch.norm(goal['markers'][None, :, :] - Y_w[:, -1], dim=-1), dim=-1)  # [b, m, 3] -> [b,m] -> [b]
    # pelvis as goal
    pel_world = torch.einsum('bij,btj->bti', R0, pel_loc) + T0
    target = torch.tensor(body_scene['wpath'][None, 1:, :], dtype=torch.float32, device=joints.device)
    dist2target = torch.mean(torch.norm(target - pel_world, dim=-1), dim=-1)
    # markers as goal
    # target = torch.tensor(body_scene['markers'][None, 1:, :, :], dtype=torch.float32, device=joints.device)
    # dist2target = torch.mean(torch.norm(target - Y_w, dim=-1), dim=-1)

    # penetration
    if args.use_collision == 1:  # use sdf grid
        sdf_values = calc_sdf(Y_w.reshape(nb * nt, -1, 3), body_scene['obj_sdf'])
        loss_pene = torch.tensor(0.0, dtype=torch.float32, device=joints.device) if sdf_values.lt(0.0).sum().item() < 1 else torch.mean(
            sdf_values[sdf_values < 0].abs())
    elif args.use_collision == 2:  # use coap
        assert nb * nt == 1 * 10
        bparams = bparams.reshape(nb * nt, -1)
        scene_points = torch.FloatTensor(body_scene['obj_points']).to(joints.device).unsqueeze(0)  # [1, p, 3]
        scene_points = torch.einsum('bij,bpj->bpi', R0.permute(0, 2, 1), scene_points - T0)
        smpl_output = coap_model(transl=bparams[-1:, :3], global_orient=bparams[-1:, 3:6],
                                 body_pose=bparams[-1:, 6:69],
                                 left_hand_pose=bparams[-1:, 69:81], right_hand_pose=bparams[-1:, 81:93],
                                 betas=betas.expand(1, -1),
                                 return_verts=True, return_full_pose=True)
        loss_pene, _collision_mask = coap_model.coap.collision_loss(scene_points, smpl_output, ret_collision_mask=True)
    else:
        loss_pene = torch.tensor(0.0, dtype=torch.float32, device=joints.device)

    return dist2gp, dist2skat, dist2target, loss_pene

def calc_loss(outmps, body_scene, latent_codes):
    # pred_marker_b, pred_params, prev_betas, prev_gender, R0, T0, pred_pelvis_loc, pred_joints, _ = outmps[-1]
    # loss = get_cost_search(pred_params, pred_joints, pred_marker_b, R0, T0, goal)
    # return loss
    loss_skate_list = []
    loss_gp_list = []
    loss_target_list = []
    loss_latent_list = []
    loss_pene_list = []
    for mp_idx, mp in enumerate(outmps):
        pred_marker_b, pred_params, prev_betas, prev_gender, R0, T0, pred_pelvis_loc, pred_joints, _ = mp
        loss_gp, loss_skate, loss_target, loss_pene = get_cost_search(pred_params, prev_betas, pred_joints, pred_marker_b, R0, T0, body_scene)
        loss_latent = latent_codes[mp_idx].abs().mean()
        loss_skate_list.append(loss_skate)
        loss_gp_list.append(loss_gp)
        loss_target_list.append(loss_target)
        loss_latent_list.append(loss_latent)
        loss_pene_list.append(loss_pene)

    loss_target_last = torch.stack(loss_target_list[-1:]).mean()
    loss_target_sequence = torch.stack(loss_target_list).mean()
    # loss_target = 0.5 * (loss_target_last + loss_target_sequence)
    loss_target = loss_target_last
    loss_skate = torch.stack(loss_skate_list).mean()
    loss_gp = torch.stack(loss_gp_list).mean()
    loss_latent = torch.stack(loss_latent_list).mean()
    loss_pene = torch.stack(loss_pene_list).mean()
    print(' target:', loss_target.item(), ' pene:', loss_pene.item(),
          ' target_last:', loss_target_last.item(), ' target_seq:', loss_target_sequence.item(),
          ' latent:', loss_latent.item(), ' skate:', loss_skate.item(), 'gp:', loss_gp.item())
    loss_total = loss_skate + loss_gp + loss_target * args.weight_target + loss_latent * args.weight_reg + loss_pene * args.weight_pene
    return loss_total

def gen_motion(body_s, max_depth=10):
    '''
    idx: the sequence seed index, which is to name the output file actually.
    max_depth: this determines how long the generated motion is (0.25sec per motion prmitive)
    '''
    # specify the start node
    [marker_start, body_param_start, body_betas,
        R_start, T_start, pelvis_loc_start] = canonicalize_static_pose(body_s)
    wpath = body_s['wpath']
    start_node = MPTNode(str(body_s['gender']), body_betas, R_start, T_start, pelvis_loc_start, None,
                         marker_start, marker_start, body_param_start, None, 'start-frame',
                         timestamp=-1,
                         curr_target_wpath=(1, wpath[1])
                         )
    if not args.load_search or not os.path.exists(os.path.join(outfoldername, 'search')):
        # depending on a static pose, generate a list of tree roots
        log_and_print('[INFO] generate roots in a heap')
        mp_heap_prev = gen_tree_roots(start_node, wpath, scene=body_s)
        log_and_print('[INFO] |--valid MPs={}'.format(mp_heap_prev.len()))
        if mp_heap_prev.len() == 0:
            log_and_print('[INFO] |--no movements searched. Program terminates.')
            return

        # generate tree leaves
        mp_heap_prev = expand_tree(mp_heap_prev, wpath, max_depth=max_depth, scene=body_s)
        if mp_heap_prev is None:
            return
        output = {'motion': None, 'wpath': wpath, 'obj_path':body_s['obj_path'], 'wpath_orients': body_s['wpath_orients']}
        log_and_print('[INFO] save results...')
        mp_leaves = mp_heap_prev
        motion_idx = 0
        gen_results = None
        while not mp_leaves.is_empty():
            if motion_idx >= 1:
                break
            gen_results = []
            mp_leaf = mp_leaves.pop()
            gen_results.append(mp_leaf.data)
            while mp_leaf.parent is not None:
                gen_results.append(mp_leaf.parent.data)
                mp_leaf = mp_leaf.parent
            gen_results.reverse()  # not including the start-frame,root node is 1-frame generation
            output['motion'] = gen_results
            ### save to file
            outfilename = 'results_{}.pkl'.format(body_repr)
            os.makedirs(os.path.join(outfoldername, 'search'), exist_ok=True)
            outfilename_f = os.path.join(outfoldername, 'search', outfilename)
            with open(outfilename_f, 'wb') as f:
                pickle.dump(output, f)
            motion_idx += 1
    else:
        outfilename = 'results_{}.pkl'.format(body_repr)
        outfilename_f = os.path.join(outfoldername, 'search', outfilename)
        with open(outfilename_f, 'rb') as f:
            serialized = pickle.load(f)
            gen_results = serialized['motion']
        print('search results loaded')

    # if args.use_optim:
    #     # optimize using search results for initialization
    #     num_primitives = len(gen_results)
    #     # init latent codes
    #     latent_codes = [primitive['mp_latent'].to(dtype=torch.float32, device=device).requires_grad_(True) for primitive in gen_results]
    #     if args.use_optim == 1:  # global optimization
    #         # init optimizer
    #         optimizer = optim.Adam(latent_codes, lr=args.lr)
    #         outmps = None
    #         for _ in tqdm(range(args.steps)):
    #             optimizer.zero_grad()
    #             outmps = []
    #             data_mp, _ = gen_tree_roots_optim(start_node.data, latent_codes[0], wpath, scene=body_s)
    #             outmps.append(data_mp)
    #             for d in range(1, num_primitives):
    #                 data_mp, _ = expand_tree_optim(data_mp, latent_codes[d], wpath, scene=body_s)
    #                 outmps.append(data_mp)
    #             loss = calc_loss(outmps, body_s, latent_codes)
    #             loss.backward()
    #             optimizer.step()
    #     elif args.use_optim == 2:  # CCD IK style optimization
    #         outmps = None
    #         for _ in tqdm(range(args.steps)):
    #             for latent_idx in reversed(range(num_primitives)):
    #                 optimizer = optim.Adam([latent_codes[latent_idx]], lr=args.lr)
    #                 optimizer.zero_grad()
    #                 outmps = []
    #                 data_mp, _ = gen_tree_roots_optim(start_node.data, latent_codes[0], wpath, scene=body_s)
    #                 outmps.append(data_mp)
    #                 for d in range(1, num_primitives):
    #                     data_mp, _ = expand_tree_optim(data_mp, latent_codes[d], wpath, scene=body_s)
    #                     outmps.append(data_mp)
    #                 loss = calc_loss(outmps, body_s, latent_codes)
    #                 loss.backward()
    #                 optimizer.step()
    #
    #     # add optim results to dict
    #     output = {'motion': [], 'wpath': wpath, 'markers': body_s['markers'],
    #               'obj_id': body_s['obj_id'], 'obj_transform': body_s['obj_transform']}
    #     mp_keys = ['blended_marker', 'smplx_params', 'betas', 'gender', 'transf_rotmat', 'transf_transl', 'pelvis_loc',
    #                'joints', 'mp_type']
    #     for mp in outmps:
    #         mp_node = {}
    #         for idx, key in enumerate(mp_keys):
    #             if key in ['gender', 'mp_type', 'betas']:
    #                 mp_node[key] = mp[idx] if type(mp[idx]) == str else mp[idx].detach().cpu().numpy()
    #             elif key in ['smplx_params']:
    #                 mp_node[key] = mp[idx].detach().cpu().numpy()
    #             else:
    #                 mp_node[key] = mp[idx].detach().cpu().numpy()
    #         output['motion'].append(mp_node)
    #     ## save to file
    #     outfilename = 'results_{}.pkl'.format(body_repr)
    #     os.makedirs(os.path.join(outfoldername, 'optim' + str(args.use_optim)), exist_ok=True)
    #     outfilename_f = os.path.join(outfoldername, 'optim' + str(args.use_optim), outfilename)
    #     with open(outfilename_f, 'wb') as f:
    #         pickle.dump(output, f)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_policy', default='MPVAEPolicy_v0')
    # parser.add_argument('--cfg1', default='MPVAEPolicy_v0',
    #                     help='config for locomotion stage.')
    # parser.add_argument('--cfg2', default='MPVAEPolicy_babel',
    #                     help='config for interaction stage')
    parser.add_argument('--checkpoint', type=str, default='epoch-500.ckp')
    parser.add_argument('--max_depth', type=int, default=60,
                        help='the maximal number of (0.25-second) motion primitives in each motion.')
    parser.add_argument('--num_sequence', type=int, default=2)
    parser.add_argument('--num_primitives', type=int, default=16)
    parser.add_argument('--switch_stage', type=int, default=0,
                        help='opt to switch off policy when close to target than switch thresh')
    parser.add_argument('--switch_thresh', type=float, default=0.75)
    parser.add_argument('--goal_thresh', type=float, default=0.05)
    parser.add_argument('--use_policy', type=int, default=0)
    parser.add_argument('--ground_euler', nargs=3, type=float, default=[0, 0, 0],
                        help='the gorund plan rotation. Normally we set it to flat with Z-up Y-forward.')  # which dataset to evaluate? choose only one
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--use_collision', type=int, default=0)

    # search
    parser.add_argument('--weight_target', type=float, default=0.1)
    parser.add_argument('--weight_ori', type=float, default=0.1)
    parser.add_argument('--weight_pene', type=float, default=1)
    parser.add_argument('--weight_gp', type=float, default=1)
    parser.add_argument('--weight_skate', type=float, default=1)
    parser.add_argument('--sample_1frame', type=int, default=16)
    parser.add_argument('--sample_2frame', type=int, default=8)
    parser.add_argument('--max_keep_sample', type=int, default=64)

    # optimizer
    parser.add_argument('--load_search', type=int, default=0)
    parser.add_argument('--use_optim', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_reg', type=float, default=0.01)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda',
                          index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() else torch.device(
        'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    # torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = False  # https://github.com/pytorch/captum/issues/564

    """global parameter"""
    random_seed = args.random_seed
    # default
    # n_gens_1frame = 16  # the number of primitives to generate from a single-frame motion seed
    # n_gens_2frame = 4  # the nunber of primitives to generate from a two-frame motion seed
    # max_nodes_to_expand = 4  # in the tree search, how many nodes to expand at the same level.
    # sample more
    n_gens_1frame = args.sample_1frame     # the number of primitives to generate from a single-frame motion seed
    n_gens_2frame = args.sample_2frame     # the nunber of primitives to generate from a two-frame motion seed
    max_nodes_to_expand = args.max_keep_sample # in the tree search, how many nodes to expand at the same level.
    # rely simply on policy
    # n_gens_1frame = 1  # the number of primitives to generate from a single-frame motion seed
    # n_gens_2frame = 1  # the nunber of primitives to generate from a two-frame motion seed
    # max_nodes_to_expand = 1  # in the tree search, how many nodes to expand at the same level.
    GOAL_THRESH = args.goal_thresh  # the threshold to reach the goal.
    HARD_CONTACT = False  # for ranking the primitives in the tree search. If True, then motion primitives with implausible foot-ground contact are discarded.
    USE_POLICY_MEAN = False  # only use the mean of the policy. If False, random samples are drawn from the policy.
    USE_POLICY = args.use_policy  # If False, random motion generation will be performed.
    SCENE_ORI = 'ZupYf'  # the coordinate setting of the scene.
    max_depth = args.max_depth
    NUM_SEQ = args.num_sequence  # the number of sequences to produce

    cfg_policy = ConfigCreator(args.cfg_policy)
    cfg_1frame_male = cfg_policy.trainconfig['cfg_1frame_male']
    cfg_2frame_male = cfg_policy.trainconfig['cfg_2frame_male']
    cfg_1frame_female = cfg_policy.trainconfig['cfg_1frame_female']
    cfg_2frame_female = cfg_policy.trainconfig['cfg_2frame_female']
    body_repr = cfg_policy.modelconfig['body_repr']
    REPROJ_FACTOR = cfg_policy.modelconfig.get('reproj_factor', 1.0)

    """data"""
    # ground data
    Rg = R.from_euler('xyz', np.array([0, 0, 0]), degrees=True)
    rotmat_g = Rg.as_matrix()
    # body motion data
    # bm_path = config_env.get_body_model_path()
    # batch_gen = BatchGeneratorFollowPathInCubes(dataset_path='exp_GAMMAPrimitive/data/Cubes/scene_cubes_000_navimesh.obj_traj',
    #                                             body_model_path=bm_path,
    #                                             scene_ori=SCENE_ORI)
    # batch_gen.get_rec_list()

    """set GAMMA primitive networks"""
    genop_1frame_male = configure_model(cfg_1frame_male, args.gpu_index, args.random_seed)
    genop_1frame_female = configure_model(cfg_1frame_female, args.gpu_index, args.random_seed)
    genop_2frame_male = configure_model(cfg_2frame_male, args.gpu_index, args.random_seed)
    genop_2frame_female = configure_model(cfg_2frame_female, args.gpu_index, args.random_seed)

    if USE_POLICY:
        policy_model = GAMMAPolicy(cfg_policy.modelconfig)
        policy_model.eval()
        policy_model.to(device)
        ckp_list = sorted(glob.glob(os.path.join(cfg_policy.trainconfig['save_dir'], args.checkpoint)),
                          key=os.path.getmtime)
        if len(ckp_list) > 0:
            ckptfile = os.path.join(ckp_list[-1])
            checkpoint = torch.load(ckptfile, map_location=device)
            policy_model.load_state_dict(checkpoint['model_state_dict'])
            print('[INFO] --load checkpoint from {}'.format(ckptfile))
    else:
        policy_model = None

    """body model parsers"""
    pconfig_mp = {
        'n_batch': 10,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_mp = SMPLXParser(pconfig_mp)

    pconfig_2frame = {
        'n_batch': 2,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_2frame = SMPLXParser(pconfig_2frame)

    pconfig_1frame = {
        'n_batch': 1,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_1frame = SMPLXParser(pconfig_1frame)
    """import coap model"""
    smplx_model = smplx.create(model_path=get_body_model_path(), model_type='smplx',
                               gender='male', num_pca_comps=12, batch_size=1)
    coap_model = attach_coap(smplx_model, pretrained=True, device=device)
    smplx_model_10 = smplx.create(model_path=get_body_model_path(), model_type='smplx',
                               gender='male', num_pca_comps=12, batch_size=10)
    coap_model_10 = attach_coap(smplx_model_10, pretrained=True, device=device)

    """main block for motion generation"""
    resultdir = 'results/two_stage/{}'.format(args.exp_name)
    bm_path = config_env.get_body_model_path()
    batch_gen = BatchGeneratorSceneNav(dataset_path='data/scene_nav',
                                      body_model_path=bm_path,
                                     )

    data = batch_gen.next_body(sigma=1, visualize=False, use_zero_pose=False)
    idx_seq = 0
    while idx_seq < NUM_SEQ:
        outfoldername = '{}/randseed{:03d}_seq{:03d}_sit/'.format(resultdir, random_seed, idx_seq)
        if not os.path.exists(outfoldername):
            os.makedirs(outfoldername)
        logger = get_logger(outfoldername, mode='eval')
        log_and_print('[INFO] generate sequence {:d}'.format(idx_seq))

        gen_motion(data, max_depth=max_depth)
        idx_seq += 1