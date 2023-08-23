"""Information
- This script is to generate motions by optimization.
"""

import warnings
import numpy as np
import random
import argparse
import os, sys, glob
import copy
import pickle
import pdb
import torch
from torch import nn
from torch import optim
import torch.functional as F
from copy import deepcopy
from tqdm import tqdm

sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorFollowPathInCubes
from exp_GAMMAPrimitive.utils import config_env
from exp_GAMMAPrimitive.utils.utils_canonicalize_samp import canonicalize_subsequence

from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP
from models.models_policy import GAMMAPolicy
from models.baseops import SMPLXParser

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


def gen_motion_one_step(motion_model, marker_seed, bparam_seed, prev_betas, t_his=-1, n_gens=1, z=None):
    [pred_markers,
     pred_params] = motion_model.generate(
                                        marker_seed.permute([1, 0, 2]),
                                        bparam_seed.permute([1, 0, 2]),
                                        prev_betas,
                                        n_gens=n_gens,
                                        to_numpy=False,
                                        param_blending=True,
                                        t_his=2,
                                        z=z,
                                        )
    pred_markers = pred_markers.reshape(pred_markers.shape[0],
                                        pred_markers.shape[1], -1, 3)  # [n_gens, t, V, 3]
    # pred_markers, pred_params = pred_markers.permute([1, 0, 2, 3]), pred_params.permute([1, 0, 2])
    return pred_markers, pred_params

def get_cost_search(bparams, joints, Y_l, R0, T0, goal):
    R0 = R0.repeat(N_GENS_LEAF, 1, 1)
    T0 = T0.repeat(N_GENS_LEAF, 1, 1)
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
    # dist2gp = (torch.abs(Y_wz.min(dim=-1)[0]) - 0.05).clamp(min=0)
    dist2skat = (torch.abs(Y_w_speed.min(dim=-1)[0]) - 0.075).clamp(min=0)

    # r_contact = torch.exp(-dist2gp) * torch.exp(-dist2skat) #(b, )
    # '''evaluate the distance to the final target'''
    # target_wpath_l = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1), pt_wpath[None, ...] - T0)[:, :, :2]
    # dist2target = torch.norm(target_wpath_l[:, 0] - pel_loc[:, -1, :2], dim=-1)

    '''evaluate body facing orientation'''
    # joints_end = joints[:, -1]  # [b,p,3]
    # x_axis = joints_end[:, 2, :] - joints_end[:, 1, :]
    # x_axis[:, -1] = 0
    # x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
    # z_axis = torch.cuda.FloatTensor([[0, 0, 1]], device=device).repeat(x_axis.shape[0], 1)
    # y_axis = torch.cross(z_axis, x_axis)
    # b_ori = y_axis[:, :2]
    # t_ori = target_wpath_l[:, 0] - pel_loc[:, -1, :2]
    # t_ori = t_ori / torch.norm(t_ori, dim=-1, keepdim=True)
    # dist2ori = -torch.einsum('bi,bi->b', t_ori, b_ori)

    dist2target = torch.mean(torch.norm(goal['markers'][None, :, :] - Y_w[:, -1], dim=-1), dim=-1)  # [b, m, 3] -> [b,m] -> [b]
    cost = dist2skat + 0.05 * dist2target
    # reward = r_contact
    return dist2skat, dist2target

def expand(data_mp, latent_code):
    prev_markers_b, prev_params, prev_betas, prev_gender, prev_rotmat, prev_transl, prev_pelvis_loc, prev_joints, _ = data_mp

    '''produce marker seed'''
    t_his = 2
    body_param_seed = prev_params[:, -t_his:]  # [b,t,d], need deepcopy, body_param will be changed inplace
    nb, nt = body_param_seed.shape[:2]
    ## move frame to the second last body's pelvis
    assert nb == 1
    R_, T_ = smplxparser_1f_root.get_new_coordinate(
        betas=prev_betas,
        gender=prev_gender,
        xb=body_param_seed[:, 0],
        to_numpy=False)  # [b,3,3][b,1,3]
    T0 = torch.einsum('bij,btj->bti', prev_rotmat, T_) + prev_transl
    R0 = torch.einsum('bij,bjk->bik', prev_rotmat, R_)

    marker_seed = torch.einsum('bij,btpj->btpi', R_.permute(0, 2, 1), prev_markers_b[:, -t_his:] - T_[..., None, :])
    # pel_loc_seed = torch.einsum('bij,btj->bti', R_.permute(0, 2, 1), prev_pelvis_loc[:, -t_his:] - T_)

    # original update_transl_glorot changes xb inplace !!
    body_param_seed = smplxparser_2f.update_transl_glorot(
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

    [pred_markers, pred_params] = gen_motion_one_step(motion_model,
                                                      marker_seed, body_param_seed, prev_betas,
                                                      t_his=2, n_gens=N_GENS_LEAF,
                                                      z=latent_code)

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

def calc_loss(outmps, goal, latent_codes):
    # pred_marker_b, pred_params, prev_betas, prev_gender, R0, T0, pred_pelvis_loc, pred_joints, _ = outmps[-1]
    # loss = get_cost_search(pred_params, pred_joints, pred_marker_b, R0, T0, goal)
    # return loss
    loss_skate_list = []
    loss_target_list = []
    loss_latent_list = []
    for mp_idx, mp in enumerate(outmps):
        pred_marker_b, pred_params, prev_betas, prev_gender, R0, T0, pred_pelvis_loc, pred_joints, _ = mp
        loss_skate, loss_target = get_cost_search(pred_params, pred_joints, pred_marker_b, R0, T0, goal)
        loss_latent = latent_codes[mp_idx].abs().mean()
        loss_skate_list.append(loss_skate)
        loss_target_list.append(loss_target)
        loss_latent_list.append(loss_latent)

    # loss_target = torch.stack(loss_target_list[-1:]).mean()
    loss_target = torch.stack(loss_target_list).mean()
    loss_skate = torch.stack(loss_skate_list).mean()
    loss_latent = torch.stack(loss_latent_list).mean()
    print('target:', loss_target.item(), ' latent:', loss_latent.item(), ' skate:', loss_skate.item())
    loss_total = loss_skate * args.weight_skate + loss_target * args.weight_target + loss_latent * args.weight_reg
    return loss_total

def save_rollout_results(outmps, goal, outfolder, idx_seq=0):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    mp_keys = ['blended_marker', 'smplx_params', 'betas', 'gender', 'transf_rotmat', 'transf_transl', 'pelvis_loc', 'joints'
               'mp_type']
    nb = 1 if USE_POLICY_MEAN else N_GENS_ROOT
    for b in range(nb)[:1]:  # only one body
        outmps_nodes = {'motion': [], 'goal_markers': goal['markers'].detach().cpu().numpy()}
        for mp in outmps:
            mp_node = {}
            for idx, key in enumerate(mp_keys):
                if key in ['gender', 'mp_type', 'betas']:
                    mp_node[key] = mp[idx] if type(mp[idx]) == str else mp[idx].detach().cpu().numpy()
                elif key in ['smplx_params']:
                    mp_node[key] = mp[idx][b:b + 1].detach().cpu().numpy()
                else:
                    mp_node[key] = mp[idx][b].detach().cpu().numpy()
            outmps_nodes['motion'].append(mp_node)
        with open(outfolder + '/motion_{:02d}_{:02d}.pkl'.format(idx_seq, b), 'wb') as f:
            pickle.dump(outmps_nodes, f)


def gen_optimize(subsequence, save_file=True, idx_seq=0):
    # init latent codes
    latent_codes = torch.randn(MAX_DEPTH, 1, motion_model.model.predictor.z_dim,
                               requires_grad=True, device=device)
    # init optimizer
    optimizer = optim.Adam([latent_codes], lr=args.lr)

    rotmat = subsequence['transf_rotmat']
    transl = subsequence['transf_transl']
    goal_pelvis = np.dot(subsequence['joints'][-1:, 0], rotmat.T) + transl
    goal_markers = np.dot(subsequence['marker_ssm2_67'][-1, :], rotmat.T) + transl
    goal_pelvis = torch.tensor(goal_pelvis, dtype=torch.float32, device=device)  # [1, 3]
    goal_markers = torch.tensor(goal_markers, dtype=torch.float32, device=device)  # [m, 3]
    goal = {'pelvis': goal_pelvis,
            'markers': goal_markers}

    """generating tree roots"""
    t_his = 2
    smplx_transl = torch.tensor(subsequence['trans'][:t_his], dtype=torch.float32, device=device)
    smplx_poses = torch.tensor(subsequence['poses'][:t_his, :66], dtype=torch.float32, device=device)
    smplx_handposes = torch.cuda.FloatTensor(smplx_transl.shape[0], 24).zero_()
    # print(smplx_transl.shape, smplx_poses.shape, smplx_handposes.shape)
    prev_params = torch.cat([smplx_transl,
                             smplx_poses, smplx_handposes],
                            dim=-1)  # [2,d]
    prev_betas, gender = torch.tensor(subsequence['betas'], dtype=torch.float32, device=device), \
                         subsequence['gender']
    R0, T0 = torch.tensor(subsequence['transf_rotmat'], dtype=torch.float32, device=device).unsqueeze(0), \
             torch.tensor(subsequence['transf_transl'], dtype=torch.float32, device=device).unsqueeze(0)

    body_param_seed = prev_params.unsqueeze(0)  # [b, t, d]
    nb, nt = body_param_seed.shape[:2]
    assert nb == 1
    marker_seed = torch.tensor(subsequence['marker_ssm2_67'][:t_his], dtype=torch.float32, device=device)  # [2, m, 3]
    marker_seed = marker_seed.unsqueeze(0)
    joints_seed = torch.tensor(subsequence['joints'][:t_his], dtype=torch.float32, device=device)  # [2, 22, 3]
    joints_seed = joints_seed.unsqueeze(0)
    pelvis_seed = torch.tensor(subsequence['joints'][:t_his, 0], dtype=torch.float32, device=device)  # [2, 3]
    pelvis_seed = pelvis_seed.unsqueeze(0)

    outmps = None
    for step in tqdm(range(args.steps)):
        outmps = []
        data_mp = [marker_seed, body_param_seed, prev_betas, gender, R0, T0, pelvis_seed, joints_seed, 'start-frame']
        optimizer.zero_grad()
        for d in range(MAX_DEPTH):
            data_mp, _ = expand(data_mp, latent_codes[d])
            outmps.append(data_mp)
        loss = calc_loss(outmps, goal, latent_codes)
        loss.backward()
        optimizer.step()

    if save_file:
        """visualze rollout results in blender, for debugging"""
        save_rollout_results(outmps, goal, 'results/tmp123/GAMMAVAECombo_optim/{}'.format(args.cfg_policy),
                             idx_seq=idx_seq)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_policy', default='MPVAEPolicy_tmp',
                        help='specify the motion model and the policy config.')
    parser.add_argument('--max_depth', type=int, default=60,
                        help='the maximal number of (0.25-second) motion primitives in each motion.')
    parser.add_argument('--ground_euler', nargs=3, type=float, default=[0, 0, 0],
                        help='the gorund plan rotation. Normally we set it to flat with Z-up Y-forward.')  # which dataset to evaluate? choose only one
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)

    parser.add_argument('--num_primitive', type=int, default=4)
    parser.add_argument('--num_gen', type=int, default=4)
    parser.add_argument('--beam_size', type=int, default=16)

    # optimizer
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_reg', type=float, default=0.1)
    parser.add_argument('--weight_target', type=float, default=0.1)
    parser.add_argument('--weight_skate', type=float, default=0.1)
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
    torch.backends.cudnn.enabled = False  #  https://github.com/pytorch/captum/issues/564

    cfg_policy = ConfigCreator(args.cfg_policy)
    # cfg_1frame_male = cfg_policy.trainconfig['cfg_1frame_male']
    cfg_2frame_male = cfg_policy.trainconfig['cfg_2frame_male']
    # cfg_1frame_female = cfg_policy.trainconfig['cfg_1frame_female']
    # cfg_2frame_female = cfg_policy.trainconfig['cfg_2frame_female']
    body_repr = cfg_policy.modelconfig['body_repr']

    """global parameter"""
    REPROJ_FACTOR = 0.5
    USE_POLICY_MEAN = False
    USE_POLICY = False
    SCENE_ORI = 'ZupYf'  # the coordinate setting of the scene.

    MAX_DEPTH = int(args.num_primitive * 10 / 8)
    GOAL_THRESH = 0.75
    N_GENS_ROOT = 1
    N_GENS_LEAF = 1
    # NUM_SEQ = 4  # the number of sequences to produce

    """set GAMMA primitive networks"""
    # genop_1frame_male = configure_model(cfg_1frame_male, args.gpu_index, args.random_seed)
    # genop_1frame_female = configure_model(cfg_1frame_female, args.gpu_index, args.random_seed)
    genop_2frame_male = configure_model(cfg_2frame_male, args.gpu_index, args.random_seed)
    # genop_2frame_female = configure_model(cfg_2frame_female, args.gpu_index, args.random_seed)
    """retrieve current motion model"""
    motion_model = genop_2frame_male

    """body model parsers"""
    pconfig_mp = {
        'n_batch': 10 * N_GENS_ROOT * N_GENS_LEAF,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_mp = SMPLXParser(pconfig_mp)

    pconfig_1f_root = {
        'n_batch': N_GENS_ROOT,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_1f_root = SMPLXParser(pconfig_1f_root)

    pconfig_2f = {
        'n_batch': 2,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_2f = SMPLXParser(pconfig_2f)

    pconfig_mp_root = {
        'n_batch': 10 * N_GENS_ROOT,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_mp_root = SMPLXParser(pconfig_mp_root)

    """policy network and their optimizer"""
    policy_model = GAMMAPolicy(cfg_policy.modelconfig)
    policy_model.eval()
    policy_model.to(device)
    ckp_list = sorted(glob.glob(os.path.join(cfg_policy.trainconfig['save_dir'], 'epoch-450.ckp')),
                      key=os.path.getmtime)
    # if len(ckp_list)>0:
    try:
        checkpoint = torch.load(ckp_list[-1], map_location=device)
        policy_model.load_state_dict(checkpoint['model_state_dict'])
        print('[INFO] --load checkpoint from {}'.format(ckp_list[-1]))
    except:
        print('[INFO] --use initial policy weights')


    # reference sequence
    seq_name = 'armchair_stageII.pkl'
    seq_path = os.path.join('/home/kaizhao/dataset/samp/', seq_name)
    end_frame = 1480
    start_frame = end_frame - 30 * args.num_primitive
    subsequence = canonicalize_subsequence(seq_path, start_frame, end_frame)
    """rollout"""
    for idx_seq in range(args.num_gen):
        traj = gen_optimize(subsequence, save_file=True, idx_seq=idx_seq)

