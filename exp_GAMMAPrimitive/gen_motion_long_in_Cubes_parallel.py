"""Information
- This script is to generate long-term motions in the scene with cubes. 
- By setting the flags, motion is generated either by the policy or randomly.
- The tree-based search is performed in parallel with a high speed. 
- Rather than sorting, it just pickes the best one and continue.
- Therefore, variables are directly in torch.Tensor.
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


sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorFollowPathInCubes
from exp_GAMMAPrimitive.utils import config_env

from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP
from models.models_policy import GAMMAPolicy
from models.baseops import SMPLXParser


sys.setrecursionlimit(10000) # if too small, deepcopy will reach the maximal depth limit.



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



def gen_motion_one_step(motion_model, states, bparam_seed, prev_betas, t_his = -1):
    if t_his == 1:
        n_gens = N_GENS_ROOT
    elif t_his == 2:
        n_gens = N_GENS_LEAF
    else:
        n_gens = -1
    [pred_markers,
     pred_params,
     act,
     act_log_prob,
     value] = motion_model.generate_ppo(policy_model,
                                        states.permute([1,0,2]), 
                                        bparam_seed.permute([1,0,2]),
                                        prev_betas,
                                        n_gens=n_gens,
                                        to_numpy=False,
                                        param_blending=True,
                                        use_policy_mean=USE_POLICY_MEAN,
                                        use_policy=USE_POLICY
                                        )
    pred_markers = pred_markers.reshape(pred_markers.shape[0],
                                    pred_markers.shape[1], -1,3)#[n_gens, t, V, 3]
    return pred_markers.permute([1,0,2,3]), pred_params.permute([1,0,2]), act, act_log_prob, value

                                                

def canonicalize_static_pose(data):
    smplx_transl = data['transl']
    smplx_glorot = data['global_orient']
    smplx_poses = data['body_pose']
    gender=data['gender']
    smplx_handposes = torch.cuda.FloatTensor(smplx_transl.shape[0], 24).zero_()
    prev_params = torch.cat([smplx_transl,smplx_glorot,
                                    smplx_poses,smplx_handposes],
                                    dim=-1) #[1,d]
    if N_GENS_ROOT != 1:
        raise NotImplementedError('currently we only support 1 node to expand in the tree.')
    prev_params = prev_params.repeat(N_GENS_ROOT, 1, 1)
    prev_betas = data['betas']
    nb,nt = prev_params.shape[:2]
    body_param_seed = prev_params.reshape(nt*nb, -1)
    ## move frame to the body's pelvis
    R0, T0 = smplxparser_1f_root.get_new_coordinate(
                                        betas=prev_betas,
                                        gender=gender,
                                        xb=prev_params[:,0],
                                        to_numpy=False)
    
    ## get the last body param and marker in the new coordinate
    body_param_seed = smplxparser_1f_root.update_transl_glorot(R0, T0,
                                            betas=prev_betas,
                                            gender=gender,
                                            xb=body_param_seed,
                                            to_numpy=False).reshape(nb, nt, -1)
    
    return body_param_seed, prev_betas, gender, R0, T0



def get_wpath_feature(Y_l, pel, R0, T0, pt_wpath):
    '''
    --Y_l = [b,t,d] local marker
    --pel = [b,t,d]
    --pt_wpath = [1,d]
    '''
    nb, nt = pel.shape[:2]
    Y_l = Y_l.reshape(nb, nt, -1, 3)
    pt_wpath_l_3d = torch.einsum('bij,btj->bti', R0.permute(0,2,1), pt_wpath[None,...]-T0)

    '''extract path feature = normalized direction + unnormalized height'''
    fea_wpathxy = pt_wpath_l_3d[:,:,:2]-pel[:,:,:2]
    dist_xy = torch.norm(fea_wpathxy, dim=-1, keepdim=True)
    fea_wpathxy = fea_wpathxy/dist_xy
    fea_wpathz = pt_wpath_l_3d[:,:,-1:]-pel[:,:,-1:]
    fea_wpath = torch.cat([fea_wpathxy, fea_wpathz], dim=-1)

    '''extract marker feature'''
    fea_marker = pt_wpath_l_3d[:,:,None,:]-Y_l
    dist_m_3d = torch.norm(fea_marker, dim=-1, keepdim=True)
    fea_marker_3d_n = (fea_marker/dist_m_3d).reshape(nb, nt, -1)

    '''extract marker feature with depth'''
    dist_m_2d = torch.norm(fea_marker[:,:,:,:2], dim=-1, keepdim=True)
    fea_marker_xyn = fea_marker[:,:,:,:2] / dist_m_2d
    fea_marker_h = torch.cat([fea_marker_xyn, fea_marker[:,:,:,-1:]],dim=-1).reshape(nb, nt, -1)

    return dist_xy, fea_wpath, fea_marker_3d_n, fea_marker_h



def update_local_target(marker, pelvis_loc, idx_target_curr, R0, T0, wpath):
    dist, fea_wpath, fea_marker, fea_marker_h = get_wpath_feature(marker, pelvis_loc, R0, T0, wpath[idx_target_curr][None,...])
    while torch.any(dist<GOAL_THRESH) and idx_target_curr<wpath.shape[0]-1:
        idx_target_curr = idx_target_curr+1
        dist, fea_wpath, fea_marker, fea_marker_h = get_wpath_feature(marker, pelvis_loc, R0, T0, wpath[idx_target_curr][None,...])
        
    if torch.any(dist<GOAL_THRESH) and idx_target_curr==wpath.shape[0]-1:
        return None
    else:
        return idx_target_curr, fea_wpath, fea_marker, fea_marker_h




def get_cost_search(bparams, joints, Y_l, R0, T0, pt_wpath):
    R0 = R0.repeat(N_GENS_LEAF, 1,1)
    T0 = T0.repeat(N_GENS_LEAF, 1,1)
    pel_loc = joints[:,:,0]
    nb, nt = Y_l.shape[:2]
    Y_w = torch.einsum('bij,btpj->btpi',R0, Y_l)+T0[:,None,...]
    #----select motion index of proper contact with the ground
    Y_wz = Y_w[:,:,:,-1] #[b,t, P]
    Y_wz = Y_wz.reshape(nb, -1)
    h = 1/40
    Y_w_speed = torch.norm(Y_w[:,2:]-Y_w[:,:-2], dim=-1)/(2*h) #[b, t=9,P=67]
    Y_w_speed = Y_w_speed.reshape(nb, -1)
    '''evaluate contact soft'''
    dist2gp = (torch.abs(Y_wz.min(dim=-1)[0])-0.05).clamp(min=0)
    dist2skat = (torch.abs(Y_w_speed.min(dim=-1)[0])-0.075).clamp(min=0)
    
    # r_contact = torch.exp(-dist2gp) * torch.exp(-dist2skat) #(b, )
    # '''evaluate the distance to the final target'''
    target_wpath_l = torch.einsum('bij,btj->bti', R0.permute(0,2,1), pt_wpath[None,...]-T0)[:,:,:2]
    dist2target = torch.norm(target_wpath_l[:,0]-pel_loc[:, -1,:2],dim=-1)
    
    '''evaluate body facing orientation'''
    joints_end = joints[:,-1] #[b,p,3]
    x_axis = joints_end[:,2,:] - joints_end[:,1,:]
    x_axis[:,-1] = 0
    x_axis = x_axis / torch.norm(x_axis,dim=-1,keepdim=True)
    z_axis = torch.cuda.FloatTensor([[0,0,1]], device=device).repeat(x_axis.shape[0],1)
    y_axis = torch.cross(z_axis, x_axis)
    b_ori = y_axis[:,:2]
    t_ori = target_wpath_l[:,0]-pel_loc[:, -1,:2]
    t_ori = t_ori/torch.norm(t_ori, dim=-1, keepdim=True)
    dist2ori = -torch.einsum('bi,bi->b', t_ori, b_ori)
    
    cost = dist2gp + dist2skat + 0.05*dist2target + 0.1*dist2ori
    # reward = r_contact
    return cost



def gen_tree_roots(data_mp, wpath, idx_target_curr):
    body_param_seed, prev_betas, gender, R0, T0 = data_mp
    nb, nt = body_param_seed.shape[:2]
    t_his = 1

    """retrieve current motion model"""
    motion_model = genop_1frame_male if gender=='male' else genop_1frame_female
    """retrieve current states"""

    marker_seed = smplxparser_1f_root.get_markers(
                                        betas=prev_betas,
                                        gender=gender,
                                        xb=body_param_seed.reshape(nb*nt, -1),
                                        to_numpy=False
                                        ).reshape(nb, nt, -1)

    pelvis_loc = smplxparser_1f_root.get_jts(betas=prev_betas,
                                    gender=gender,
                                    xb=body_param_seed.reshape(nb*nt, -1),
                                    to_numpy=False
                                    )[:, 0]#[b*t, 3]
    pelvis_loc = pelvis_loc.reshape(nb, nt, -1)


    # dist, fea_wpath, fea_marker, fea_marker_h = get_wpath_feature(marker_seed, pelvis_loc[:,:t_his], R0, T0, wpath[-1:])
    wpath_feature = update_local_target(marker_seed, pelvis_loc[:,:t_his],
                                                        idx_target_curr,
                                                        R0, T0, wpath)
    if wpath_feature is not None:
        idx_target_curr, fea_wpath, fea_marker, fea_marker_h = wpath_feature
    else:
        return None

    if body_repr == 'ssm2_67_condi_marker':
        states = torch.cat([marker_seed,fea_marker],dim=-1)
    else:
        raise NotImplementedError

    # generate markers and regress to body params
    [pred_markers, pred_params,
     act, act_log_prob, value] = gen_motion_one_step(motion_model,
                                    states, body_param_seed, prev_betas)
    nb, nt = pred_params.shape[:2]

    pred_joints = smplxparser_mp_root.get_jts(betas=prev_betas,
                                        gender=gender,
                                        xb=pred_params.reshape([nb*nt, -1]),
                                        to_numpy=False).reshape([nb,nt,-1,3])
    pred_pelvis_loc = pred_joints[:,:,0]
    pred_markers_proj = smplxparser_mp_root.get_markers(betas=prev_betas,
                                            gender=gender,
                                            xb=pred_params.reshape([nb*nt, -1]),
                                            to_numpy=False).reshape((nb, nt, -1, 3))  #[b, t, p, 3]

    pred_marker_b = REPROJ_FACTOR*pred_markers_proj+(1-REPROJ_FACTOR)*pred_markers
    
    return [pred_marker_b, pred_params, prev_betas, gender, R0, T0, pred_pelvis_loc, 'start-frame'],idx_target_curr




def expand_tree(data_mp, wpath, idx_targer_curr):
    prev_markers_b, prev_params, prev_betas, prev_gender, prev_rotmat, prev_transl, prev_pelvis_loc, _ = data_mp

    '''produce marker seed'''
    t_his = 2
    body_param_seed = prev_params[:,-t_his:] #[b,t,d]
    nb, nt = body_param_seed.shape[:2]
    ## move frame to the second last body's pelvis
    R_, T_ = smplxparser_1f_root.get_new_coordinate(
                                        betas=prev_betas,
                                        gender=prev_gender,
                                        xb=body_param_seed[:,0],
                                        to_numpy=False) #[b,3,3][b,1,3]
    T0 = torch.einsum('bij,btj->bti', prev_rotmat, T_)+prev_transl
    R0 = torch.einsum('bij,bjk->bik', prev_rotmat, R_)

    marker_seed = torch.einsum('bij,btpj->btpi', R_.permute(0,2,1), prev_markers_b[:,-t_his:]-T_[...,None,:])
    pel_loc_seed = torch.einsum('bij,btj->bti', R_.permute(0,2,1), prev_pelvis_loc[:,-t_his:]-T_)
    wpath_feature = update_local_target(marker_seed, pel_loc_seed,
                                        idx_targer_curr,
                                        R0, T0, wpath)
    if wpath_feature is not None:
        idx_target_curr, fea_wpath, fea_marker, fea_marker_h = wpath_feature
    else:
        return None
    
    '''generate future motions'''
    motion_model = genop_2frame_male if prev_gender=='male' else genop_2frame_female
    marker_seed = marker_seed.reshape(nb,t_his,-1) #[b,t,d]
    if body_repr == 'ssm2_67_condi_marker':
        states = torch.cat([marker_seed,fea_marker],dim=-1)
    else:
        raise NotImplementedError

    [pred_markers, pred_params,
    act, act_log_prob, value] = gen_motion_one_step(motion_model,
                                        states, body_param_seed, prev_betas, t_his=2)

    nb, nt = pred_params.shape[:2]
    pred_joints = smplxparser_mp.get_jts(betas=prev_betas,
                                        gender=prev_gender,
                                        xb=pred_params.reshape([nb*nt, -1]),
                                        to_numpy=False).reshape([nb,nt,-1,3])#[t*b, 3]
    pred_pelvis_loc = pred_joints[:,:,0]
    pred_markers_proj = smplxparser_mp.get_markers(betas=prev_betas,
                                            gender=prev_gender,
                                            xb=pred_params.reshape([nb*nt, -1]),
                                            to_numpy=False).reshape((nb, nt, -1, 3))  #[b, t, p, 3]

    """trim the tree and keep the best K=1 motion"""
    pred_marker_b = REPROJ_FACTOR*pred_markers_proj+(1-REPROJ_FACTOR)*pred_markers
    traj_rewards_search = get_cost_search(pred_params, pred_joints, pred_marker_b, R0, T0, wpath[idx_target_curr])
    rank_idx = torch.topk(traj_rewards_search, k=N_GENS_ROOT, dim=0, largest=False, sorted=True)[1]
    pred_marker_b = pred_marker_b[rank_idx]
    pred_params = pred_params[rank_idx]
    pred_joints = pred_joints[rank_idx]
    pred_pelvis_loc = pred_pelvis_loc[rank_idx]

    return [pred_marker_b, pred_params, prev_betas, prev_gender, R0, T0, pred_pelvis_loc, '2-frame'], idx_target_curr




def save_rollout_results(wpath, outmps, outfolder, idx_seq=0):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    mp_keys = ['blended_marker','smplx_params', 'betas', 'gender', 'transf_rotmat', 'transf_transl','pelvis_loc','mp_type']
    nb = 1 if USE_POLICY_MEAN else N_GENS_ROOT
    for b in range(nb)[:1]: #only one body
        outmps_nodes = {'wpath':wpath.detach().cpu().numpy(), 'motion':[]}
        for mp in outmps:
            mp_node = {}
            for idx, key in enumerate(mp_keys):
                if key in ['gender','mp_type', 'betas']:
                    mp_node[key] = mp[idx] if type(mp[idx])==str else mp[idx].detach().cpu().numpy()
                elif key in ['smplx_params']:
                    mp_node[key] = mp[idx][b:b+1].detach().cpu().numpy()
                else:
                    mp_node[key] = mp[idx][b].detach().cpu().numpy()
            outmps_nodes['motion'].append(mp_node)
        with open(outfolder+'/motion_{:02d}_{:02d}.pkl'.format(idx_seq, b), 'wb') as f:
            pickle.dump(outmps_nodes,f)



def rollout(body_s, save_file=True, idx_seq=0):
    '''
    idx: the sequence seed index, which is to name the output file actually.
    '''

    outmps = []
    # specify the start node
    data_mp0 = canonicalize_static_pose(body_s)
    wpath = body_s['wpath']
    idx_target_curr = 1

    """generating tree roots"""
    rootdata = gen_tree_roots(data_mp0, wpath, idx_target_curr)
    if rootdata is None:
        warnings.WarningMessage('the initial pose is too close to the target. Terminates.')
        return None
    else:
        data_mp, idx_target_curr = rootdata
    outmps.append(data_mp)
    
    """expanding tree"""
    for d in range(1,MAX_DEPTH):
        leafdata=expand_tree(data_mp, wpath, idx_target_curr)
        if leafdata is not None:
            data_mp, idx_target_curr = leafdata
            outmps.append(data_mp)
        else:
            print('--goal is reached. Search terminates.')
            break
        print('--depth={}/{}, reach target={}/{}'.format(d,MAX_DEPTH, idx_target_curr,wpath.shape[0]-1))


    
    if save_file:
        """visualze rollout results in blender, for debugging"""
        save_rollout_results(wpath, outmps, 'results/tmp123/GAMMAVAEComboPolicy_PPO/{}'.format(args.cfg_policy), idx_seq=idx_seq)
    

    return None





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_policy', default='MPVAEPolicy_tmp',
                        help='specify the motion model and the policy config.')
    parser.add_argument('--max_depth', type=int, default=60,
                        help='the maximal number of (0.25-second) motion primitives in each motion.') 
    parser.add_argument('--ground_euler', nargs=3, type=float, default=[0, 0, 0],
                        help='the gorund plan rotation. Normally we set it to flat with Z-up Y-forward.') # which dataset to evaluate? choose only one
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
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

    cfg_policy = ConfigCreator(args.cfg_policy)
    cfg_1frame_male = cfg_policy.trainconfig['cfg_1frame_male']
    cfg_2frame_male = cfg_policy.trainconfig['cfg_2frame_male']
    cfg_1frame_female = cfg_policy.trainconfig['cfg_1frame_female']
    cfg_2frame_female = cfg_policy.trainconfig['cfg_2frame_female']
    body_repr = cfg_policy.modelconfig['body_repr']
    
    """global parameter"""
    REPROJ_FACTOR=0.5
    USE_POLICY_MEAN = False
    USE_POLICY = True
    SCENE_ORI='ZupYf' # the coordinate setting of the scene.

    MAX_DEPTH = args.max_depth
    GOAL_THRESH = 0.75
    N_GENS_ROOT = 1
    N_GENS_LEAF = 32 
    NUM_SEQ = 4 # the number of sequences to produce


    """data"""
    # ground data
    Rg = R.from_euler('xyz', np.array([0, 0, 0]), degrees=True)
    rotmat_g = Rg.as_matrix()
    # body motion data
    bm_path = config_env.get_body_model_path()
    batch_gen = BatchGeneratorFollowPathInCubes(dataset_path='exp_GAMMAPrimitive/data/Cubes/scene_cubes_000_navimesh.obj_traj',
                                                body_model_path=bm_path,
                                                scene_ori=SCENE_ORI)
    batch_gen.get_rec_list()



    """set GAMMA primitive networks"""
    genop_1frame_male = configure_model(cfg_1frame_male, args.gpu_index, args.random_seed)
    genop_1frame_female = configure_model(cfg_1frame_female, args.gpu_index, args.random_seed)
    genop_2frame_male = configure_model(cfg_2frame_male, args.gpu_index, args.random_seed)
    genop_2frame_female = configure_model(cfg_2frame_female, args.gpu_index, args.random_seed)

    """body model parsers"""
    pconfig_mp = {
        'n_batch':10*N_GENS_ROOT*N_GENS_LEAF,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_mp = SMPLXParser(pconfig_mp)

    pconfig_1f_root = {
        'n_batch':N_GENS_ROOT,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_1f_root = SMPLXParser(pconfig_1f_root)

    pconfig_mp_root = {
        'n_batch':10*N_GENS_ROOT,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_mp_root = SMPLXParser(pconfig_mp_root)



    """policy network and their optimizer"""
    policy_model = GAMMAPolicy(cfg_policy.modelconfig)
    policy_model.eval()
    policy_model.to(device)
    ckp_list = sorted(glob.glob(os.path.join(cfg_policy.trainconfig['save_dir'],'epoch-450.ckp')),
                        key=os.path.getmtime)
    # if len(ckp_list)>0:
    try:
        checkpoint = torch.load(ckp_list[-1], map_location=device)
        policy_model.load_state_dict(checkpoint['model_state_dict'])
        print('[INFO] --load checkpoint from {}'.format(ckp_list[-1]))
    except:
        print('[INFO] --use initial policy weights')

    """rollout"""
    ii = 0
    while ii < NUM_SEQ:
        data = batch_gen.next_body_cuda()
        with torch.no_grad():
            traj_ppo = rollout(data, save_file=True, idx_seq=ii)
            
        ii+=1
    