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

sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorFollowPathInCubes
from exp_GAMMAPrimitive.utils import config_env

from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP
from models.models_policy import GAMMAPolicy
from models.baseops import SMPLXParser
from models.searchop import MPTNode, MinHeap
from models.baseops import get_logger


sys.setrecursionlimit(10000) # if too small, deepcopy will reach the maximal depth limit.

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
                                        states.permute([1,0,2]),
                                        bparam_seed.transpose([1,0,2]),
                                        prev_betas,
                                        n_gens=n_gens,
                                        param_blending=True,
                                        use_policy_mean=USE_POLICY_MEAN,
                                        use_policy=USE_POLICY
                                        )
    pred_markers = np.reshape(pred_markers, (pred_markers.shape[0],
                                    pred_markers.shape[1], -1,3))#[t, n_gens, V, 3]
    return pred_markers.transpose([1,0,2,3]), pred_params.transpose([1,0,2]), act


def canonicalize_static_pose(data):
    smplx_transl = data['transl']
    smplx_glorot = data['global_orient']
    smplx_poses = data['body_pose']
    gender=data['gender']
    betas = data['betas']
    smplx_handposes = np.zeros([smplx_transl.shape[0], 24])
    prev_params = np.concatenate([smplx_transl,smplx_glorot,
                                    smplx_poses,smplx_handposes],
                                    axis=-1) #[t,d]
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
                                    xb=body_param_seed)[:, 0]#[t, 3]
    return marker_seed, body_param_seed[None,...], R0, T0, pelvis_loc




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
    pt_wpath_l_3d = np.einsum('ij,tj->ti', R0[0].T, pt_wpath-T0[0])

    '''extract path feature = normalized direction + unnormalized height'''
    fea_wpathxy = pt_wpath_l_3d[:,:2]-pel[:,:2]
    dist_xy = np.linalg.norm(fea_wpathxy, axis=-1, keepdims=True)
    fea_wpathxy = fea_wpathxy/dist_xy
    fea_wpathz = pt_wpath_l_3d[:,-1:]-pel[:,-1:]
    fea_wpath = np.concatenate([fea_wpathxy, fea_wpathz], axis=-1)

    '''extract marker feature'''
    fea_marker = pt_wpath_l_3d[:,None,:]-Y_l
    dist_m_3d = np.linalg.norm(fea_marker, axis=-1, keepdims=True)
    fea_marker_3dn = (fea_marker/dist_m_3d).reshape(nt, -1)

    '''extract marker feature with depth'''
    fea_marker_xy = fea_marker[:,:,:2]
    dist_m_2d = np.linalg.norm(fea_marker_xy, axis=-1, keepdims=True)
    fea_marker_xy = fea_marker_xy/dist_m_2d
    fea_marker_h = np.concatenate([fea_marker_xy, fea_marker[:,:,-1:]],axis=-1).reshape(nt, -1)

    return dist_xy, fea_wpath, fea_marker_3dn, fea_marker_h


def update_local_target(marker, pelvis_loc, curr_target_wpath, R0, T0, wpath):
    idx_target_curr, pt_target_curr = curr_target_wpath
    dist, fea_wpath, fea_marker, fea_marker_h = get_wpath_feature(marker, pelvis_loc, R0, T0, wpath[idx_target_curr][None,...])
    while np.any(dist<GOAL_THRESH) and idx_target_curr < len(wpath)-1:
        idx_target_curr = idx_target_curr+1
        dist, fea_wpath, fea_marker, fea_marker_h = get_wpath_feature(marker, pelvis_loc, R0, T0, wpath[idx_target_curr][None,...])
    return idx_target_curr, fea_wpath, fea_marker, fea_marker_h





def gen_tree_roots(start_node, wpath):
    mp_heap = MinHeap()

    # canonicalize the starting pose
    marker_seed = start_node.data['markers']
    body_param_seed = start_node.data['smplx_params']
    R0 = start_node.data['transf_rotmat']
    T0 = start_node.data['transf_transl']
    motion_model = genop_1frame_male if start_node.data['gender']=='male' else genop_1frame_female
    prev_betas = torch.FloatTensor(start_node.data['betas']).unsqueeze(0).to(device)
    gender = str(start_node.data['gender'])
    pelvis_loc = start_node.data['pelvis_loc']

    ## retrieve current target and update it
    idx_target_curr, fea_wpath, fea_marker, fea_marker_h = update_local_target(marker_seed, pelvis_loc,
                                    start_node.data['curr_target_wpath'],
                                    R0, T0, wpath)
    if body_repr == 'ssm2_67_condi_marker':
        states = torch.cuda.FloatTensor(np.concatenate([marker_seed.reshape([1,-1]), fea_marker],axis=-1),
                                    device=device)[None,...]
    else:
        raise NotImplementedError


    # generate markers and regress to body params
    pred_markers, pred_params, pred_latent = gen_motion_one_step(motion_model,
                                            states, body_param_seed, prev_betas, t_his=1)
    for ii in range(pred_markers.shape[0]):
        joints = smplxparser_mp.get_jts(betas=start_node.data['betas'],
                                            gender=gender,
                                            xb=pred_params[ii])
        pelvis_loc = joints[:,0]#[t, 3]
        pred_markers_proj = smplxparser_mp.get_markers(betas=start_node.data['betas'],
                                            gender=gender,
                                            xb=pred_params[ii]).reshape((pred_params.shape[1], -1, 3))  #[t, p, 3]
        rootnode = MPTNode(gender, start_node.data['betas'], R0, T0, pelvis_loc, joints,
                            pred_markers[ii:ii+1], pred_markers_proj, pred_params[ii:ii+1], pred_latent[ii:ii+1], '1-frame',
                            timestamp=0, curr_target_wpath=(idx_target_curr, wpath[idx_target_curr]) )
        if HARD_CONTACT:
            rootnode.evaluate_quality_hard_contact_wpath(terrian_rotmat=rotmat_g, wpath=wpath)
        else:
            rootnode.evaluate_quality_soft_contact_wpath(terrian_rotmat=rotmat_g, wpath=wpath)
        if rootnode.quality != 0:
            mp_heap.push(rootnode)
    return mp_heap



def expand_tree(mp_heap_prev, wpath, max_depth=10):
    mp_heap_curr = MinHeap()
    # generate child treenodes
    for iop in range(0, max_depth):
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
            body_param_seed = prev_params[0,-t_his:]
            ## move frame to the second last body's pelvis
            R_, T_ = smplxparser_1frame.get_new_coordinate(
                                                betas=prev_betas,
                                                gender=prev_gender,
                                                xb=body_param_seed[:1])
            T0 = np.einsum('bij,bpj->bpi', prev_rotmat, T_)+prev_transl
            R0 = np.einsum('bij,bjk->bik', prev_rotmat, R_)
            ## get the last body param and marker in the new coordinate
            body_param_seed = smplxparser_2frame.update_transl_glorot(
                                                    np.tile(R_, (t_his,1,1)), 
                                                    np.tile(T_, (t_his,1,1)),
                                                    betas=prev_betas,
                                                    gender=prev_gender,
                                                    xb=body_param_seed)

            ## blend predicted markers and the reprojected markers to eliminated jitering
            marker_seed_rproj = smplxparser_2frame.get_markers(
                                            betas=prev_betas,
                                            gender=prev_gender,
                                            xb=body_param_seed).reshape([t_his,-1])
            pred_markers = np.einsum('ij, tnj->tni', R_[0].T, prev_markers[0,-t_his:]-T_).reshape([t_his, -1])
            marker_seed = REPROJ_FACTOR*marker_seed_rproj + (1-REPROJ_FACTOR)*pred_markers
            pel_loc_seed = np.einsum('ij,tj->ti',R_[0].T, prev_pel_loc[-t_his:]-T_[0])
            idx_target_curr, fea_wpath, fea_marker, fea_marker_h = update_local_target(marker_seed, pel_loc_seed,
                                                    mp_prev.data['curr_target_wpath'],
                                                    R0, T0, wpath)
            if body_repr == 'ssm2_67_condi_marker':
                states = torch.cuda.FloatTensor(np.concatenate([marker_seed, fea_marker],axis=-1),
                                    device=device)[None,...]
            else:
                raise NotImplementedError


            '''generate future motions'''
            motion_model = genop_2frame_male if prev_gender=='male' else genop_2frame_female
            body_param_seed = body_param_seed[None,...]
            prev_betas_torch = torch.FloatTensor(prev_betas).unsqueeze(0).to(device)

            pred_markers, pred_params, pred_latent = gen_motion_one_step(motion_model, states,
                                body_param_seed,
                                prev_betas_torch, t_his) #smplx [n_gens, n_frames, d]

            '''sort generated primitives'''
            for ii in range(pred_markers.shape[0]):
                joints = smplxparser_mp.get_jts(betas=prev_betas,
                                                gender=prev_gender,
                                                xb=pred_params[ii])
                pelvis_loc = joints[:, 0]#[t, 3]
                pred_markers_proj = smplxparser_mp.get_markers(betas=prev_betas,
                                        gender=prev_gender,
                                        xb=pred_params[ii]).reshape((pred_params.shape[1], -1, 3))  #[t, p, 3]
                mp_curr = MPTNode(prev_gender, prev_betas, R0, T0, pelvis_loc, joints,
                            pred_markers[ii:ii+1], pred_markers_proj, pred_params[ii:ii+1], pred_latent[ii:ii+1], '2-frame',
                            timestamp=iop, curr_target_wpath=(idx_target_curr,wpath[idx_target_curr])  )
                if HARD_CONTACT:
                    mp_curr.evaluate_quality_hard_contact_wpath(terrian_rotmat=rotmat_g, wpath=wpath)
                else:
                    mp_curr.evaluate_quality_soft_contact_wpath(terrian_rotmat=rotmat_g, wpath=wpath)
                if mp_curr.quality != 0:
                    mp_curr.set_parent(mp_prev)
                    mp_heap_curr.push(mp_curr)

            ## if all children is 0 quality with 2frame model, switch to 1 frame model and continue
            if mp_heap_curr.len() == 0:
                motion_model = genop_1frame_male if prev_gender=='male' else genop_1frame_female
                states = states[:,:1]
                body_param_seed = body_param_seed[:,:1]
                pred_markers, pred_params, pred_latent = gen_motion_one_step(motion_model, states,
                                                                            body_param_seed,
                                                                            prev_betas_torch, t_his-1) #smplx [n_gens, n_frames, d]
                for ii in range(pred_markers.shape[0]):
                    pelvis_loc = smplxparser_mp.get_jts(betas=prev_betas,
                                                gender=prev_gender,
                                                data=pred_params[ii])[:, 0]#[t, 3]
                    pred_markers_proj = smplxparser_mp.get_markers(betas=prev_betas,
                                            gender=prev_gender,
                                            params=pred_params[ii]).reshape((pred_params.shape[1], -1, 3))  #[t, p, 3]
                    mp_curr = MPTNode(prev_gender, prev_betas, R0, T0, pelvis_loc,
                                pred_markers[ii:ii+1], pred_markers_proj, pred_params[ii:ii+1], pred_latent[ii:ii+1], '1-frame',
                                                    timestamp=iop, curr_target_wpath=(idx_target_curr,wpath[idx_target_curr]))
                    if HARD_CONTACT:
                        mp_curr.evaluate_quality_hard_contact_wpath(terrian_rotmat=rotmat_g, wpath=wpath)
                    else:
                        mp_curr.evaluate_quality_soft_contact_wpath(terrian_rotmat=rotmat_g, wpath=wpath)
                    if mp_curr.quality != 0:
                        mp_curr.set_parent(mp_prev)
                        mp_heap_curr.push(mp_curr)

        if mp_heap_curr.len() == 0:
            log_and_print('[INFO] |--no movements searched. Program terminates.')
            return None
            # sys.exit()
        log_and_print('[INFO] |--valid MPs={}, dist_to_target={:.2f}, path_finished={}/{}, dist_to_target_curr={:.2f}, dist_to_gp={:.2f}, dist_to_skat={:.2f}'.format(
                    mp_heap_curr.len(),
                    mp_heap_curr.data[0].dist2target,
                    mp_heap_curr.data[0].data['curr_target_wpath'][0],len(wpath),
                    mp_heap_curr.data[0].dist2target_curr,
                    mp_heap_curr.data[0].dist2g, mp_heap_curr.data[0].dist2skat))
        mp_heap_prev.clear()
        mp_heap_prev = copy.deepcopy(mp_heap_curr)
        mp_heap_curr.clear()
        ## if the best node is close enough to the target, search stops and return
        if np.abs(mp_heap_prev.data[0].dist2target) < GOAL_THRESH:
            log_and_print('[INFO] |--find satisfactory solutions. Search finishes.')
            return mp_heap_prev

    return mp_heap_prev




def gen_motion(body_s, max_depth=10):
    '''
    idx: the sequence seed index, which is to name the output file actually.
    max_depth: this determines how long the generated motion is (0.25sec per motion prmitive)
    '''
    # specify the start node
    [marker_start, body_param_start,
        R_start, T_start, pelvis_loc_start] = canonicalize_static_pose(body_s)
    wpath = body_s['wpath']
    start_node = MPTNode(str(body_s['gender']), body_s['betas'], R_start, T_start, pelvis_loc_start, None,
                            marker_start, marker_start, body_param_start, None, 'start-frame',
                            timestamp=-1,
                            curr_target_wpath=(1, wpath[1])
                            )
    # depending on a static pose, generate a list of tree roots
    log_and_print('[INFO] generate roots in a heap')
    mp_heap_prev = gen_tree_roots(start_node, wpath)
    log_and_print('[INFO] |--valid MPs={}'.format(mp_heap_prev.len()))
    if mp_heap_prev.len() == 0:
        log_and_print('[INFO] |--no movements searched. Program terminates.')
        return

    # generate tree leaves
    mp_heap_prev=expand_tree(mp_heap_prev, wpath, max_depth=max_depth)
    if mp_heap_prev is None:
        return
    output = {'motion':None, 'wpath':wpath}
    log_and_print('[INFO] save results...')
    mp_leaves = mp_heap_prev
    motion_idx = 0
    while not mp_leaves.is_empty():
        if motion_idx >=1:
            break
        gen_results = []
        mp_leaf = mp_leaves.pop()
        gen_results.append(mp_leaf.data)
        while mp_leaf.parent is not None:
            gen_results.append(mp_leaf.parent.data)
            mp_leaf = mp_leaf.parent
        gen_results.reverse()
        output['motion'] = gen_results
        ### save to file
        outfilename = 'results_{}_{}.pkl'.format(body_repr, motion_idx)
        outfilename_f = os.path.join(outfoldername, outfilename)
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_policy', default='MPVAEPolicy_v0',
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

    """global parameter"""
    random_seed = args.random_seed
    n_gens_1frame = 16     # the number of primitives to generate from a single-frame motion seed
    n_gens_2frame = 4      # the nunber of primitives to generate from a two-frame motion seed
    max_nodes_to_expand = 4 # in the tree search, how many nodes to expand at the same level.
    GOAL_THRESH = 0.75  # the threshold to reach the goal.
    HARD_CONTACT=False  # for ranking the primitives in the tree search. If True, then motion primitives with implausible foot-ground contact are discarded.
    USE_POLICY_MEAN = False # only use the mean of the policy. If False, random samples are drawn from the policy.
    USE_POLICY = True # If False, random motion generation will be performed.
    SCENE_ORI='ZupYf' # the coordinate setting of the scene.
    max_depth = args.max_depth
    NUM_SEQ = 10 # the number of sequences to produce


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

    policy_model = GAMMAPolicy(cfg_policy.modelconfig)
    policy_model.eval()
    policy_model.to(device)
    ckp_list = sorted(glob.glob(os.path.join(cfg_policy.trainconfig['save_dir'],'epoch-500.ckp')),
                        key=os.path.getmtime)
    if len(ckp_list)>0:
        ckptfile = os.path.join(ckp_list[-1])
        checkpoint = torch.load(ckptfile, map_location=device)
        policy_model.load_state_dict(checkpoint['model_state_dict'])
        print('[INFO] --load checkpoint from {}'.format(ckptfile))


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


    """main block for motion generation"""
    resultdir = 'results/tmp222/GAMMAVAEComboPolicy_PPO_demo/{}'.format(args.cfg_policy)

    idx_seq = 0
    while idx_seq < NUM_SEQ:
        data = batch_gen.next_body()
        outfoldername = '{}/randseed{:03d}_seq{:03d}_{}/'.format(resultdir, random_seed,
                                                        idx_seq,data['wpath_filename'])
        if not os.path.exists(outfoldername):
            os.makedirs(outfoldername)
        logger = get_logger(outfoldername, mode='eval')
        log_and_print('[INFO] generate sequence {:d}'.format(idx_seq))

        gen_motion(data, max_depth=max_depth)
        idx_seq += 1