from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, glob
import numpy as np
from tqdm import tqdm
import torch
import smplx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
import json
import csv
import pdb
import pickle
from copy import deepcopy

import sys, os, pdb
sys.path.append(os.getcwd())
from os.path import join as ospj
import json
from collections import *
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.common import flatten
from exp_GAMMAPrimitive.utils.config_env import *

'''
In the AMASS dataset, all bodies are located in a bounding box, with
np.min(transall, axis=0) = array([-4.18054399, -3.86190773,  0.00487521])
np.max(transall, axis=0) = array([4.28418131, 4.64069242, 1.91651809])
, which are in terms of meters.
'''


def calc_calibrate_offset(body_mesh_model, betas, transl, pose):
    '''
    The factors to influence this offset is not clear. Maybe it is shape and pose dependent.
    Therefore, we calculate such delta_T for each individual body mesh.
    It takes a batch of body parameters
    input:
        body_params: dict, basically the input to the smplx model
        smplx_model: the model to generate smplx mesh, given body_params
    Output:
        the offset for params transform
    '''
    n_batches = transl.shape[0]
    bodyconfig = {}
    bodyconfig['body_pose'] = torch.FloatTensor(pose[:,3:]).cuda()
    bodyconfig['betas'] = torch.FloatTensor(betas).unsqueeze(0).repeat(n_batches,1).cuda()
    bodyconfig['transl'] = torch.zeros([n_batches,3], dtype=torch.float32).cuda()
    bodyconfig['global_orient'] = torch.zeros([n_batches,3], dtype=torch.float32).cuda()
    smplx_out = body_mesh_model(return_verts=True, **bodyconfig)
    delta_T = smplx_out.joints[:,0,:] # we output all pelvis locations
    delta_T = delta_T.detach().cpu().numpy() #[t, 3]

    return delta_T




def get_new_coordinate(body_mesh_model, betas, transl, pose):
    '''
    this function produces transform from body local coordinate to the world coordinate.
    it takes only a single frame.
    local coodinate:
        - located at the pelvis
        - x axis: from left hip to the right hip
        - z axis: point up (negative gravity direction)
        - y axis: pointing forward, following right-hand rule
    '''
    bodyconfig = {}
    bodyconfig['transl'] = torch.FloatTensor(transl)
    bodyconfig['global_orient'] = torch.FloatTensor(pose[:,:3])
    bodyconfig['body_pose'] = torch.FloatTensor(pose[:,3:])
    bodyconfig['betas'] = torch.FloatTensor(betas).unsqueeze(0)
    smplxout = body_mesh_model(**bodyconfig)
    joints = smplxout.joints.squeeze().detach().cpu().numpy()
    x_axis = joints[2,:] - joints[1,:]
    x_axis[-1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0,0,1])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis/np.linalg.norm(y_axis)
    global_ori_new = np.stack([x_axis, y_axis, z_axis], axis=1)
    transl_new = joints[:1,:] # put the local origin to pelvis

    return global_ori_new, transl_new




def get_body_model(type, gender, batch_size,device='cpu'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model_path = get_body_model_path()
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

## set mosh markers
## read the corresponding smplx verts indices as markers.
with open(os.path.join(get_body_marker_path(), 'CMU.json')) as f:
    marker_cmu_41 = list(json.load(f)['markersets'][0]['indices'].values())

with open(os.path.join(get_body_marker_path(), 'SSM2.json')) as f:
    marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())

bm_one_male = get_body_model('smplx', 'male', 1)
bm_one_female = get_body_model('smplx', 'female', 1)

def canonicalize_subsequence(seq, start_frame, end_frame):
    with open(seq, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    fps = data['mocap_framerate']
    assert fps == 120.0
    downsample_rate = 3
    len_subseq = (end_frame - start_frame) // downsample_rate
    ## define body model according to gender
    bm_batch_male = get_body_model('smplx', 'male', len_subseq, device='cuda')
    bm_batch_female = get_body_model('smplx', 'female', len_subseq, device='cuda')
    bodymodel_batch = bm_batch_male
    bodymodel_one = bm_one_male

    ## read data
    transl_all = data['pose_est_trans']
    pose_all = data['pose_est_fullposes']
    betas = data['shape_est_betas'][:10]
    ## break if remaining frames are not sufficient
    if transl_all.shape[0] <= end_frame:
        return None

    ## get subsequence and setup IO
    transl = deepcopy(transl_all[start_frame:end_frame:downsample_rate, :])
    pose = deepcopy(pose_all[start_frame:end_frame:downsample_rate, :])
    data_out = {}

    # print(start_frame, end_frame)
    ## perform transformation from the world coordinate to the amass coordinate
    ### get transformation from amass space to world space
    # print(betas[:10].shape, transl[:1, :].shape, pose[:1, :66].shape)
    transf_rotmat, transf_transl = get_new_coordinate(bodymodel_one, betas[:10], transl[:1, :], pose[:1, :66])
    ### calibrate offset
    delta_T = calc_calibrate_offset(bodymodel_batch, betas[:10], transl, pose[:, :66])
    ### get new global_orient
    global_ori = R.from_rotvec(pose[:, :3]).as_matrix()  # to [t,3,3] rotation mat
    global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat.T, global_ori)
    pose[:, :3] = R.from_matrix(global_ori_new).as_rotvec()
    ### get new transl
    transl = np.einsum('ij,tj->ti', transf_rotmat.T, transl + delta_T - transf_transl) - delta_T
    data_out['transf_rotmat'] = transf_rotmat
    data_out['transf_transl'] = transf_transl
    data_out['trans'] = transl
    data_out['poses'] = pose
    data_out['betas'] = betas
    data_out['gender'] = 'male'
    data_out['mocap_framerate'] = int(fps)

    ## under this new amass coordinate, extract the joints/markers' locations
    ## when get generated joints/markers, one can directly transform them back to world coord
    ## note that hand pose is not considered here. In amass, the hand pose is regularized.
    body_param = {}
    body_param['transl'] = torch.FloatTensor(transl).cuda()
    body_param['global_orient'] = torch.FloatTensor(pose[:, :3]).cuda()
    body_param['betas'] = torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(len_subseq, 1).cuda()
    body_param['body_pose'] = torch.FloatTensor(pose[:, 3:66]).cuda()
    smplxout = bodymodel_batch(return_verts=True, **body_param)
    ### extract joints and markers
    joints = smplxout.joints[:, :22, :].detach().squeeze().cpu().numpy()
    markers_41 = smplxout.vertices[:, marker_cmu_41, :].detach().squeeze().cpu().numpy()
    markers_67 = smplxout.vertices[:, marker_ssm_67, :].detach().squeeze().cpu().numpy()
    data_out['joints'] = joints
    data_out['marker_cmu_41'] = markers_41
    data_out['marker_ssm2_67'] = markers_67

    return data_out

def get_cats(ann, split):
    # Get sequence labels and frame labels if they exist
    seq_l, frame_l = [], []
    if 'extra' not in split:
        if ann['seq_ann'] is not None:
            seq_l = flatten([seg['act_cat'] for seg in ann['seq_ann']['labels']])
        if ann['frame_ann'] is not None:
            frame_l = flatten([seg['act_cat'] for seg in ann['frame_ann']['labels']])
    else:
        # Load all labels from (possibly) multiple annotators
        if ann['seq_anns'] is not None:
            seq_l = flatten([seg['act_cat'] for seq_ann in ann['seq_anns'] for seg in seq_ann['labels']])
        if ann['frame_anns'] is not None:
            frame_l = flatten([seg['act_cat'] for frame_ann in ann['frame_anns'] for seg in frame_ann['labels']])

    return list(seq_l), list(frame_l)

def get_seq_files(action='sit'):
    act_anns = defaultdict(list)  # { seq_id_1: [ann_1_1, ann_1_2], seq_id_2: [ann_2_1], ...}
    n_act_spans = 0
    dur = 0
    dur_frame = 0
    sids = []
    file_paths = []
    start_times = []
    end_times = []
    for spl in babel:
        for sid in babel[spl]:
            seq_l, frame_l = get_cats(babel[spl][sid], spl)
            # print(seq_l + frame_l)
            if 'frame_ann' in babel[spl][sid] and babel[spl][sid]['frame_ann'] is not None:
                for seg in babel[spl][sid]['frame_ann']['labels']:
                    if action in seg["act_cat"]:
                        dur_frame += seg['end_t'] - seg['start_t']

                        sids.append(sid)
                        file_path = os.path.join(*(babel[spl][sid]['feat_p'].split(os.path.sep)[1:]))
                        dataset_name = file_path.split(os.path.sep)[0]
                        if dataset_name in amass_dataset_rename_dict:
                            file_path = file_path.replace(dataset_name, amass_dataset_rename_dict[dataset_name])
                        file_path = file_path.replace('poses',
                                                      'stageii')  # file naming suffix changed in different amass versions
                        # replace space
                        file_path = file_path.replace(" ",
                                                      "_")  # set replace count to string length, so all will be replaced
                        file_paths.append(file_path)
                        start_times.append(seg['start_t'])
                        end_times.append(seg['end_t'])

            if action in seq_l + frame_l:
                # Store all relevant mocap sequence annotations
                act_anns[sid].append(babel[spl][sid])
                # # Individual spans of the action in the sequence
                n_act_spans += Counter(seq_l + frame_l)[action]
                dur += babel[spl][sid]['dur']

    print('# Seqs. containing action {0} = {1}'.format(action, len(act_anns)))
    print('# Segments containing action {0} = {1}'.format(action, n_act_spans))
    print('dur:', dur)
    print('dur_frame:', dur_frame)
    print('datasets:', set([file_path.split(os.path.sep)[0] for file_path in file_paths]))

    return sids, file_paths, start_times, end_times

# AMASS dataset names from website are slightly different from what used in BABEL
amass_dataset_rename_dict = {
    'ACCAD': 'ACCAD',
    'BMLmovi': 'BMLmovi',
    'BioMotionLab_NTroje': 'BMLrub',
    'MPI_HDM05': 'HDM05',
    'CMU': 'CMU',
    'Eyes_Japan_Dataset': 'EyesJapanDataset/Eyes_Japan_Dataset',
    'HumanEva': 'HumanEva',
    'TCD_handMocap': 'TCDHands',
    'KIT': 'KIT',
    'Transitions_mocap': 'Transitions',
}

if __name__=='__main__':
    hostname = socket.gethostname()
    base_folder = Path("data/amass")
    # load babel labels
    d_folder = babel_dir = base_folder / "babel_v1-0_release/babel_v1.0_release/"
    babel_set = l_babel_dense_files = ['train', 'val']
    # BABEL Dataset
    babel = {}
    for file in l_babel_dense_files:
        babel[file] = json.load(open(ospj(d_folder, file + '.json')))

    # canonicalize
    N_MPS = int(sys.argv[1])
    MP_FRAME = 10
    downsample_rate = 3
    len_subseq = int(MP_FRAME * N_MPS)
    bm_batch_male = get_body_model('smplx', 'male', len_subseq, device='cuda')
    bm_batch_female = get_body_model('smplx', 'female', len_subseq, device='cuda')
    #### set input output dataset paths
    raw_dataset_path = base_folder / 'smplx_g'
    if N_MPS > 1:
        result_smplx_path = 'data/DIMOS_mp/Canonicalized-MPx{:d}/data'.format(N_MPS)
    else:
        result_smplx_path = 'data/DIMOS_mp/Canonicalized-MP/data'
    subsets = [
               'walk',
               ]

    height_stat = {}
    #### main loop to each subset in AMASS
    for subset in subsets:
        sids, seqs, start_times, end_times = get_seq_files(subset)
        outfolder = os.path.join(result_smplx_path, subset + '_frame_label')
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        print('-- processing subset {:s}'.format(subset))
        index_subseq = 0 # index subsequences for subsets separately
        #### main loop to process each sequence
        for sid, seq, start_t, end_t in zip(tqdm(sids), seqs, start_times, end_times):
            print('processing:', sid, seq, start_t, end_t)
            seq = os.path.join(raw_dataset_path, seq)
            ## read data
            if os.path.basename(seq) == 'shape.npz':
                continue
            if not os.path.exists(seq):
                print(seq, ' not exist!')
                continue
            print('loading:', seq)
            data = dict(np.load(seq, allow_pickle=True))
            # print(data.keys())
            if not 'mocap_frame_rate' in data:
                continue
            fps = data['mocap_frame_rate']
            assert fps == 120.0
            ## define body model according to gender
            bm_batch_male = get_body_model('smplx', 'male', len_subseq, device='cuda')
            bm_batch_female = get_body_model('smplx', 'female', len_subseq, device='cuda')
            bodymodel_batch = bm_batch_male if str(data['gender'].astype(str)) == 'male' else bm_batch_female
            bodymodel_one = bm_one_male if str(data['gender'].astype(str)) == 'male' else bm_one_female

            ## read data and downsample
            transl_all = data['trans'][int(start_t * fps):int(end_t * fps) + 1][::downsample_rate]
            pose_all = data['poses'][int(start_t * fps):int(end_t * fps) + 1][::downsample_rate]
            betas = data['betas'][:10]

            ## skip too short sequences
            n_frames = transl_all.shape[0]
            if n_frames < len_subseq:
                continue

            # skip sequences with significant height elevation
            height = transl_all[:, 2]
            height_range = np.max(height) - np.min(height)
            height_stat['_'.join([str(sid), str(seq), str(start_t), str(end_t)])] = height_range
            if height_range > 0.1:
                print('height range too large:', height_range, '_'.join([str(sid), str(seq), str(start_t), str(end_t)]))
                continue

            t = 0
            while t < n_frames:
                ## get subsequence and setup IO
                outfilename = os.path.join(outfolder, 'subseq_{:07d}.npz'.format(index_subseq))
                transl = deepcopy(transl_all[t:t+len_subseq, :])
                pose = deepcopy(pose_all[t:t+len_subseq, :])
                data_out = {}

                ## break if remaining frames are not sufficient
                if transl.shape[0] < len_subseq:
                    break

                ## perform transformation from the world coordinate to the amass coordinate
                ### get transformation from amass space to world space
                transf_rotmat, transf_transl = get_new_coordinate(bodymodel_one, betas[:10], transl[:1,:], pose[:1,:66])
                ### calibrate offset
                delta_T = calc_calibrate_offset(bodymodel_batch, betas[:10], transl, pose[:,:66])
                ### get new global_orient
                global_ori = R.from_rotvec(pose[:,:3]).as_matrix() # to [t,3,3] rotation mat
                global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat.T, global_ori)
                pose[:,:3] = R.from_matrix(global_ori_new).as_rotvec()
                ### get new transl
                transl = np.einsum('ij,tj->ti', transf_rotmat.T, transl+delta_T-transf_transl)-delta_T
                data_out['transf_rotmat'] = transf_rotmat
                data_out['transf_transl'] = transf_transl
                data_out['trans'] = transl
                data_out['poses'] = pose
                data_out['betas'] = betas
                data_out['gender'] = data['gender'].astype(str)
                data_out['mocap_framerate'] = data['mocap_frame_rate']

                ## under this new amass coordinate, extract the joints/markers' locations
                ## when get generated joints/markers, one can directly transform them back to world coord
                ## note that hand pose is not considered here. In amass, the hand pose is regularized.
                body_param = {}
                body_param['transl'] = torch.FloatTensor(transl).cuda()
                body_param['global_orient'] = torch.FloatTensor(pose[:,:3]).cuda()
                body_param['betas'] = torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(len_subseq,1).cuda()
                body_param['body_pose'] = torch.FloatTensor(pose[:, 3:66]).cuda()
                smplxout = bodymodel_batch(return_verts=True, **body_param)
                ### extract joints and markers
                joints = smplxout.joints[:,:22,:].detach().squeeze().cpu().numpy()
                markers_41 = smplxout.vertices[:,marker_cmu_41,:].detach().squeeze().cpu().numpy()
                markers_67 = smplxout.vertices[:,marker_ssm_67,:].detach().squeeze().cpu().numpy()
                data_out['joints'] = joints
                data_out['marker_cmu_41'] = markers_41
                data_out['marker_ssm2_67'] = markers_67

                np.savez(outfilename, **data_out)
                t = t+len_subseq
                # t = t + len_subseq if N_MPS == 1 else t + MP_FRAME
                index_subseq = index_subseq + 1

    with open(os.path.join(result_smplx_path, 'height_stat.json'), 'w') as f:
        json.dump({'height_stat': height_stat}, f)