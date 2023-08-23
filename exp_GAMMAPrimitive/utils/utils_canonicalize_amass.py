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


import sys

if __name__=='__main__':
    N_MPS = int(sys.argv[1])
    MP_FRAME = 10
    #### set input output dataset paths
    amass_dataset_path = '/home/kaizhao/dataset/amass/smplx_g'
    if N_MPS > 1:
        amass_smplx_path = '/home/kaizhao/dataset/amass/AMASS-Canonicalized-locomotion-MPx{:d}/data'.format(N_MPS)
    else:
        amass_smplx_path = '/home/kaizhao/dataset/amass/AMASS-Canonicalized-locomotion-MP/data'
    amass_subsets = ['CMU',
                     # 'MPI_HDM05',
                     'BMLmovi', 'KIT', 'Eyes_Japan_Dataset']
    amass_subsets = ['HumanEva']


    ## set mosh markers
    ## read the corresponding smplx verts indices as markers.
    with open('/home/kaizhao/dataset/models_smplx_v1_1/models/markers/CMU.json') as f:
            marker_cmu_41 = list(json.load(f)['markersets'][0]['indices'].values())

    with open('/home/kaizhao/dataset/models_smplx_v1_1/models/markers/SSM2.json') as f:
            marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())

    bm_one_male = get_body_model('smplx','male',1)
    bm_one_female = get_body_model('smplx','female',1)


    #### main loop to each subset in AMASS
    for subset in amass_subsets:
        # if not subset in ['HumanEva']:
        #     continue
        seqs = glob.glob(os.path.join(amass_dataset_path, subset, '*/*.npz'))

        outfolder = os.path.join(amass_smplx_path, subset)
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        print('-- processing subset {:s}'.format(subset))

        index_subseq = 0 # index subsequences for subsets separately
        #### main loop to process each sequence
        for seq in tqdm(seqs):
            ## read data
            if os.path.basename(seq) == 'shape.npz':
                continue
            print('loading:', seq)
            data = dict(np.load(seq, allow_pickle=True))
            # print(data.keys())
            if not 'mocap_frame_rate' in data:
                continue
            fps = data['mocap_frame_rate']
            len_subseq = int(MP_FRAME*N_MPS)
            ## define body model according to gender
            bm_batch_male = get_body_model('smplx','male',len_subseq,device='cuda')
            bm_batch_female = get_body_model('smplx','female',len_subseq,device='cuda')
            bodymodel_batch = bm_batch_male if str(data['gender'].astype(str)) =='male' else bm_batch_female
            bodymodel_one = bm_one_male if str(data['gender'].astype(str)) =='male' else bm_one_female

            ## read data
            transl_all = data['trans']
            pose_all = data['poses']
            betas = data['betas']

            ## skip too short sequences
            n_frames = transl_all.shape[0]
            if n_frames < len_subseq:
                continue

            t = 0
            while t < n_frames:
                ## get subsequence and setup IO
                outfilename = os.path.join(outfolder, 'subseq_{:05d}.npz'.format(index_subseq))
                transl = transl_all[t:t+len_subseq, :]
                pose = pose_all[t:t+len_subseq, :]
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
                index_subseq = index_subseq+1