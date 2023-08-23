import pickle

import smplx
import torch

from pigraph_config import *


def smplx_forward(seq_data):
    T = len(seq_data['transl'])

    body_model = smplx.create(smplx_model_folder, model_type='smplx',
                              gender=seq_data['gender'], ext='npz',
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
                              batch_size=T,
                              )
    torch_param = {}
    torch_param['betas'] = torch.tensor(seq_data['betas']).repeat(T, 1)
    torch_param['global_orient'] = torch.tensor(seq_data['global_orient'])
    torch_param['transl'] = torch.tensor(seq_data['transl'])
    torch_param['left_hand_pose'] = torch.tensor(seq_data['left_hand_pose'])
    torch_param['right_hand_pose'] = torch.tensor(seq_data['right_hand_pose'])
    torch_param['jaw_pose'] = torch.tensor(seq_data['jaw_pose'])
    torch_param['leye_pose'] = torch.tensor(seq_data['leye_pose'])
    torch_param['reye_pose'] = torch.tensor(seq_data['reye_pose'])
    torch_param['expression'] = torch.tensor(seq_data['expression'])
    torch_param['body_pose'] = torch.tensor(seq_data['body_pose'])
    smplx_output = body_model(return_verts=True, return_full_pose=True, **torch_param)

    return smplx_output, body_model

#load human body from given file and fit with smplx
def load_human(human_path):
    with open(human_path, 'rb') as human_file:
        human_params = pickle.load(human_file)
    T = len(human_params['transl'])#number of frames

    body_model = smplx.create(smplx_model_folder, model_type='smplx',
                              gender='male', ext='npz',
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
                              batch_size=T
                              )

    torch_param = {}
    torch_param['betas'] = torch.tensor(human_params['betas']).repeat(T, 1)
    torch_param['global_orient'] = torch.tensor(human_params['global_orient'])
    torch_param['transl'] = torch.tensor(human_params['transl'])
    torch_param['left_hand_pose'] = torch.tensor(human_params['left_hand_pose'])
    torch_param['right_hand_pose'] = torch.tensor(human_params['right_hand_pose'])
    torch_param['jaw_pose'] = torch.tensor(human_params['jaw_pose'])
    torch_param['leye_pose'] = torch.tensor(human_params['leye_pose'])
    torch_param['reye_pose'] = torch.tensor(human_params['reye_pose'])
    torch_param['expression'] = torch.tensor(human_params['expression'])
    torch_param['body_pose'] = torch.tensor(human_params['body_pose'])

    smplx_output = body_model(return_verts=True, return_full_pose=True, **torch_param)
    # vertices = smplx_output.vertices.detach().cpu().numpy()  # [n_frames, 10475, 3]
    # joints = smplx_output.joints.detach().cpu().numpy()  # [n_frames, 127, 3]
    return smplx_output, body_model

# load human body in a given sequence and fit with smplx
def load_sequence(recording_name, start_frame, end_frame):
    human_path = os.path.join(human_folder, recording_name, 'results.pkl')
    with open(human_path, 'rb') as human_file:
        human_params = pickle.load(human_file)
    T = end_frame - start_frame + 1 #number of frames

    body_model = smplx.create(smplx_model_folder, model_type='smplx',
                              gender='male', ext='npz',
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
                              batch_size=T
                              )

    torch_param = {}
    torch_param['betas'] = torch.tensor(human_params['betas']).repeat(T, 1)
    torch_param['global_orient'] = torch.tensor(human_params['global_orient'][start_frame - 1:end_frame])
    torch_param['transl'] = torch.tensor(human_params['transl'][start_frame - 1:end_frame])
    torch_param['left_hand_pose'] = torch.tensor(human_params['left_hand_pose'][start_frame - 1:end_frame])
    torch_param['right_hand_pose'] = torch.tensor(human_params['right_hand_pose'][start_frame - 1:end_frame])
    torch_param['jaw_pose'] = torch.tensor(human_params['jaw_pose'][start_frame - 1:end_frame])
    torch_param['leye_pose'] = torch.tensor(human_params['leye_pose'][start_frame - 1:end_frame])
    torch_param['reye_pose'] = torch.tensor(human_params['reye_pose'][start_frame - 1:end_frame])
    torch_param['expression'] = torch.tensor(human_params['expression'][start_frame - 1:end_frame])
    torch_param['body_pose'] = torch.tensor(human_params['body_pose'][start_frame - 1:end_frame])

    smplx_output = body_model(return_verts=True, return_full_pose=True, **torch_param)
    # vertices = smplx_output.vertices.detach().cpu().numpy()  # [n_frames, 10475, 3]
    # joints = smplx_output.joints.detach().cpu().numpy()  # [n_frames, 127, 3]
    return smplx_output, body_model
