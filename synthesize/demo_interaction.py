import os
import pickle
import sys

import numpy as np
import torch
import glob

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as R
from test_navmesh import *
from exp_GAMMAPrimitive.utils.environments import *
from exp_GAMMAPrimitive.utils import config_env

np.random.seed(233)
torch.manual_seed(233)

def params2torch(params, dtype=torch.float32):
    return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}
def params2numpy(params):
    return {k: v.detach().cpu().numpy() if type(v) == torch.Tensor else v for k, v in params.items()}

bm_path = config_env.get_body_model_path()
bm = smplx.create(bm_path, model_type='smplx',
                                    gender='neutral', ext='npz',
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
                                    batch_size=1
                                    ).eval().cuda()

def demo_sit():
    scene_dir = Path('./data/test_room')
    action_in = 'sit on'
    action_out = action_in.split(' ')[0]
    floor_height = 0
    scene_path = scene_dir / 'room.ply'
    scene_name = 'test_room'
    obj_name = 'sofa'  # choose from ['sofa', 'chair']
    mesh_path = scene_dir / f'{obj_name}.ply'
    sdf_path = scene_dir / f'{obj_name}_sdf_gradient.pkl'

    load_goal_interaction = True
    if load_goal_interaction:
        interaction_path_dir = Path('./data/test_room') / f'{obj_name}_{action_in}'
    else:
        command = (f'python synthesize/coins_sample.py --exp_name test --lr_posa 0.01 --max_step_body 100  '
                   f'--weight_penetration 10 --weight_pose 10 --weight_init 0  --weight_contact_semantic 1 '
                   f'--num_sample 8 --num_try 8  --visualize 1 --full_scene 1 '
                   f'--action \'{action_in}\' --obj_path \'{mesh_path}\' --obj_category \'{obj_name}\' '
                   f'--obj_id 0 --scene_path \'{scene_path}\' --scene_name \'{scene_name}\'')
        print(command)
        os.system(command)
        interaction_path_dir = Path(
            'results/coins/two_stage/test/optimization_after_get_body') / action_in / f'{action_in}_{obj_name}_0/'
    interaction_path_list = list(interaction_path_dir.glob('*.pkl'))
    interaction_path_list = [p for p in interaction_path_list if p.name != 'results.pkl']
    if len(interaction_path_list) == 0:
        print('no interaction goal found')
        exit()

    """sit down"""
    num_seq = 4
    for seq_id in range(num_seq):
        target_interaction_path = random.choice(interaction_path_list)
        interaction_name = 'inter_' + obj_name + '_' + action_out
        target_point_path = Path('results', 'tmp', scene_name, interaction_name, 'target_point.pkl')
        target_point_path.parent.mkdir(exist_ok=True, parents=True)
        target_body_path = Path('results', 'tmp', scene_name, interaction_name, 'target_body.pkl')

        with open(target_interaction_path, 'rb') as f:
            print('load interaction at:', target_interaction_path)
            target_interaction = pickle.load(f)
        smplx_params = target_interaction['smplx_param']
        if 'left_hand_pose' in smplx_params:
            del smplx_params['left_hand_pose']
        if 'right_hand_pose' in smplx_params:
            del smplx_params['right_hand_pose']
        smplx_params['transl'][:, 2] -= floor_height
        smplx_params['gender'] = 'male'
        with open(target_body_path, 'wb') as f:
            pickle.dump(smplx_params, f)

        smplx_params = params2torch(smplx_params)
        pelvis = bm(**smplx_params).joints[0, 0, :].detach().cpu().numpy()
        r = torch.cuda.FloatTensor(1).uniform_() * 0.4 + 0.6
        # r = 0.6
        theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
        forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
        forward_dir[2] = 0
        forward_dir = forward_dir / torch.norm(forward_dir)
        random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
                                                                 convention="XYZ")
        forward_dir = torch.matmul(random_rot, forward_dir)
        target_point = pelvis + (forward_dir * r).detach().cpu().numpy()
        target_point[2] = 0

        with open(target_point_path, 'wb') as f:
            pickle.dump(target_point, f)

        seq_name = interaction_name + '_down_' + str(seq_id)
        command = "python synthesize/gen_interaction_unify.py --goal_thresh_final -1 --max_depth 15 --num_gen1 128 --num_gen2 32 --num_expand 4 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_sit_marker/sit_1frame " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                  "--target_body_path {} --interaction_name {} --start_point_path {} " \
                  "--use_zero_pose 0 --weight_target_dist 1 " \
                  "--visualize 0".format(seq_id, scene_path, scene_name, sdf_path, mesh_path, floor_height,
                                         target_body_path,
                                         seq_name, target_point_path)
        print(command)
        os.system(command)

        last_motion_path = f'results/interaction/{scene_name}/{seq_name}/MPVAEPolicy_sit_marker/sit_1frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'
        """stand up"""
        command = "python synthesize/gen_interaction_unify.py --goal_thresh_final 0.3 --max_depth 10 --num_gen1 128 --num_gen2 16 --num_expand 8 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_sit_marker/sit_1frame " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                  "--target_point_path {} --interaction_name {} --last_motion_path {} " \
                  "--use_zero_pose 0 --weight_target_dist 1 " \
                  "--visualize 0".format(seq_id, scene_path, scene_name, sdf_path, mesh_path, floor_height,
                                         target_point_path, interaction_name + '_up_' + str(seq_id), last_motion_path)
        print(command)
        os.system(command)


def get_forward_dir(interaction_path_dir):
    interaction_path_list = list(interaction_path_dir.glob('*.pkl'))
    interaction_path_list = [p for p in interaction_path_list if p.name != 'results.pkl']
    target_interaction_path = random.choice(interaction_path_list)
    with open(target_interaction_path, 'rb') as f:
        print('load interaction at:', target_interaction_path)
        target_interaction = pickle.load(f)
    smplx_params = target_interaction['smplx_param']
    if 'left_hand_pose' in smplx_params:
        del smplx_params['left_hand_pose']
    if 'right_hand_pose' in smplx_params:
        del smplx_params['right_hand_pose']
    smplx_params['transl'][:, 2] -= floor_height
    smplx_params['gender'] = 'male'
    body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
    forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]

    return forward_dir

if __name__ == "__main__":
    scene_dir = Path('./data/test_room')
    action_in = 'sit on' # choose from ['sit on', 'lie on'], 'lie on' can not be applied to the 'chair'
    action_out = action_in.split(' ')[0]
    floor_height = 0
    scene_path = scene_dir / 'room.ply'
    scene_name = 'test_room'
    obj_name = 'sofa'  # choose from ['sofa', 'chair']
    mesh_path = scene_dir / f'{obj_name}.ply'
    sdf_path = scene_dir / f'{obj_name}_sdf_gradient.pkl'

    load_goal_interaction = True
    if load_goal_interaction:
        interaction_path_dir = Path('./data/test_room') / f'{obj_name}_{action_in}'
    else:
        command = (f'python synthesize/coins_sample.py --exp_name test --lr_posa 0.01 --max_step_body 100  '
                   f'--weight_penetration 10 --weight_pose 10 --weight_init 0  --weight_contact_semantic 1 '
                   f'--num_sample 8 --num_try 8  --visualize 1 --full_scene 1 '
                   f'--action \'{action_in}\' --obj_path \'{mesh_path}\' --obj_category \'{obj_name}\' '
                   f'--obj_id 0 --scene_path \'{scene_path}\' --scene_name \'{scene_name}\'')
        print(command)
        os.system(command)
        interaction_path_dir = Path(
            f'results/coins/two_stage/{scene_name}/test/optimization_after_get_body') / action_in / f'{action_in}_{obj_name}_0/'
    interaction_path_list = list(interaction_path_dir.glob('*.pkl'))
    interaction_path_list = [p for p in interaction_path_list if p.name != 'results.pkl']
    if len(interaction_path_list) == 0:
        print('no interaction goal found')
        exit()

    """sit down"""
    num_seq = 4
    for seq_id in range(num_seq):
        target_interaction_path = random.choice(interaction_path_list)
        interaction_name = 'inter_' + obj_name + '_' + action_out
        target_point_path = Path('results', 'tmp', scene_name, interaction_name, 'target_point.pkl')
        target_point_path.parent.mkdir(exist_ok=True, parents=True)
        target_body_path = Path('results', 'tmp', scene_name, interaction_name, 'target_body.pkl')

        with open(target_interaction_path, 'rb') as f:
            print('load interaction at:', target_interaction_path)
            target_interaction = pickle.load(f)
        smplx_params = target_interaction['smplx_param']
        if 'left_hand_pose' in smplx_params:
            del smplx_params['left_hand_pose']
        if 'right_hand_pose' in smplx_params:
            del smplx_params['right_hand_pose']
        smplx_params['transl'][:, 2] -= floor_height
        smplx_params['gender'] = 'male'
        with open(target_body_path, 'wb') as f:
            pickle.dump(smplx_params, f)

        smplx_params = params2torch(smplx_params)
        pelvis = bm(**smplx_params).joints[0, 0, :].detach().cpu().numpy()
        r = torch.cuda.FloatTensor(1).uniform_() * 0.4 + 0.6
        # r = 0.6
        theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
        forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:,
                      2] if action_out == 'sit' else get_forward_dir(Path('./data/test_room') / f'{obj_name}_sit on')
        forward_dir[2] = 0
        forward_dir = forward_dir / torch.norm(forward_dir)
        random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
                                                                 convention="XYZ")
        forward_dir = torch.matmul(random_rot, forward_dir)
        target_point = pelvis + (forward_dir * r).detach().cpu().numpy()
        target_point[2] = 0

        with open(target_point_path, 'wb') as f:
            pickle.dump(target_point, f)

        seq_name = interaction_name + '_down_' + str(seq_id)
        command = "python synthesize/gen_interaction_unify.py --goal_thresh_final -1 --max_depth 15 --num_gen1 128 --num_gen2 32 --num_expand 4 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_{}_marker/{}_1frame " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                  "--target_body_path {} --interaction_name {} --start_point_path {} " \
                  "--use_zero_pose 0 --weight_target_dist 1 " \
                  "--visualize 0".format(action_out, action_out, seq_id, scene_path, scene_name, sdf_path, mesh_path, floor_height,
                                         target_body_path,
                                         seq_name, target_point_path)
        print(command)
        os.system(command)

        last_motion_path = f'results/interaction/{scene_name}/{seq_name}/MPVAEPolicy_{action_out}_marker/{action_out}_1frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'
        """stand up"""
        command = "python synthesize/gen_interaction_unify.py --goal_thresh_final 0.3 --max_depth 10 --num_gen1 128 --num_gen2 16 --num_expand 8 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_{}_marker/{}_1frame " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                  "--target_point_path {} --interaction_name {} --last_motion_path {} " \
                  "--use_zero_pose 0 --weight_target_dist 1 " \
                  "--visualize 0".format(action_out, action_out, seq_id, scene_path, scene_name, sdf_path, mesh_path, floor_height,
                                         target_point_path, interaction_name + '_up_' + str(seq_id), last_motion_path)
        print(command)
        os.system(command)