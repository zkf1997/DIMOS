import os
import pickle
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as R
from test_navmesh import *
from synthesize.demo_locomotion import get_navmesh
from exp_GAMMAPrimitive.utils.environments import *
from exp_GAMMAPrimitive.utils import config_env

np.random.seed(233)
torch.manual_seed(233)

def params2torch(params, dtype=torch.float32):
    return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}

def params2numpy(params):
    return {k: v.detach().cpu().numpy() if type(v) == torch.Tensor else v for k, v in params.items()}

def project_to_navmesh(navmesh, points):
    closest, _, _ = trimesh.proximity.closest_point(navmesh, points)
    return closest


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
if __name__ == '__main__':
    scene_name = 'test_room'
    scene_dir = Path('./data/test_room')
    scene_path = scene_dir / 'room.ply'
    floor_height = 0
    navmesh_tight_path = scene_dir / 'navmesh_tight.ply'
    navmesh_loose_path = scene_dir / 'navmesh_loose.ply'
    # get loose navmesh for path planning
    navmesh_tight = get_navmesh(navmesh_tight_path, scene_path, agent_radius=0.05, floor_height=floor_height,
                                visualize=True)
    # get tight navmesh for path planning
    navmesh_loose = get_navmesh(navmesh_loose_path, scene_path, agent_radius=0.2, floor_height=floor_height,
                                visualize=True)

    seq_num = 4
    visualize = False
    for seq_id in range(seq_num):
        # scene_dir = Path('./data/test_room')
        action = 'sit'
        obj_category = 'sofa'
        obj_id = 0
        target_interaction_path = f'data/test_room/{obj_category}_{action} on/goal.pkl'
        path_name = f'to_{obj_category}_{obj_id}_{seq_id}'
        interaction_name = '_'.join(['loco_inter', action, obj_category, str(obj_id), str(seq_id)])
        wpath_path = scene_dir / 'waypoints' / f'{path_name}.pkl'
        wpath_path.parent.mkdir(exist_ok=True, parents=True)
        sdf_path = scene_dir / f'{obj_category}_sdf_gradient.pkl'
        sdf_path.parent.mkdir(exist_ok=True, parents=True)
        mesh_path = scene_dir / f'{obj_category}.ply'
        mesh_path.parent.mkdir(exist_ok=True, parents=True)
        target_point_path = Path('results', 'tmp', scene_name, interaction_name, 'target_point.pkl')
        target_point_path.parent.mkdir(exist_ok=True, parents=True)
        target_body_path = Path('results', 'tmp', scene_name, interaction_name, 'target_body.pkl')

        with open(target_interaction_path, 'rb') as f:
            target_interaction = pickle.load(f)
        smplx_params = target_interaction['smplx_param']
        del smplx_params['left_hand_pose']
        del smplx_params['right_hand_pose']
        smplx_params['transl'][:, 2] -= floor_height
        smplx_params['gender'] = 'male'
        with open(target_body_path, 'wb') as f:
            pickle.dump(smplx_params, f)

        smplx_params = params2torch(smplx_params)
        pelvis = bm(**smplx_params).joints[0, 0, :].detach().cpu().numpy()

        start_point = np.array([-1.9, 2.45, 0])
        r = torch.cuda.FloatTensor(1).uniform_() * 0.2 + 1.0
        # r = 1.0
        body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
        forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
        forward_dir[2] = 0
        forward_dir = forward_dir / torch.norm(forward_dir)
        # theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        # random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]), convention="XYZ")
        # forward_dir = torch.matmul(random_rot, forward_dir)
        target_point = pelvis + (forward_dir * r).detach().cpu().numpy()
        target_point[2] = 0
        start_target = np.stack([start_point, target_point])

        scene_mesh = trimesh.load(scene_path, force='mesh')
        wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=visualize, scene_mesh=scene_mesh)
        if len(wpath) == 0:
            start_target = project_to_navmesh(navmesh_loose, start_target)
            wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=visualize,
                              scene_mesh=scene_mesh)
        print('find a path of length:', len(wpath))
        with open(wpath_path, 'wb') as f:
            pickle.dump(wpath, f)

        cfg_policy = 'MPVAEPolicy_frame_label_walk_collision/map_nostop'
        command = (
            f"python synthesize/gen_locomotion_unify.py --goal_thresh 0.5 --goal_thresh_final 0.2 --max_depth 180 --num_gen1 128 --num_gen2 16 --num_expand 8 "
            f"--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/{cfg_policy} "
            f"--gen_name policy_search --num_sequence 1 "
            f"--random_seed {seq_id} --scene_path {scene_path} --scene_name {scene_name} --navmesh_path {navmesh_tight_path} "
            f"--floor_height {floor_height:.2f} --wpath_path {wpath_path} --path_name {path_name} "
            f"--clip_far 1 --history_mode 1 --weight_pene 1 "
            f"--visualize 0 --use_zero_pose 1 --use_zero_shape 1"
        )
        print(command)
        os.system(command)

        last_motion_path = f'results/locomotion/{scene_name}/{path_name}/{cfg_policy}/policy_search/seq000/results_ssm2_67_condi_marker_map_0.pkl'
        """sit down"""
        command = (
            f"python synthesize/gen_interaction_unify.py --goal_thresh_final -1 --max_depth 15 --num_gen1 128 --num_gen2 32 --num_expand 4 "
            f"--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_sit_marker/sit_2frame "
            f"--gen_name policy_search --num_sequence 1 "
            f"--random_seed {seq_id} --scene_path {scene_path} --scene_name {scene_name} --sdf_path {sdf_path} --mesh_path {mesh_path} "
            f"--floor_height {floor_height:.2f} "
            f"--target_body_path {target_body_path} --interaction_name {interaction_name + '_down'} --last_motion_path {last_motion_path} "
            f"--history_mode 2 --weight_target_dist 1 "
            f"--visualize 0"
        )
        print(command)
        os.system(command)



