import os
import pickle
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as R
from test_navmesh import *
from exp_GAMMAPrimitive.utils.environments import *
from exp_GAMMAPrimitive.utils import config_env
from pathlib import Path
from synthesize.demo_locomotion import get_navmesh
from synthesize.demo_loco_inter import project_to_navmesh

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
if __name__ == "__main__":
    scene_name = 'MPH8'
    floor_height = 0
    base_dir = Path('/home/kaizhao/projects/gamma/data/scenes/PROX')
    scene_path = base_dir / 'MPH8_floor.ply'
    navmesh_tight_path = base_dir / 'MPH8_navmesh_tight.ply'
    navmesh_loose_path = base_dir / 'MPH8_navmesh_loose.ply'
    # get loose navmesh for path planning
    navmesh_tight = get_navmesh(navmesh_tight_path, scene_path, agent_radius=0.05, floor_height=floor_height,
                                visualize=True)
    # get tight navmesh for path planning
    navmesh_loose = get_navmesh(navmesh_loose_path, scene_path, agent_radius=0.4, floor_height=floor_height,
                                visualize=True)


    action = 'sit'
    obj_category = 'bed'
    obj_id = 9
    sdf_path = base_dir / 'bed_9_sdf_grad.pkl'
    mesh_path = base_dir / 'MPH8-bed-9-floor.ply'
    target_interaction_path = '/home/kaizhao/projects/gamma/results/coins/two_stage/prox/optimization_after_get_body/sit on/sit on_bed_9/1.pkl'

    for seq_id in range(1):
        path_name = 'to_bed_sit_{}'.format(seq_id)
        wpath_path = base_dir / scene_name / 'waypoints' / (path_name + '.pkl')
        wpath_path.parent.mkdir(exist_ok=True, parents=True)
        interaction_name = '_'.join([action, obj_category, str(obj_id), str(seq_id)])
        target_point_path = Path('results', 'tmp', scene_name, interaction_name, 'target_point.pkl')
        target_point_path.parent.mkdir(exist_ok=True, parents=True)
        target_body_path = Path('results', 'tmp', scene_name, interaction_name, 'target_body.pkl')

        with open(target_interaction_path, 'rb') as f:
            target_interaction = pickle.load(f)
        smplx_params = target_interaction['smplx_param']
        del smplx_params['left_hand_pose']
        del smplx_params['right_hand_pose']
        smplx_params['transl'][:, 2] -= 0.3
        smplx_params['transl'][:, 1] -= 0.1
        smplx_params['gender'] = 'male'
        with open(target_body_path, 'wb') as f:
            pickle.dump(smplx_params, f)

        smplx_params = params2torch(smplx_params)
        pelvis = bm(**smplx_params).joints[0, 0, :].detach().cpu().numpy()

        start_point = np.array([1.65, -0.42, 0])
        target_point = np.array([-1.94, -0.13, 0])
        start_target = np.stack([start_point, target_point])

        scene_mesh = trimesh.load(scene_path, force='mesh')
        scene_mesh.vertices[:, 2] -= floor_height + 0.02
        wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=False, scene_mesh=scene_mesh)
        with open(wpath_path, 'wb') as f:
            pickle.dump(wpath, f)

        command = "python synthesize/gen_locomotion_unify.py --goal_thresh 0.5 --goal_thresh_final 0.3 --max_depth 120 --num_gen1 128 --num_gen2 32 --num_expand 8 " \
                  "--project_dir /mnt/atlas_root/vlg-nfs/kaizhao/gamma --cfg_policy ../../../../../../mnt/atlas_root/vlg-nfs/kaizhao/gamma/results/exp_GAMMAPrimitive/MPVAEPolicy_samp_collision/map_kl10_batchfix_pene1 " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --navmesh_path {} --floor_height {:.2f} --wpath_path {} --path_name {} " \
                  "--clip_far 1 --random_orient 1 --weight_pene 1 " \
                  "--visualize 0 --use_zero_pose 1 --use_zero_shape 1".format(seq_id, scene_path, scene_name, navmesh_path, floor_height, wpath_path, path_name)
        print(command)
        os.system(command)

        command = "python synthesize/gen_locomotion_unify.py --goal_thresh 0.5 --goal_thresh_final 0.3 --max_depth 120 --num_gen1 128 --num_gen2 32 --num_expand 8 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_babel_walk_collision/map_babel_walk " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --navmesh_path {} --floor_height {:.2f} --wpath_path {} --path_name {} " \
                  "--clip_far 1 --random_orient 0 --weight_pene 1 " \
                  "--visualize 0 --use_zero_pose 1 --use_zero_shape 1".format(seq_id, scene_path, scene_name,
                                                                              navmesh_path, floor_height, wpath_path,
                                                                              path_name)
        print(command)
        os.system(command)

        last_motion_path = 'results/locomotion/MPH8/{}/MPVAEPolicy_babel_walk_collision/map_babel_walk/policy_search/seq000/results_ssm2_67_condi_marker_map_0.pkl'.format(path_name)
        """sit down"""
        command = "python synthesize/gen_interaction_unify.py --goal_thresh_final -1 --max_depth 15 --num_gen1 128 --num_gen2 32 --num_expand 4 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_babel_marker/sit_2frame_test " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                  "--target_body_path {} --interaction_name {} --last_motion_path {} " \
                  "--history_mode 2 --weight_target_dist 5 " \
                  "--visualize 1".format(seq_id, scene_path, scene_name, sdf_path, mesh_path, floor_height, target_body_path, interaction_name + '_down', last_motion_path)
        print(command)
        os.system(command)

