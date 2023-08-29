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

np.random.seed(233)
torch.manual_seed(233)


def params2torch(params, dtype=torch.float32):
    return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}


def params2numpy(params):
    return {k: v.detach().cpu().numpy() if type(v) == torch.Tensor else v for k, v in params.items()}


def get_navmesh(navmesh_path, scene_path, agent_radius, floor_height=0.0, visualize=False):
    if navmesh_path.exists():
        navmesh = trimesh.load(navmesh_path, force='mesh')
    else:
        scene_mesh = trimesh.load(scene_path, force='mesh')
        """assume the scene coords are z-up"""
        scene_mesh.vertices[:, 2] -= floor_height
        scene_mesh.apply_transform(zup_to_shapenet)
        navmesh = create_navmesh(scene_mesh, export_path=navmesh_path, agent_radius=agent_radius, visualize=visualize)
    navmesh.vertices[:, 2] = 0
    return navmesh


if __name__ == "__main__":
    scene_name = 'test_room'
    scene_dir = Path('./data/test_room')
    scene_path = scene_dir / 'room.ply'
    floor_height = 0
    navmesh_tight_path = scene_dir / 'navmesh_tight.ply'
    navmesh_loose_path = scene_dir / 'navmesh_loose.ply'
    # get loose navmesh for path planning
    navmesh_tight = get_navmesh(navmesh_tight_path, scene_path, agent_radius=0.05, floor_height=floor_height, visualize=True)
    # get tight navmesh for path planning
    navmesh_loose = get_navmesh(navmesh_loose_path, scene_path, agent_radius=0.4, floor_height=floor_height, visualize=True)

    visualize = False
    path_idx = 0
    path_name = f'path_{path_idx}'
    wpath_path = scene_dir / f'{path_name}.pkl'

    """automatic path finding"""
    # specify start and target location
    start_point = np.array([-1.7, 2.35, 0])
    target_point = np.array([-1.4, 0.54, 0])
    start_target = np.stack([start_point, target_point])
    # find collision free path
    scene_mesh = trimesh.load(scene_path, force='mesh')
    wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=visualize, scene_mesh=scene_mesh)

    """optional: manually specify waypoints"""
    # wpath = np.array([
    #     [2.1, -0.43, 0.],
    #     [0., 2.2, 0.],
    #     [-2.8, 2.2, 0.],
    # ])

    with open(wpath_path, 'wb') as f:
        pickle.dump(wpath, f)
    max_depth = 30 * len(wpath)

    num_sequence = 4
    command = "python synthesize/gen_locomotion_unify.py --goal_thresh 0.5 --goal_thresh_final 0.2 --max_depth {} --num_gen1 128 --num_gen2 16 --num_expand 8 " \
              "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_samp_collision/locomotion " \
              "--gen_name policy_search --num_sequence {} " \
              "--scene_path {} --scene_name {} --navmesh_path {} --floor_height {:.2f} --wpath_path {} --path_name {} " \
              "--weight_pene 1 " \
              "--visualize 0 --use_zero_pose 1 --use_zero_shape 1 --random_orient 0 --clip_far 1".format(
        max_depth,
        num_sequence,
        scene_path,
        scene_name,
        navmesh_tight_path,
        floor_height,
        wpath_path,
        path_name)
    print(command)
    os.system(command)
