"""
Transformation between PROX and POSA scene assets.
"""
import genericpath
import sys
sys.path.append('..')
from configuration.config import *

import pickle
import numpy as np
import trimesh

def scene_registration(scene_name):
    PROX_scene = trimesh.load_mesh(Path.joinpath(scene_folder, scene_name + '.ply'))
    POSA_scene = trimesh.load_mesh(Path.joinpath(proxe_base_folder, 'POSA_dir/scenes', scene_name + '.ply'))
    transform, cost = trimesh.registration.mesh_other(POSA_scene, PROX_scene)
    # (PROX_scene + POSA_scene.apply_transform(transform)).show()
    return transform

def prox_to_posa(scene_name, points):
    transform = np.linalg.inv(POSA_to_PROX_transform[scene_name])
    return np.dot(transform[:3, :3], points.T).T + transform[:3, 3].reshape((1, 3))

if not Path.exists(scene_registration_file):
    POSA_to_PROX_transform = {}
    for scene_name in scene_names:
        POSA_to_PROX_transform[scene_name] = scene_registration(scene_name)
    with open(scene_registration_file, 'wb') as file:
        pickle.dump(POSA_to_PROX_transform, file)

with open(scene_registration_file, 'rb') as file:
    POSA_to_PROX_transform = pickle.load(file)

if __name__ == '__main__':
    for scene_name in scene_names:
        PROX_scene = trimesh.load_mesh(Path.joinpath(scene_folder, scene_name + '.ply'))
        num_vertex = len(PROX_scene.vertices)
        PROX_scene.visual.vertex_colors = np.array([[255, 0, 0, 255]]*num_vertex, dtype=np.uint8)
        POSA_scene = trimesh.load_mesh(Path.joinpath(proxe_base_folder, 'POSA_dir/scenes', scene_name + '.ply'))
        POSA_scene.visual.vertex_colors = np.array([[0, 255, 0, 255]]*num_vertex, dtype=np.uint8)
        (PROX_scene + POSA_scene.apply_transform(POSA_to_PROX_transform[scene_name])).show()
