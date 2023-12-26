import os
import pickle
import trimesh
import torch
import torch.nn.functional as F
import numpy as np
import smplx
import json
from pathlib import Path
import pyrender
import glob

zup_to_shapenet = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]]
)
shapenet_to_zup = np.array(
    [[1, 0, 0, 0],
     [0, 0, -1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]]
)

def transform_real_size_on_floor(obj_path, json_path):
    with open(json_path, 'r') as f:
        meta_info = json.load(f)
    obj_mesh = trimesh.load(obj_path, force='mesh')
    scale = (np.array(meta_info['max']) - np.array(meta_info['min'])) / (obj_mesh.bounds[1] - obj_mesh.bounds[0])
    transform = np.diag([scale[0], scale[1], scale[2], 1])
    transform[1, 3] = -obj_mesh.bounds[0, 1] * scale[1]
    return transform

# https://github.com/mikedh/trimesh/issues/507#issuecomment-514973337
def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, visual=g.visual)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def export_obj(mesh, export_dir):
    # export the mesh including data
    # with open(export_dir / 'model.obj', 'w', encoding='utf-8') as f:
    #     mesh.export(f, file_type='obj', include_texture=True)

    # export the mesh including data
    obj, data = trimesh.exchange.export.export_obj(mesh, include_texture=True, return_texture=True)

    obj_path = export_dir / 'model.obj'
    with open(obj_path, 'w') as f:
        f.write(obj)
    # save the MTL and images
    for k, v in data.items():
        with open(export_dir / k, 'wb') as f:
            f.write(v)

    # obj, data = trimesh.exchange.export.export_obj(mesh, include_texture=True)
    #
    # with open(export_dir / 'model.obj', 'w') as f:
    #     f.write(obj)
    # # save the MTL and images
    # for k, v in data.items():
    #     with open(os.path.join(export_dir, k), 'wb') as f:
    #         f.write(v)

output_dir = Path('data/shapenet_real')

# more beds from shapenet
input_dir = Path('data/ShapeNetCore.v2/')
with open('utils/obj_ids.json', 'r') as f:
    obj_dict = json.load(f)

for obj_category in obj_dict.keys():
    obj_list = obj_dict[obj_category]
    for obj_id in obj_list:
        category_name, category_id = obj_category.split('_')
        obj_dir = input_dir / category_id / obj_id
        print(obj_dir)
        obj_path = obj_dir / 'models' / 'model_normalized.obj'
        json_path = obj_dir / 'models' / 'model_normalized.json'
        with open(json_path, 'r') as f:
            meta_info = json.load(f)
        obj_mesh = trimesh.load(obj_path, force='mesh')
        obj_mesh.apply_transform(transform_real_size_on_floor(obj_path, json_path))
        # obj_mesh.show()
        export_dir = output_dir / category_name / obj_id
        export_dir.mkdir(exist_ok=True, parents=True)
        # obj_mesh.export(export_dir / 'model.obj', file_type='obj', include_texture=True)
        print("to_export", obj_dir)
        export_obj(obj_mesh, export_dir)
