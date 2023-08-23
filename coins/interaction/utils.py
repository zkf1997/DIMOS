import sys
sys.path.append('..')

import torch
import numpy as np
import trimesh

from configuration.config import *
from configuration.joints import *

def padded_idx_to_code(verb_ids):
    codebook = torch.cat([torch.eye(4, dtype=torch.float32, device=verb_ids.device),
               torch.zeros((1, 4), dtype=torch.float32, device=verb_ids.device)], dim=0)
    return codebook[verb_ids.long()]

def transform_back(vertices, centroid, rotation):
    B, N, C = vertices.shape
    vertices = vertices.matmul(torch.inverse(rotation).transpose(1, 2))
    vertices = vertices + centroid.unsqueeze(1)
    return vertices

def skeleton_to_mesh(skeleton, color):
    joint_num = skeleton.shape[0]
    body = trimesh.primitives.Sphere(radius=0.05, center=skeleton[0])
    body.visual.vertex_colors = np.array(color[0] * 255, dtype=np.uint8)
    for idx in range(1, joint_num):
        joint = skeleton[idx]
        joint_mesh = trimesh.primitives.Sphere(radius=0.05, center=joint)
        joint_mesh.visual.vertex_colors = np.array(color[idx] * 255, dtype=np.uint8)
        body = body + joint_mesh
        parent_joint = skeleton[parent_joint_idx[idx]]
        bone = np.array([joint, parent_joint])
        body = body + trimesh.creation.cylinder(0.02, segment=bone, vertex_colors=color[0])
    return body
