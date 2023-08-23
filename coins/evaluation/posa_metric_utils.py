import sys
sys.path.append('..')

import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import torchgeometry as tgm
import smplx
import cv2
import os.path as osp
import torch
import numpy as np
import torchgeometry as tgm
from interaction import eulerangles
import pickle
import json

def load_scene_data(device, name, sdf_dir, use_semantics, no_obj_classes, **kwargs):
    R = torch.tensor(eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz'), dtype=torch.float32, device=device)
    t = torch.zeros(1, 3, dtype=torch.float32, device=device)

    with open(osp.join(sdf_dir, name + '.json'), 'r') as f:
        sdf_data = json.load(f)
        grid_dim = sdf_data['dim']
        badding_val = sdf_data['badding_val']
        grid_min = torch.tensor(np.array(sdf_data['min']), dtype=torch.float32, device=device)
        grid_max = torch.tensor(np.array(sdf_data['max']), dtype=torch.float32, device=device)
        voxel_size = (grid_max - grid_min) / grid_dim
        bbox = torch.tensor(np.array(sdf_data['bbox']), dtype=torch.float32, device=device)

    sdf = np.load(osp.join(sdf_dir, name + '_sdf.npy')).astype(np.float32)
    sdf = sdf.reshape(grid_dim, grid_dim, grid_dim, 1)
    sdf = torch.tensor(sdf, dtype=torch.float32, device=device)

    semantics = scene_semantics = None
    if use_semantics:
        semantics = np.load(osp.join(sdf_dir, name + '_semantics.npy')).astype(np.float32).reshape(grid_dim, grid_dim,
                                                                                                   grid_dim, 1)
        # Map `seating=34` to `Sofa=10`. `Seating is present in `N0SittingBooth only`
        semantics[semantics == 34] = 10
        # Map falsly labelled`Shower=34` to `lightings=28`.
        semantics[semantics == 25] = 28
        scene_semantics = torch.tensor(np.unique(semantics), dtype=torch.long, device=device)
        scene_semantics = torch.zeros(1, no_obj_classes, dtype=torch.float32, device=device).scatter_(-1,
                                                                                                      scene_semantics.reshape(
                                                                                                          1, -1), 1)

        semantics = torch.tensor(semantics, dtype=torch.float32, device=device)

    return {'R': R, 't': t, 'grid_dim': grid_dim, 'grid_min': grid_min,
            'grid_max': grid_max, 'voxel_size': voxel_size,
            'bbox': bbox, 'badding_val': badding_val,
            'sdf': sdf, 'semantics': semantics, 'scene_semantics': scene_semantics}

def read_sdf(vertices, sdf_grid, grid_dim, grid_min, grid_max, mode='bilinear'):
    assert vertices.dim() == 3
    assert sdf_grid.dim() == 4
    # sdf_normals: B*dim*dim*dim*3
    batch_size = vertices.shape[0]
    nv = vertices.shape[1]
    sdf_grid = sdf_grid.unsqueeze(0).permute(0, 4, 1, 2, 3)  # B*C*D*D*D
    norm_vertices = (vertices - grid_min) / (grid_max - grid_min) * 2 - 1
    x = F.grid_sample(sdf_grid,
                      norm_vertices[:, :, [2, 1, 0]].view(1, batch_size, nv, 1, 3),
                      padding_mode='border', mode=mode, align_corners=True)
    x = x.permute(2, 0, 3, 4, 1)
    # x = F.grid_sample(sdf_grid,
    #                   norm_vertices[:, :, [2, 1, 0]].view(batch_size, nv, 1, 1, 3),
    #                   padding_mode='border', mode=mode, align_corners=True)
    # x = x.permute(0, 2, 3, 4, 1)
    return x

def eval_physical_metric(vertices, scene_data):
    nv = float(vertices.shape[1])
    x = read_sdf(vertices, scene_data['sdf'],
                            scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                            mode='bilinear').squeeze()

    if x.lt(0).sum().item() < 1:  # if the number of negative sdf entries is less than one
        non_collision_score = torch.tensor(1)
        contact_score = torch.tensor(0.0)
    else:
        non_collision_score = (x > 0).sum().float() / nv
        contact_score = torch.tensor(1.0)

    return float(non_collision_score.detach().cpu().squeeze()), float(contact_score.detach().cpu().squeeze())