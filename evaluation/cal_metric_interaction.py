import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import trimesh
import pyrender
import json
import pickle
import time
from tqdm import tqdm
from pathlib import  Path
import torch
import torch.nn.functional as F
import smplx
from smplx import joint_names

unity_to_zup = np.array(
            [[-1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )

shapenet_to_zup = np.array(
            [[1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )

zup_to_shapenet = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]]
)
def params2torch(params, dtype=torch.float32):
    return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}
def params2numpy(params):

    return {k: v.detach().cpu().numpy() if type(v) == torch.Tensor else v for k, v in params.items()}
def rollout_primitives(motion_primitives):
    gender = motion_primitives[0]['gender']
    model_path = "/home/kaizhao/dataset/models_smplx_v1_1/models"
    body_model = smplx.create(model_path=model_path,
                              model_type='smplx',
                              gender=gender,
                              use_pca=False,
                              batch_size=10,
                              ).to(device='cuda')
    joints_list = []
    vertices_list = []
    for idx, motion_primitive in enumerate(motion_primitives):
        # pelvis_original = body_model(betas=torch.cuda.FloatTensor(motion_primitive['betas']).repeat(10, 1)).joints[:, 0, :].detach().cpu().numpy()  # [10, 3]
        smplx_param = motion_primitive['smplx_params'][0]  #[10, 96]
        smplx_param_dict = {
            'betas': torch.cuda.FloatTensor(motion_primitive['betas']).repeat(10, 1),
            'transl': smplx_param[:, :3],
            'global_orient': smplx_param[:, 3:6],
            'body_pose': smplx_param[:, 6:69]
        }
        smplx_param_dict = params2torch(smplx_param_dict)
        output = body_model(**smplx_param_dict)
        vertices, joints = output.vertices, output.joints[:, :55, :]

        rotation = torch.cuda.FloatTensor(motion_primitive['transf_rotmat']).reshape((1, 3, 3)) # [1, 3, 3]
        transl = torch.cuda.FloatTensor(motion_primitive['transf_transl']).reshape((1, 1, 3)) # [1, 1, 3]
        vertices = torch.einsum('bij,bpj->bpi', rotation, vertices) + transl
        joints = torch.einsum('bij,bpj->bpi', rotation, joints) + transl

        if idx == 0:
            start_frame = 0
        elif motion_primitive['mp_type'] == '1-frame':
            start_frame = 1
        elif motion_primitive['mp_type'] == '2-frame':
            start_frame = 2
        else:
            print(motion_primitive['mp_type'])
            start_frame = 0
        vertices_list.append(vertices[start_frame:, :, :])
        joints_list.append(joints[start_frame:, :, :])

    vertices_rollout = torch.cat(vertices_list, dim=0).detach().cpu().numpy()
    joints_rollout = torch.cat(joints_list, dim=0).detach().cpu().numpy()
    return vertices_rollout, joints_rollout, body_model.faces

def calc_sdf(vertices, sdf_dict, return_gradient=False):
    sdf_centroid = sdf_dict['centroid']
    sdf_scale = sdf_dict['scale']
    sdf_grids = sdf_dict['grid']


    batch_size, num_vertices, _ = vertices.shape
    vertices = vertices.reshape(1, -1, 3)  # [B, V, 3]
    vertices = (vertices - sdf_centroid) / sdf_scale  # convert to [-1, 1]
    sdf_values = F.grid_sample(sdf_grids,
                                   vertices[:, :, [2, 1, 0]].view(1, batch_size * num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                   padding_mode='border',
                                   align_corners=True
                                   # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                   ).reshape(batch_size, num_vertices)
    if return_gradient:
        sdf_gradients = sdf_dict['gradient_grid']
        gradient_values = F.grid_sample(sdf_gradients,
                                   vertices[:, :, [2, 1, 0]].view(1, batch_size * num_vertices, 1, 1, 3),
                                   # [2,1,0] permute because of grid_sample assumes different dimension order, see below
                                   padding_mode='border',
                                   align_corners=True
                                   # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                   ).permute(2, 1, 0, 3, 4).reshape(batch_size, num_vertices, 3)
        gradient_values = gradient_values / torch.norm(gradient_values, dim=-1, keepdim=True).clip(min=1e-12)
        return sdf_values, gradient_values
    '''
    # illustration of grid_sample dimension order, assume first dimension to be innermost
    # import torch
    # import torch.nn.functional as F
    # import numpy as np
    # sz = 5
    # input_arr = torch.from_numpy(np.arange(sz * sz).reshape(1, 1, sz, sz)).float()
    # indices = torch.from_numpy(np.array([-1, -1, -0.5, -0.5, 0, 0, 0.5, 0.5, 1, 1, -1, 0.5, 0.5, -1]).reshape(1, 1, 7, 2)).float()
    # out = F.grid_sample(input_arr, indices, align_corners=True)
    # print(input_arr)
    # print(out)
    '''
    return sdf_values

def visualize(vertices, joints, faces, scene_mesh):
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(scene_mesh))
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    body_node = None
    for frame_idx in tqdm(range(len(vertices))):
        body_mesh = trimesh.Trimesh(
            faces=faces,
            vertices=vertices[frame_idx],
            vertex_colors=np.array([200, 200, 200, 255])
        )
        joints_mesh = []
        for joint in joints[frame_idx]:
            trans_mat = np.eye(4)
            trans_mat[:3, 3] = joint
            sm = trimesh.creation.uv_sphere(radius=0.02)
            sm.visual.vertex_colors = [1.0, 0.0, 0.0]
            sm.apply_transform(trans_mat)
            joints_mesh.append(sm)

        viewer.render_lock.acquire()
        if body_node is not None:
            scene.remove_node(body_node)
        body_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(body_mesh + joints_mesh), name='body')
        scene.add_node(body_node)
        viewer.render_lock.release()

def calc_metric_primitives(result_list):
    foot_joints_names = ["left_ankle", "left_foot", "right_ankle", "right_foot", ]
    foot_joints_idx = [joint_names.JOINT_NAMES.index(joint) for joint in foot_joints_names]
    frame_rate = 40.0
    h = 1 / frame_rate
    interaction_vertices = [3464, 6225]

    metrics = {
        'contact': [],
        'second': [],
        'penetration_mean': [],
        'penetration_max': [],
        'interaction_mean': [],
        'interaction_min': [],
    }
    for result_idx, result_path in enumerate(result_list):
        obj_category, obj_id = result_path.parents[5].name.split('_')
        scene_path = scene_base_dir / obj_category / obj_id / 'model.obj'
        sdf_path = scene_base_dir / obj_category / obj_id / 'sdf_gradient.pkl'

        with open(result_path, 'rb') as f:
            motion_data = pickle.load(f)
        vertices, joints, faces = rollout_primitives(motion_data['motion'])  # [T, V, 3], [T, J, 3]
        scene_mesh = trimesh.load(scene_path, force='mesh')
        if 'obj' in scene_path.name:
            scene_mesh.apply_transform(shapenet_to_zup)
        with open(sdf_path, 'rb') as f:
            object_sdf = pickle.load(f)
        """sdf to tensor"""
        sdf_grids = torch.from_numpy(object_sdf['grid'])
        object_sdf['grid'] = sdf_grids.squeeze().unsqueeze(0).unsqueeze(0).to(device='cuda',
                                                                              dtype=torch.float32)  # 1x1xDxDxD
        if 'gradient_grid' in object_sdf:
            gradient_grids = torch.from_numpy(object_sdf['gradient_grid'])
            object_sdf['gradient_grid'] = gradient_grids.permute(3, 0, 1, 2).unsqueeze(0).to(device='cuda',
                                                                                             dtype=torch.float32)  # 1x3xDxDxD
        object_sdf['centroid'] = torch.tensor(object_sdf['centroid']).reshape(1, 1, 3).to(device='cuda',
                                                                                          dtype=torch.float32)
        if result_idx == 0 and debug:
            visualize(vertices, joints, faces, scene_mesh)

        foot_joints = joints[:, foot_joints_idx, :]
        foot_joints_speed = np.linalg.norm(foot_joints[2:, :, :] - foot_joints[:-2, :, :], axis=-1) / (
                    2 * h)  # [T-2, 4]
        foot_joints_speed = np.clip(foot_joints_speed.min(axis=-1) - 0.075, a_min=0, a_max=32767)  # [T-2, ]
        foot_joints_abs_height = np.clip(np.abs(foot_joints[:, :, 2].min(axis=-1)) - 0.05, a_min=0,
                                         a_max=32767)  # [T, 4] -> [T, ]
        contact_score = np.exp(-foot_joints_speed) * np.exp(-foot_joints_abs_height[1:-1])
        contact_score = contact_score.mean()
        vertices_torch = torch.cuda.FloatTensor(vertices)
        """penetration"""
        sdf_values = calc_sdf(vertices_torch, object_sdf)  # [T, V]
        num_inside = sdf_values.lt(0.0).sum(dim=-1)  # [T, ]
        negative_values = sdf_values * (sdf_values < 0)
        # r_penetration = torch.exp(-num_inside / nt / 512)
        penetration_mean = negative_values.abs().sum(axis=-1).mean().item()
        penetration_max = negative_values.abs().sum(axis=-1).amax().item()
        # r_penetration = negative_values.sum(dim=-1) / nt
        """interaction"""

        interaction_marker_sdf = sdf_values[:, interaction_vertices]  # [t, v]
        interaction_mean = interaction_marker_sdf.abs().mean(dim=-1).mean().item()
        interaction_min = interaction_marker_sdf.abs().mean(dim=-1).amin().item()
        metrics['interaction_mean'].append(interaction_mean)
        metrics['interaction_min'].append(interaction_min)
        metrics['penetration_mean'].append(penetration_mean)
        metrics['penetration_max'].append(penetration_max)
        metrics['contact'].append(contact_score)
        metrics['second'].append(vertices.shape[0] / frame_rate)

    for key in metrics:
        metrics[key] = float(np.array(metrics[key]).mean()) if len(metrics[key]) else 0
    return metrics

debug = 0
scene_base_dir = Path('data/shapenet_real')
selected_objs = {
    'Armchairs': ['9faefdf6814aaa975510d59f3ab1ed64',
        'cacb9133bc0ef01f7628281ecb18112',
        'ea918174759a2079e83221ad0d21775',],

    'L-Sofas': ['5cea034b028af000c2843529921f9ad7',],

    'Sofas': ['1dd6e32097b09cd6da5dde4c9576b854',
        '71fd7103997614db490ad276cd2af3a4',
        '277231dcb7261ae4a9fe1734a6086750',],

    'StraightChairs':['2ed17abd0ff67d4f71a782a4379556c7',
        '68dc37f167347d73ea46bea76c64cc3d',
        'd93760fda8d73aaece101336817a135f']
                 }
our_results = []

for obj_category in selected_objs:
    for obj_id in selected_objs[obj_category]:
        our_results += list(Path('results/interaction', obj_category + '_' + obj_id).glob('sit_up*/MPVAEPolicy_babel_marker/bidir_contactfeet2/policy_search/*/*.pkl'))
our_results = sorted(our_results)

metrics = {
    'ours': calc_metric_primitives(our_results),
           }
print(metrics)
with open('evaluation/metric_sit.json', 'w') as f:
    json.dump(metrics, f)
