import sys

sys.path.append('..')
# sys.path.append('../human_body_prior/src')

import json
import os
import os.path

import open3d as o3d
import numpy as np
import trimesh
import smplx
import pickle
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from copy import deepcopy
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from tqdm import tqdm

from data.load_interaction import get_interaction_segments
from data.scene import scenes, to_trimesh
from data.body_model import body_model_dict
from data.utils import *
from configuration.config import *
from interaction.pointnet2 import farthest_point_sample
from interaction.chamfer_distance import chamfer_dists
from interaction.mesh import Mesh

def to_smplx_input(record):
    """
    Convert interaction record to SMPLX body model input.
    """
    for param in smplx_param_names:
        if param in record:
            record[param] = torch.tensor(record[param], dtype=torch.float32,
                                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                         )
    record['left_hand_pose'] = record['left_hand_pose'][:, :num_pca_comps]
    record['right_hand_pose'] = record['right_hand_pose'][:, :num_pca_comps]
    return record

def get_body_by_batch(body_model, smplx_input, batch_size=256):
    """
        Get SMPLX bodies in batch.
    """
    body_model = smplx.create(smplx_model_folder, model_type='smplx',
                 gender='neutral', ext='npz',
                 num_pca_comps=num_pca_comps, batch_size=batch_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    frame_num = smplx_input['transl'].shape[0]
    last_frame = 0
    vertices = []
    pelvis = []
    joints = []
    full_poses = []
    while last_frame < frame_num:
        cur_frame = min(last_frame + batch_size, frame_num)
        if (cur_frame - last_frame) != batch_size:
            body_model = smplx.create(smplx_model_folder, model_type='smplx',
                                      gender='neutral', ext='npz',
                                      num_pca_comps=num_pca_comps, batch_size=cur_frame - last_frame).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        smplx_params = {}
        for key in smplx_input:
            smplx_params[key] = smplx_input[key][last_frame:cur_frame, :]
        #     print(key, smplx_params[key].shape)
        # print(cur_frame)
        smplx_output = body_model(**smplx_params, return_full_pose=True)
        vertices += [smplx_output.vertices.detach().cpu().numpy()]
        pelvis += [smplx_output.joints[:, 0, :].detach().cpu().numpy()]
        joints += [smplx_output.joints.detach().cpu().numpy()]
        full_poses += [smplx_output.full_pose.detach().cpu().numpy()]
        last_frame = cur_frame
    vertices = np.concatenate(vertices, axis=0)
    pelvis = np.concatenate(pelvis, axis=0)
    joints = np.concatenate(joints, axis=0)
    full_poses = np.concatenate(full_poses, axis=0)
    return vertices, pelvis, joints, full_poses

mesh = None
def batch_downsample(vertices):
    """
    Downsample body mesh (from 10475 vertices to 655 vertices) in batch.
    """
    global mesh
    if mesh is None:
        mesh = Mesh(num_downsampling=2)

    batch_size = 256
    vertices_downsampled = []
    frame_num = vertices.shape[0]
    last_frame = 0
    while last_frame < frame_num:
        cur_frame = min(last_frame + batch_size, frame_num)
        vertices_batch = vertices[last_frame:cur_frame, :, :]
        vertices_downsampled.append(mesh.downsample(torch.from_numpy(vertices_batch).to(torch.device('cuda'))).cpu().numpy()
                                    )
        last_frame = cur_frame
    return np.concatenate(vertices_downsampled, axis=0)

def get_vertex_obj_dists(vertices, frame_scene_names):
    """
    calculate distance from each vertex to each category of objects, if the object category not exist in one scene, assign a big distance
    """
    not_exist_distance = 10.0
    max_points = 4096 * 2  # maximum number of points for one object
    vertices = batch_downsample(vertices)
    num_body, num_vertex = vertices.shape[:2]
    vertex_obj_dists = np.ones((num_body, num_vertex, obj_category_num), dtype=np.float32) * not_exist_distance
    for body_idx in tqdm(range(num_body)):
        scene_name = frame_scene_names[body_idx]
        scene = scenes[scene_name]
        if not hasattr(scene, 'category_pointclouds'):
            category_instances = defaultdict(list)
            for node in scene.object_nodes:
                category_instances[node.category].append(node)
            category_pointclouds = {}
            for category in range(obj_category_num):
                if len(category_instances[category]):
                    category_pointclouds[category] = np.concatenate(
                        [np.asarray(instance.pointcloud.points, dtype=np.float32) for instance in
                         category_instances[category]], axis=0)
            scene.category_pointclouds = category_pointclouds

        for category_idx in range(obj_category_num):
            category_pointclouds = scene.category_pointclouds
            if category_idx in category_pointclouds:
                squared_dists = chamfer_dists(torch.from_numpy(vertices[[body_idx], :, :]).float().cuda(),
                                                 torch.from_numpy(category_pointclouds[category_idx]).float().cuda().unsqueeze(0))  # 1xBody
                vertex_obj_dists[body_idx, :, category_idx] = np.sqrt(squared_dists.squeeze().cpu().numpy())

    return vertex_obj_dists

def to_padded_array(input, length=maximum_atomics):
    """
    Convert input list to numpy array and pad with -1.
    """
    output = np.ones(length) * -1
    output[:len(input)] = np.array(input)
    return output

class InteractionDataset(Dataset):
    """
    Class for dataset of interaction frames.
    """
    def __init__(self, interaction_data, obj_code='onehot', verb_code='onehot', use_augment='',
                 num_points=4096, center_type='human', scale_obj=False, normalize=False, used_interaction='all', used_instance=None,
                 point_sample='random', rotation=None, use_composite=True, skip_prox_composite=None,
                 include_motion=False,
                 single_frame=None, keep_frame=None, data_overwrite=False, split='train'):
        """
        Args:
            num_points: int, number of points we use to represent each object
            center_type: ['human', 'object'], use the human pelvis or object center to canonicalize coordinates
            point_sample: ['random, 'uniform'], method to sample points from object mesh vertices, 'random' uses random sampling and 'uniform' uses farthest point sampling
            used_interaction: str, select one category of interaction data like 'stand on-floor' or all data using 'all'
            use_composite: bool, whether to include composite interaction data
            include_motion: bool, whether to include the dynamic interaction such as 'step up-chair', 'lie down-sofa'
            data_overwrite: bool, whether to overwrite data file
            split: ['train', 'test'], dataset split
        """
        # get scene object pointclouds
        pointcloud_dict = {}
        used_scenes = train_scenes if split =='train' else test_scenes
        for scene_name in used_scenes:
            pointcloud_dict[scene_name] = {}
            scene = scenes[scene_name]
            for node_idx in range(len(scene.object_nodes)):
                obj = scene.get_mesh_with_accessory(node_idx)
                obj_vertices = np.asarray(obj.vertices, dtype=np.float32)
                obj_vertex_colors = np.asarray(obj.visual.vertex_colors[:, :3]).astype(np.float32)  # [0, 255]
                obj_vertex_normals = np.asarray(obj.vertex_normals).astype(np.float32)
                if point_sample == 'uniform':  # use uniform sampled pointcloud for validation
                    if obj_vertices.shape[0] > num_points:
                        idx = np.squeeze(farthest_point_sample(torch.from_numpy(obj_vertices[None, :, :]).to(torch.device('cuda')), npoint=num_points).cpu().numpy())
                        obj_vertices = obj_vertices[idx, :]
                        obj_vertex_colors = obj_vertex_colors[idx, :]
                        obj_vertex_normals = obj_vertex_normals[idx, :]
                    elif obj_vertices.shape[0] < num_points:
                        idx = np.random.choice(np.arange(obj_vertices.shape[0]), num_points, replace=True)
                        obj_vertices = obj_vertices[idx, :]
                        obj_vertex_colors = obj_vertex_colors[idx, :]
                        obj_vertex_normals = obj_vertex_normals[idx, :]
                pointcloud_dict[scene_name][node_idx] = (obj_vertices, obj_vertex_colors, obj_vertex_normals)

        # load precomputed data
        export_dir = Path.joinpath(project_folder, "data", "interaction_dataset",
                                  split + '_' + used_interaction)
        if not export_dir.exists() or data_overwrite:
            export_dir.mkdir(exist_ok=True, parents=True)
            body_model = body_model_dict['neutral']
            # select whether to use dynamic interactions
            all_interactions = atomic_interaction_names_include_motion if include_motion else atomic_interaction_names
            if use_composite:
                all_interactions += composed_interaction_names
            # skip come composite interactions for test composition
            if skip_prox_composite == 'all':
                skip_composite_interactions = composed_interaction_names
            elif skip_prox_composite == 'test':
                skip_composite_interactions = test_composed_interaction_names
            else:
                skip_composite_interactions = []
            for interaction in all_interactions:
                if used_interaction != 'all' and interaction != used_interaction:
                    continue
                if interaction in skip_composite_interactions:
                    print('skip ', interaction, 'in dataset')
                    continue
                # print(interaction_data[0])
                # records = get_interaction_segments(interaction.split('+'), interaction_data, mode='verb-noun')
                records = deepcopy(get_interaction_segments(interaction.split('+'), interaction_data, mode='verb-noun'))
                if len(records) == 0:
                    print('no data of ', interaction)
                    continue
                print('loading ', interaction, ':', len(records))
                smplx_params = [to_smplx_input(record['smplx_param']) for record in records]
                frame_scene_names = [record['scene_name'] for record in records]
                smplx_input = {}
                for key in smplx_param_names:
                    smplx_input[key] = torch.cat([smplx_param[key] for smplx_param in smplx_params], dim=0)
                # print('get vertices')
                body_vertices, pelvis, joints, full_poses = get_body_by_batch(body_model, smplx_input)
                downsampled_vertices = batch_downsample(body_vertices)
                # print('vertices got')
                for idx, record in enumerate(records):
                    record['interaction'] = interaction
                    record['body_vertices'] = body_vertices[idx]
                    record['pelvis'] = pelvis[idx]
                    record['joints'] = joints[idx, :55, :]
                    record['full_poses'] = full_poses[idx, :]
                    record['smplx_param'] = {}
                    # need to copy param, do not do anything related to cuda in getitem which can cause multiprocess related error need manual handle
                    for key in smplx_param_names:
                        record['smplx_param'][key] = smplx_params[idx][key].cpu()
                    scene_name = record['scene_name']
                    scene = scenes[scene_name]
                    # contact dists
                    # vertex_obj_dists = np.ones(downsampled_vertices.shape[1], dtype=np.float32) * 233
                    specified_obj_idx = [record['interaction_obj_idx'][record['interaction_labels'].index(atomic)] for
                                         atomic in interaction.split('+')]
                    obj_vertices = np.concatenate(
                        [scene.get_mesh_with_accessory(obj_idx).vertices for obj_idx in specified_obj_idx], axis=0)
                    vertex_obj_dists = chamfer_dists(torch.from_numpy(downsampled_vertices[[idx], :, :]).float().cuda(),
                                                     torch.from_numpy(obj_vertices).float().cuda().unsqueeze(0)) # 1xBody
                    joint_obj_dists = chamfer_dists(torch.from_numpy(joints[[idx], :, :]).float().cuda(),
                                                     torch.from_numpy(obj_vertices).float().cuda().unsqueeze(
                                                         0))  # 1xBody
                    record['contact_dist'] = np.sqrt(vertex_obj_dists.squeeze().cpu().numpy())
                    record['sdf'] = scene.calc_sdf(torch.from_numpy(downsampled_vertices[[idx], :, :])).squeeze().numpy()
                    record['joint_contact_dist'] = np.sqrt(joint_obj_dists.squeeze().cpu().numpy())
                    record['joint_sdf'] = scene.calc_sdf(torch.from_numpy(joints[[idx], :, :])).squeeze().numpy()

                # some categories have too few records, try to make category more balanced
                if len(records) < 512:
                    records = records * (512 // len(records))
                file_path = export_dir / (interaction + '.pkl')
                file_path.parent.mkdir(exist_ok=True)
                with open(file_path, "wb") as pkl_file:
                    pickle.dump(records, pkl_file)
                # break

        data = []
        for file_path in export_dir.iterdir():
            print('load: ', file_path)
            with open(file_path, "rb") as pkl_file:
                cache = pickle.load(pkl_file)
            data += cache

        self.data = data
        self.pointcloud_dict = pointcloud_dict
        self.obj_code = obj_code
        self.verb_code = verb_code
        self.use_augment = use_augment
        self.num_points = num_points
        self.center_type = center_type
        self.normalize = normalize
        self.single_frame = single_frame
        self.scale_obj = scale_obj
        self.point_sample = point_sample
        self.rotation = rotation
        self.split = split

    def __getitem__(self, idx):
        """
        Return:
            smplx_param: smplx body params
            full_pose: smplx body pose, 155D
            pelvis: pelvis location
            joints: body joints locations
            body_vertices: body vertices locations
            contact_dist: distance from downsampled body vertices to interaction objects
            sdf: scene SDF value of downsampled body vertices
            joint_contact_dist: distance from body joints to interaction objects
            joint_sdf: scene SDF value of body joints
            object_pointclouds: point clouds of interaction objects
            interaction: str, interaction description
            frame_idx: frame index in orginal video sequence
            num_atomics: number of atomic interactions
            interaction_obj_ids: padded instance id of interaction objects
            verb_ids: padded verb id of verbs in this interaction
            noun_ids: padded noun id of nouns in this interaction
            scene_name: name of the scene where this interaction frame is captured
            centroid: location of the local coord system origin in the scene coord system
            rotation: rotation from the scene coord system to local coord system
        """
        # record = self.data[-1]
        record = self.data[idx] if self.single_frame is None else self.data[self.single_frame]
        scene = scenes[record['scene_name']]
        body_vertices = record['body_vertices']
        pelvis = record['pelvis']
        joints = record['joints']
        smplx_param = deepcopy(record['smplx_param'])

        # interaction as verb-noun of specified atomics
        specified_atomics = record['interaction'].split('+')
        num_atomics = len(specified_atomics)
        verb_ids = []
        noun_ids = []
        for atomic in specified_atomics:
            verb, noun = atomic.split('-')
            verb_id = action_names.index(verb)
            noun_id = category_dict[category_dict['mpcat40'] == noun].index[0]
            verb_ids.append(verb_id)
            noun_ids.append(noun_id)

        # pointclouds of specified objects
        specified_obj_idx = [record['interaction_obj_idx'][record['interaction_labels'].index(atomic)] for atomic in specified_atomics]
        interaction_objs = []
        for node_idx in specified_obj_idx:
            obj_vertices, obj_vertex_colors, obj_vertex_normals = self.pointcloud_dict[record['scene_name']][node_idx]
            # resample
            num_vertices = obj_vertices.shape[0]
            if self.point_sample == 'random':
                if num_vertices >= self.num_points:
                    idx = np.random.choice(np.arange(num_vertices), self.num_points, replace=False)
                elif num_vertices < self.num_points:
                    idx = np.random.choice(np.arange(num_vertices), self.num_points, replace=True)
                obj_vertices = obj_vertices[idx, :]
                obj_vertex_colors = obj_vertex_colors[idx, :]
                obj_vertex_normals = obj_vertex_normals[idx, :]
                num_vertices = self.num_points

            # normalize positions and add point features
            obj_points = np.zeros((num_vertices, 9), dtype=np.float32)  # num_point * 9, 3D location + 3D color + 3D normal
            obj_points[:, :3] = obj_vertices
            obj_points[:, 3:6] = obj_vertex_colors / 255.0  # color
            obj_points[:, 6:9] = obj_vertex_normals
            interaction_objs.append(obj_points)

        # to human centric coordinates
        if self.center_type == 'human':
            centroid = pelvis
            if self.use_augment == 'xy':
                centroid = centroid + np.random.rand(3) * np.array([0.5, 0.5, 0]) - np.array([0.25, 0.25, 0])
            elif self.use_augment == 'xyz':
                centroid = centroid + np.random.rand(3) * np.array([0.5, 0.5, 0.5]) - np.array([0.25, 0.25, 0.5])
            centroid = np.float32(centroid)
        elif self.center_type == 'object':
            points = []
            for obj_points in interaction_objs:
                points.append(obj_points[:, :3])
            points = np.concatenate(points, axis=0)
            centroid = 0.5 * (np.max(points, axis=0) + np.min(points, axis=0))
            # centroid = np.float32(centroid + 2.0 * np.random.rand(3) - 1.0)
            centroid = np.float32(centroid)
        for obj_points in interaction_objs:
            obj_points[:, :3] = obj_points[:, :3] - centroid
        body_vertices = (body_vertices - centroid)
        pelvis = pelvis - centroid
        joints = joints - centroid
        smplx_param['transl'] = smplx_param['transl'] - centroid

        # reorient
        if self.center_type == 'human':
            global_orient = Rotation.from_rotvec(smplx_param['global_orient'].detach().cpu().numpy().squeeze())
            rotation = np.linalg.inv(global_orient.as_matrix()).astype(np.float32)
        elif self.center_type == 'object':
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation = np.array([[cosval, -sinval, 0],
                                        [sinval, cosval, 0],
                                        [0, 0, 1]], dtype=np.float32)
        # rotation denotes left multiply rotation matrix
        for obj_points in interaction_objs:
            obj_points[:, :3] = np.dot(obj_points[:, :3], rotation.T)
            obj_points[:, 6:] = np.dot(obj_points[:, 6:], rotation.T)
        # https://www.dropbox.com/scl/fi/zkatuv5shs8d4tlwr8ecc/Change-parameters-to-new-coordinate-system.paper?dl=0&rlkey=lotq1sh6wzkmyttisc05h0in0
        # smplx rotation center is pelvis, which is not origin
        # print(smplx_param['transl'].shape, pelvis.shape)
        pelvis_original = torch.from_numpy(pelvis) - smplx_param['transl'] # pelvis position in original smpl coords system
        smplx_param['transl'] = (smplx_param['transl'] + pelvis_original).matmul(torch.from_numpy(rotation.T)) - pelvis_original
        pelvis = np.dot(pelvis, rotation.T)
        joints = np.dot(joints, rotation.T)
        body_vertices = np.dot(body_vertices, rotation.T)
        r_ori = Rotation.from_rotvec(smplx_param['global_orient'].detach().cpu().numpy().squeeze())
        r_new = Rotation.from_matrix(rotation) * r_ori
        smplx_param['global_orient'] = torch.tensor(r_new.as_rotvec()[None, ...], dtype=torch.float32)
        for key in smplx_param_names:
            smplx_param[key] = smplx_param[key].squeeze()


        object_pointclouds = np.zeros((maximum_atomics, self.num_points, 9), dtype=np.float32)
        for obj_idx, obj_points in enumerate(interaction_objs):
            object_pointclouds[obj_idx, :, :] = obj_points
        # copy last padding
        if len(interaction_objs) < maximum_atomics:
            object_pointclouds[1, :, :] = object_pointclouds[0, :, :]

        record_data = {
            'smplx_param': smplx_param,
            'full_pose': np.concatenate((smplx_param['global_orient'], record['full_poses'][3:])),
            'pelvis': pelvis,
            'joints': joints,
            'body_vertices': body_vertices,
            'contact_dist': record['contact_dist'],
            'sdf': record['sdf'],
            'joint_contact_dist': record['joint_contact_dist'],
            'joint_sdf': record['joint_sdf'],
            'object_pointclouds': object_pointclouds,
            'interaction': record['interaction'],
            'frame_idx':record['frame_idx'],
            'num_atomics': num_atomics,
            'interaction_obj_ids': to_padded_array(specified_obj_idx),
            'verb_ids': to_padded_array(verb_ids),
            'noun_ids': to_padded_array(noun_ids),
            'scene_name': record['scene_name'],
            'centroid': centroid,
            'rotation': rotation,

        }
        return record_data

    def __len__(self):
        return len(self.data)

    def visualize(self):
        # render
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)

        num_records = self.__len__()
        for idx in range(num_records):
            record = self.__getitem__(idx)

            scene = scenes[record['scene_name']]
            body_vertices = record['body_vertices']
            pelvis = record['pelvis']
            joints = record['joints']
            smplx_param = deepcopy(record['smplx_param'])
            global_orient = Rotation.from_rotvec(smplx_param['global_orient'].detach().cpu().numpy().squeeze())

            body_model = smplx.create(smplx_model_folder, model_type='smplx',
                                      gender='neutral', ext='npz',
                                      num_pca_comps=num_pca_comps, batch_size=1)
            geometries = []
            for obj_idx in range(record['num_atomics']):
                obj_points = record['object_pointclouds'][obj_idx]
                obj_vertices, obj_vertex_colors, obj_vertex_normals = obj_points[:, :3], obj_points[:, 3:6], obj_points[:, 6:]
                pointcloud = o3d.geometry.PointCloud(
                    points=o3d.utility.Vector3dVector(obj_vertices),
                )
                pointcloud.normals = o3d.utility.Vector3dVector(obj_vertex_normals)
                pointcloud.colors = o3d.utility.Vector3dVector(obj_vertex_colors)
                geometries.append(pointcloud)

            body_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(body_vertices),
                triangles=o3d.utility.Vector3iVector(body_model.faces),
            )
            body_mesh.paint_uniform_color((0.8, 0.8, 0.8,))
            body_mesh.compute_vertex_normals()
            geometries.append(body_mesh)
            for geometry in geometries:
                vis.add_geometry(geometry)
            vis.poll_events()
            vis.update_renderer()
            vis.clear_geometries()


class CompositeFrameDataset(Dataset):

    def __init__(self, split='train', augment=False, num_points=4096,
                 use_prox_single=False, data_overwrite=False, use_annotate=True,
                 include_motion=False, use_floor_height=False,
                 used_interaction='all', skip_prox_composite=None):
        """
        Args:
            num_points: int, number of points we use to represent each object
            used_interaction: str, select one category of interaction data like 'stand on-floor' or all data using 'all'
            use_floor_height: bool, if set true, recenter the scene coords system origin on the floor by subtracting floor height
            include_motion: bool, whether to include the dynamic interaction such as 'step up-chair', 'lie down-sofa'
            data_overwrite: bool, whether to overwrite data file
            split: ['train', 'test'], dataset split
        """
        self.split = split
        self.augment = augment
        self.use_floor_height = use_floor_height
        self.num_points = num_points

        print('dataset: ', split)

        file_path = Path.joinpath(project_folder, "data", "pelvis_data",
                                  split + '_' + used_interaction + '.pkl')
        # load precomputed data
        if file_path.exists() and not data_overwrite:
            with open(file_path, "rb") as pkl_file:
                cache = pickle.load(pkl_file)
            self.data, self.pointcloud_dict = cache['data'], cache['pointcloud_dict']
        else:
            pointcloud_dict = {}
            for scene in scene_names:
                pointcloud_dict[scene] = {}
            self.data = []
            if use_annotate:
                # use manual data for single interaction
                manual_data_file = project_folder / 'data' / 'frame_data.json'
                with open(manual_data_file, 'r') as f:
                    frame_data = json.load(f)
                scenes_in_split = train_scenes if split == 'train' else test_scenes
                manual_data = [record for record in frame_data if record['instance_id'].split('_')[0] in scenes_in_split]
                self.data = manual_data if used_interaction == 'all' else [record for record in manual_data if
                                                                           record['interaction'] == used_interaction]
                print('added manual labelling')
                self.data = self.data * 32

            # add prox data, use composite data only be default
            prox_data_file = project_folder / "data" / (split + '.pkl')
            with open(prox_data_file, 'rb') as f:
                prox_data = pickle.load(f)
            smplx_params = [to_smplx_input(record['smplx_param']) for record in prox_data]
            smplx_input = {}
            for key in smplx_param_names:
                smplx_input[key] = torch.cat([smplx_param[key] for smplx_param in smplx_params], dim=0)
            body_model = body_model_dict['neutral']
            _, pelvis, _, _ = get_body_by_batch(body_model, smplx_input)
            for record_idx, record in enumerate(prox_data):
                record['pelvis_frame'] = {
                    'rotation': Rotation.from_rotvec(smplx_input['global_orient'][record_idx, :].cpu().numpy()).as_matrix().astype(np.float32),
                    'pelvis': pelvis[record_idx],
                }
            # select whether to use dynamic interactions
            all_interactions = composed_interaction_names
            if use_prox_single:
                atomic_interactions = atomic_interaction_names_include_motion if include_motion else atomic_interaction_names
                all_interactions = all_interactions + atomic_interactions

            if skip_prox_composite == 'all':
                skip_composite_interactions = composed_interaction_names
            elif skip_prox_composite == 'test':
                skip_composite_interactions = test_composed_interaction_names
            else:
                skip_composite_interactions = []
            for interaction in all_interactions:
                if used_interaction != 'all' and interaction != used_interaction:
                    continue
                if interaction in skip_composite_interactions:
                    print('skip ', interaction, 'in dataset')
                    continue
                records = deepcopy(get_interaction_segments(interaction.split('+'), prox_data, mode='verb-noun'))
                if len(records) == 0:
                    print('no data of ', interaction)
                    continue
                print('loading ', interaction, ':', len(records))
                for idx, record in enumerate(records):
                    record['interaction'] = interaction
                # some categories have too few records, try to make category more balanced
                if len(records) < 512:
                    records = records * (512 // len(records))
                self.data += records

            for idx, record in enumerate(self.data):
                scene_name = record['scene_name']
                scene = scenes[scene_name]
                for node_idx in record['interaction_obj_idx']:
                    if node_idx not in pointcloud_dict[scene_name]:
                        obj = scene.get_mesh_with_accessory(node_idx)
                        obj_vertices = np.asarray(obj.vertices, dtype=np.float32)
                        obj_vertex_colors = np.asarray(obj.visual.vertex_colors[:, :3]).astype(np.float32)  # [0, 255]
                        obj_vertex_normals = np.asarray(obj.vertex_normals).astype(np.float32)
                        if split == 'test':  # use uniform sampled pointcloud for validation
                            if obj_vertices.shape[0] > num_points:
                                idx = np.squeeze(farthest_point_sample(torch.from_numpy(obj_vertices[None, :, :]),
                                                                       npoint=num_points))
                                obj_vertices = obj_vertices[idx, :]
                                obj_vertex_colors = obj_vertex_colors[idx, :]
                                obj_vertex_normals = obj_vertex_normals[idx, :]
                            elif obj_vertices.shape[0] < num_points:
                                idx = np.random.choice(np.arange(obj_vertices.shape[0]), num_points, replace=True)
                                obj_vertices = obj_vertices[idx, :]
                                obj_vertex_colors = obj_vertex_colors[idx, :]
                                obj_vertex_normals = obj_vertex_normals[idx, :]
                        pointcloud_dict[scene_name][node_idx] = (obj_vertices, obj_vertex_colors, obj_vertex_normals)
            self.pointcloud_dict = pointcloud_dict

            cache = {'data': self.data,
                     'pointcloud_dict': self.pointcloud_dict
                     }
            file_path.parent.mkdir(exist_ok=True)
            with open(file_path, "wb") as pkl_file:
                pickle.dump(cache, pkl_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
                Return:
                    pelvis: pelvis location
                    pelvis_orient: pelvis orientation
                    object_pointclouds: point clouds of interaction objects
                    interaction: str, interaction description
                    num_atomics: number of atomic interactions
                    interaction_obj_ids: padded instance id of interaction objects
                    verb_ids: padded verb id of verbs in this interaction
                    noun_ids: padded noun id of nouns in this interaction
                    scene_name: name of the scene where this interaction frame is captured
                    centroid: location of the local coord system origin in the scene coord system
                    rotation: rotation from the scene coord system to local coord system
                """
        record = self.data[idx % len(self.data)]
        # interaction as verb-noun of specified atomics
        specified_atomics = record['interaction'].split('+')
        num_atomics = len(specified_atomics)
        verb_ids = []
        noun_ids = []
        for atomic in specified_atomics:
            verb, noun = atomic.split('-')
            verb_id = action_names.index(verb)
            noun_id = category_dict[category_dict['mpcat40'] == noun].index[0]
            verb_ids.append(verb_id)
            noun_ids.append(noun_id)
        pelvis_frame = record['pelvis_frame']
        pelvis, pelvis_orient = np.array(pelvis_frame['pelvis'], dtype=np.float32), np.array(pelvis_frame['rotation'], dtype=np.float32)

        # pointclouds of specified objects
        specified_obj_idx = [record['interaction_obj_idx'][record['interaction_labels'].index(atomic)] for atomic in
                             specified_atomics]
        interaction_objs = []
        for node_idx in specified_obj_idx:
            obj_vertices, obj_vertex_colors, obj_vertex_normals = self.pointcloud_dict[record['scene_name']][node_idx]
            # resample
            num_vertices = obj_vertices.shape[0]
            if self.split == 'train':
                if num_vertices >= self.num_points:
                    idx = np.random.choice(np.arange(num_vertices), self.num_points, replace=False)
                elif num_vertices < self.num_points:
                    idx = np.random.choice(np.arange(num_vertices), self.num_points, replace=True)
                obj_vertices = obj_vertices[idx, :]
                obj_vertex_colors = obj_vertex_colors[idx, :]
                obj_vertex_normals = obj_vertex_normals[idx, :]
                num_vertices = self.num_points

            # normalize positions and add point features
            obj_points = np.zeros((num_vertices, 9),
                                  dtype=np.float32)  # num_point * 9, 3D location + 3D color + 3D normal
            obj_points[:, :3] = obj_vertices
            obj_points[:, 3:6] = obj_vertex_colors / 255.0  # color
            obj_points[:, 6:9] = obj_vertex_normals
            interaction_objs.append(obj_points)

        # to human centric coordinates
        points = []
        for obj_points in interaction_objs:
            points.append(obj_points[:, :3])
        points = np.concatenate(points, axis=0)
        centroid = 0.5 * (np.max(points, axis=0) + np.min(points, axis=0))
        if self.augment:
            centroid = np.float32(centroid + 2.0 * np.random.rand(3) - 1.0)
        if self.use_floor_height:
            floor_height = scenes[record['scene_name']].get_floor_height()
            centroid[2] = floor_height
        for obj_points in interaction_objs:
            obj_points[:, :3] = obj_points[:, :3] - centroid
        pelvis = pelvis - centroid

        # reorient
        rotation = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=np.float32)
        if self.augment:
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation = np.dot(np.array([[cosval, -sinval, 0],
                                        [sinval, cosval, 0],
                                        [0, 0, 1]], dtype=np.float32), rotation)
        # rotation denotes left multiply rotation matrix
        for obj_points in interaction_objs:
            obj_points[:, :3] = np.dot(obj_points[:, :3], rotation.T)
            obj_points[:, 6:] = np.dot(obj_points[:, 6:], rotation.T)
        pelvis = np.dot(rotation, pelvis)
        pelvis_orient = np.dot(rotation, pelvis_orient)


        object_pointclouds = np.zeros((maximum_atomics, self.num_points, 9), dtype=np.float32)
        for obj_idx, obj_points in enumerate(interaction_objs):
            object_pointclouds[obj_idx, :, :] = obj_points
        # copy last padding
        if len(interaction_objs) <maximum_atomics:
            object_pointclouds[1, :, :] = object_pointclouds[0, :, :]

        record_data = {
            'pelvis': pelvis,
            'pelvis_orient':pelvis_orient,
            'object_pointclouds': object_pointclouds,
            'interaction': record['interaction'],
            'num_atomics': num_atomics,
            'interaction_obj_ids': to_padded_array(specified_obj_idx),
            'verb_ids': to_padded_array(verb_ids),
            'noun_ids': to_padded_array(noun_ids),
            'scene_name': record['scene_name'],
            'centroid': centroid,
            'rotation': rotation,

        }
        return record_data

shapenet_to_zup = np.array(
            [[1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )
class PelvisFrameDataset(Dataset):

    def __init__(self, split='train', augment=True,
                 num_points=4096, normalize_obj=False,
                 ):
        """
        Args:
            num_points: int, number of points we use to represent each object
            used_interaction: str, select one category of interaction data like 'stand on-floor' or all data using 'all'
            use_floor_height: bool, if set true, recenter the scene coords system origin on the floor by subtracting floor height
            include_motion: bool, whether to include the dynamic interaction such as 'step up-chair', 'lie down-sofa'
            data_overwrite: bool, whether to overwrite data file
            split: ['train', 'test'], dataset split
        """
        self.split = split
        self.augment = augment
        self.num_points = num_points
        self.normalize_obj = normalize_obj

        print('dataset: ', split)
        # load precomputed data
        self.mesh_dict = {}
        # add prox data, use composite data only be default
        base_dir = Path('/home/kaizhao/projects/gamma')
        data_file = Path.joinpath(base_dir, "data", "pelvis_data",
                                  split + '_filter.pkl')
        with open(data_file, 'rb') as f:
            interaction_data = pickle.load(f)
        smplx_params = [to_smplx_input(record['smplx_param']) for record in interaction_data]
        smplx_input = {}
        for key in smplx_param_names:
            smplx_input[key] = torch.cat([smplx_param[key] for smplx_param in smplx_params], dim=0)
        body_model = body_model_dict['neutral']
        _, pelvis, _, _ = get_body_by_batch(body_model, smplx_input)
        for record_idx, record in enumerate(interaction_data):
            record['pelvis_frame'] = {
                'rotation': Rotation.from_rotvec(smplx_input['global_orient'][record_idx, :].cpu().numpy()).as_matrix().astype(np.float32),
                'pelvis': pelvis[record_idx],
            }
        self.data = interaction_data

        for record in self.data:
            obj_category = record['obj_category']
            obj_id = record['obj_id']
            obj_category_id = obj_category + '_' + obj_id
            if not obj_category_id in self.mesh_dict:
                obj_mesh = trimesh.load(base_dir / 'data' / 'shapenet_real' / obj_category / obj_id / 'model.obj', force='mesh')
                obj_mesh.apply_transform(shapenet_to_zup)
                self.mesh_dict[obj_category_id] = obj_mesh

    def __len__(self):
        return len(self.data) * 32

    def __getitem__(self, idx, visualize=False):
        record = self.data[idx % len(self.data)]
        # interaction as verb-noun of specified atomics
        num_atomics = 1
        verb_ids = [action_names.index(record['action'])]
        pelvis_frame = record['pelvis_frame']
        pelvis, pelvis_orient = np.array(pelvis_frame['pelvis'], dtype=np.float32), np.array(pelvis_frame['rotation'], dtype=np.float32)
        obj_mesh = deepcopy(self.mesh_dict[record['obj_category'] + '_' + record['obj_id']])

        # scale augment
        scaling = (np.random.uniform(size=3) * 2 - 1) * np.array([0.2, 0.2, 0.05]) + 1
        scale_matrix = np.diag(np.array([scaling[0], scaling[1], scaling[2], 1.0]))
        query = trimesh.proximity.ProximityQuery(obj_mesh)
        closest, _, _ = query.on_surface(pelvis.reshape((1, 3)))
        closest = closest.reshape(3)
        diff = pelvis - closest
        pelvis = closest * scaling + diff
        # print(scaling)
        obj_mesh.apply_transform(scale_matrix)

        # flip and rotation augmentation
        rotation = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=np.float32)
        # cannot do random flip, will change pelvis frame from right-handed to left-handed
        # if np.random.uniform() < 0.5:
        #     rotation[0, 0] = -1
        # if np.random.uniform() < 0.5:
        #     rotation[1, 1] = -1

        # random reorient
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation = np.dot(np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]], dtype=np.float32), rotation)

        pelvis = np.dot(rotation, pelvis)
        pelvis_orient = np.dot(rotation, pelvis_orient)
        augment_transform = np.eye(4)
        augment_transform[:3, :3] = rotation
        obj_mesh.apply_transform(augment_transform)
        # recenter
        centroid = 0.5 * (obj_mesh.bounds[0] + obj_mesh.bounds[1])
        centroid[2] = 0
        obj_mesh.vertices -= centroid
        pelvis -= centroid

        extent = np.max(obj_mesh.extents)

        obj_points, _ = trimesh.sample.sample_surface(obj_mesh, self.num_points, sample_color=False)
        obj_points = np.array(obj_points)
        if self.normalize_obj:
            obj_points = obj_points / extent
            pelvis = pelvis / extent
        object_pointclouds = obj_points.reshape((1, self.num_points, 3)).astype(np.float32)
        object_pointclouds = np.tile(object_pointclouds, (maximum_atomics, 1, 1))

        if visualize:
            pelvis_transform = np.eye(4)
            pelvis_transform[:3, :3] = pelvis_orient
            pelvis_transform[:3, 3] = pelvis
            pelvis_mesh = trimesh.creation.axis(transform=pelvis_transform)
            import pyrender
            scene = pyrender.Scene()
            scene.add(pyrender.Mesh.from_trimesh(pelvis_mesh, smooth=False))
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = [1.0, 0.0, 0.0]
            tfs = np.tile(np.eye(4), (len(obj_points), 1, 1))
            tfs[:, :3, 3] = obj_points
            scene.add(pyrender.Mesh.from_trimesh(sm, poses=tfs))
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh))
            pyrender.Viewer(scene, use_raymond_lighting=True)

        record_data = {
            'pelvis': pelvis.astype(np.float32),
            'pelvis_orient':pelvis_orient.astype(np.float32),
            'object_pointclouds': object_pointclouds,
            'action': record['action'],
            'num_atomics': num_atomics,
            'verb_ids': to_padded_array(verb_ids),
            'obj_category': record['obj_category'],
            'obj_id': record['obj_id'],
            'rotation': rotation,
            'centroid': centroid,
            'extent': extent,
        }
        return record_data

def is_left_handed(rotation):
    basis_i, basis_j, basis_k = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    cross_product = np.cross(basis_i, basis_j)
    return np.dot(basis_k, cross_product) < 0

class PelvisFrameCanonicalDataset(Dataset):

    def __init__(self, split='train', augment=True,
                 num_points=4096, normalize_obj=False,
                 ):
        """
        Args:
            num_points: int, number of points we use to represent each object
            used_interaction: str, select one category of interaction data like 'stand on-floor' or all data using 'all'
            use_floor_height: bool, if set true, recenter the scene coords system origin on the floor by subtracting floor height
            include_motion: bool, whether to include the dynamic interaction such as 'step up-chair', 'lie down-sofa'
            data_overwrite: bool, whether to overwrite data file
            split: ['train', 'test'], dataset split
        """
        self.split = split
        self.augment = augment
        self.num_points = num_points
        self.normalize_obj = normalize_obj

        print('dataset: ', split)
        # load precomputed data
        self.mesh_dict = {}
        self.canonical_rotation = {}
        # add prox data, use composite data only be default
        base_dir = Path('/home/kaizhao/projects/gamma')
        data_file = Path.joinpath(base_dir, "data", "pelvis_data",
                                  split + '_filter.pkl')
        with open(data_file, 'rb') as f:
            interaction_data = pickle.load(f)
        smplx_params = [to_smplx_input(record['smplx_param']) for record in interaction_data]
        smplx_input = {}
        for key in smplx_param_names:
            smplx_input[key] = torch.cat([smplx_param[key] for smplx_param in smplx_params], dim=0)
        body_model = body_model_dict['neutral']
        _, pelvis, _, _ = get_body_by_batch(body_model, smplx_input)
        for record_idx, record in enumerate(interaction_data):
            record['pelvis_frame'] = {
                'rotation': Rotation.from_rotvec(smplx_input['global_orient'][record_idx, :].cpu().numpy()).as_matrix().astype(np.float32),
                'pelvis': pelvis[record_idx],
            }
        self.data = interaction_data

        for record in self.data:
            obj_category = record['obj_category']
            obj_id = record['obj_id']
            obj_category_id = obj_category + '_' + obj_id
            if not obj_category_id in self.mesh_dict:
                obj_mesh = trimesh.load(base_dir / 'data' / 'shapenet_real' / obj_category / obj_id / 'model.obj', force='mesh')
                obj_mesh.apply_transform(shapenet_to_zup)
                self.mesh_dict[obj_category_id] = obj_mesh
                with open(base_dir / 'data' / 'shapenet_real' / obj_category / obj_id / 'canonical_rotation.pkl', 'rb') as f:
                    rotation = pickle.load(f).reshape((3, 3))
                if is_left_handed(rotation):
                    rotation[:, 0] = -rotation[:, 0]
                self.canonical_rotation[obj_category_id] = rotation

    def __len__(self):
        return len(self.data) * 32

    def __getitem__(self, idx, visualize=False):
        record = self.data[idx % len(self.data)]
        # interaction as verb-noun of specified atomics
        num_atomics = 1
        verb_ids = [action_names.index(record['action'])]
        pelvis_frame = record['pelvis_frame']
        pelvis, pelvis_orient = np.array(pelvis_frame['pelvis'], dtype=np.float32), np.array(pelvis_frame['rotation'], dtype=np.float32)
        obj_mesh = deepcopy(self.mesh_dict[record['obj_category'] + '_' + record['obj_id']])

        # scale augment
        scaling = (np.random.uniform(size=3) * 2 - 1) * np.array([0.2, 0.2, 0.05]) + 1
        scale_matrix = np.diag(np.array([scaling[0], scaling[1], scaling[2], 1.0]))
        query = trimesh.proximity.ProximityQuery(obj_mesh)
        closest, _, _ = query.on_surface(pelvis.reshape((1, 3)))
        closest = closest.reshape(3)
        diff = pelvis - closest
        pelvis = closest * scaling + diff
        # print(scaling)
        obj_mesh.apply_transform(scale_matrix)

        # random reorient
        rotation = deepcopy(self.canonical_rotation[record['obj_category'] + '_' + record['obj_id']])
        # rotation = np.eye(3)

        pelvis = np.dot(rotation, pelvis)
        pelvis_orient = np.dot(rotation, pelvis_orient)
        augment_transform = np.eye(4)
        augment_transform[:3, :3] = rotation
        obj_mesh.apply_transform(augment_transform)
        # recenter
        centroid = np.zeros(3)
        # centroid = 0.5 * (obj_mesh.bounds[0] + obj_mesh.bounds[1])
        # centroid[2] = 0
        # obj_mesh.vertices -= centroid
        # pelvis -= centroid

        extent = np.max(obj_mesh.extents)

        obj_points, _ = trimesh.sample.sample_surface(obj_mesh, self.num_points, sample_color=False)
        obj_points = np.array(obj_points)
        if self.normalize_obj:
            obj_points = obj_points / extent
            pelvis = pelvis / extent
        object_pointclouds = obj_points.reshape((1, self.num_points, 3)).astype(np.float32)
        object_pointclouds = np.tile(object_pointclouds, (maximum_atomics, 1, 1))

        if visualize:
            pelvis_transform = np.eye(4)
            pelvis_transform[:3, :3] = pelvis_orient
            pelvis_transform[:3, 3] = pelvis
            pelvis_mesh = trimesh.creation.axis(transform=pelvis_transform)
            import pyrender
            scene = pyrender.Scene()
            scene.add(pyrender.Mesh.from_trimesh(pelvis_mesh, smooth=False))
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = [1.0, 0.0, 0.0]
            tfs = np.tile(np.eye(4), (len(obj_points), 1, 1))
            tfs[:, :3, 3] = obj_points
            scene.add(pyrender.Mesh.from_trimesh(sm, poses=tfs))
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh))
            pyrender.Viewer(scene, use_raymond_lighting=True)

        record_data = {
            'pelvis': pelvis.astype(np.float32),
            'pelvis_orient':pelvis_orient.astype(np.float32),
            'object_pointclouds': object_pointclouds,
            'action': record['action'],
            'num_atomics': num_atomics,
            'verb_ids': to_padded_array(verb_ids),
            'obj_category': record['obj_category'],
            'obj_id': record['obj_id'],
            'rotation': rotation,
            'centroid': centroid,
            'extent': extent,
        }
        return record_data

class InteractionFeatureDataset(Dataset):
    """Dataset of SMPLX body with POSA contact features. Only used in the baseline POSA-I."""
    def __init__(self, interaction_data, split='train', obj_code_type='onehot', verb_code_type='onehot', use_augment=True,
                 num_points=4096, center_type='random', scale_obj=False, normalize=False, used_interaction='all', used_instance=None,
                 point_sample='random', exclude=None,
                 skip_composite='no',
                 single_frame=None, keep_frame=None, data_overwrite=False):

        file_path = Path.joinpath(project_folder, "data", "feature_dataset", split + '_' + used_interaction + '.pkl')
        # load precomputed data
        if file_path.exists() and not data_overwrite:
            with open(file_path, "rb") as pkl_file:
                cache = pickle.load(pkl_file)
            data, pointcloud_dict = cache['data'], cache['pointcloud_dict']
        else:
            data = []
            body_model = body_model_dict['neutral']
            pointcloud_dict = {}
            for scene in scene_names:
                pointcloud_dict[scene] = {}
            statistics = {}
            for interaction in interaction_names:
                if used_interaction != 'all' and interaction != used_interaction:
                    continue
                records = deepcopy(get_interaction_segments(interaction.split('+'), interaction_data, mode='verb-noun'))
                print('loading ', interaction, ':', len(records))
                statistics[interaction] = len(records)
                if len(records) == 0:
                    continue
                smplx_params = [to_smplx_input(record['smplx_param']) for record in records]
                frame_scene_names = [record['scene_name'] for record in records]
                smplx_input = {}
                for key in smplx_param_names:
                    smplx_input[key] = torch.cat([smplx_param[key] for smplx_param in smplx_params], dim=0)
                # print('get vertices')
                body_vertices, pelvis, joints = get_body_by_batch(body_model, smplx_input)
                vertex_obj_dists = get_vertex_obj_dists(body_vertices, frame_scene_names)
                # print('vertices got')
                for idx, record in enumerate(records):
                    record['interaction'] = interaction
                    record['body_vertices'] = body_vertices[idx]
                    record['vertex_obj_dists'] = vertex_obj_dists[idx]
                    record['pelvis'] = pelvis[idx]
                    record['joints'] = joints[idx, :55, :]
                    record['smplx_param'] = {}
                    # need to copy param, do not do anything related to cuda in getitem which can cause multiprocess related error need manual handle
                    for key in smplx_param_names:
                        record['smplx_param'][key] = smplx_params[idx][key].cpu()
                    scene_name = record['scene_name']
                    scene = scenes[scene_name]

                    record['node_idx'] = []
                    record['verb_code'] = np.zeros(num_verb, dtype=np.float32)
                    record['noun_code'] = np.zeros(num_noun, dtype=np.float32)
                    record['interaction_code'] = np.zeros(num_noun * num_verb, dtype=np.float32)
                    for atomic_interaction in interaction.split('+'):
                        verb, noun = atomic_interaction.split('-')
                        atomic_idx = record['interaction_labels'].index(atomic_interaction)
                        node_idx = record['interaction_obj_idx'][atomic_idx]
                        node_category = scene.object_nodes[node_idx].category
                        verb_id = action_names.index(verb)
                        verb_code = np.eye(num_verb, dtype=np.float32)[verb_id]
                        noun_id = category_dict[category_dict['mpcat40'] == noun].index[0]
                        noun_code = np.eye(num_noun, dtype=np.float32)[noun_id]
                        interaction_code = np.kron(verb_code, noun_code)
                        record['node_idx'].append(str(node_idx))
                        record['verb_code'] += verb_code
                        record['noun_code'] += noun_code
                        record['interaction_code'] += interaction_code
                    record['node_idx'] = '_'.join(record['node_idx'])

                # some categories have too few records, try to make category more balanced
                if len(records) < 512:
                    records = records * (512 // len(records))
                data += records
                # break
            print(statistics)
            cache = {'data': data,
                     'pointcloud_dict': pointcloud_dict
                     }
            file_path.parent.mkdir(exist_ok=True)
            with open(file_path, "wb") as pkl_file:
                pickle.dump(cache, pkl_file)

        if keep_frame is None:
            self.data = data
        else:
            frame_num = len(data)
            keep_idx = np.random.choice(np.arange(frame_num), keep_frame, replace=False)
            self.data = []
            for idx in keep_idx:
                self.data.append(data[idx])

        if exclude is not None:
            self.data = [record for record in self.data if record['interaction'] not in exclude]
            print('used interactions:', set([record['interaction'] for record in self.data]))

        if skip_composite != 'no':
            self.data = [record for record in self.data if '+' not in record['interaction']]
            print('used interactions:', set([record['interaction'] for record in self.data]))

        self.pointcloud_dict = pointcloud_dict
        self.obj_code_type = obj_code_type
        self.verb_code_type = verb_code_type
        self.use_augment = use_augment
        self.num_points = num_points
        self.center_type = center_type
        self.normalize = normalize
        self.single_frame = single_frame
        self.scale_obj = scale_obj
        self.point_sample = point_sample



    def __getitem__(self, idx):
        # record = self.data[-1]
        record = self.data[idx] if self.single_frame is None else self.data[self.single_frame]
        scene = scenes[record['scene_name']]
        body_vertices = record['body_vertices']
        pelvis = record['pelvis']
        joints = record['joints']
        node_idx = record['node_idx']
        smplx_param = deepcopy(record['smplx_param'])

        for key in smplx_param_names:
            smplx_param[key] = smplx_param[key].squeeze()
        return smplx_param, pelvis, joints, body_vertices - pelvis.reshape((1, 3)), \
               record['vertex_obj_dists'], record['interaction'], record['verb_code'], record['noun_code'], record[
                   'interaction_code'], record['scene_name'], record['node_idx']
                    # obj_points, centroid, scale, rotation, \

    def __len__(self):
        return len(self.data)
        # return min(len(self.data), 16)



if __name__ == "__main__":
    # # data
    # with open(Path.joinpath(project_folder, "data", 'train.pkl'), 'rb') as data_file:
    #     train_data = pickle.load(data_file)
    # with open(Path.joinpath(project_folder, "data", 'test.pkl'), 'rb') as data_file:
    #     test_data = pickle.load(data_file)
    #
    from torch.utils.data import DataLoader
    # from argparse import ArgumentParser, Namespace
    #
    # parser = ArgumentParser()
    # parser.add_argument("--num_joints", type=int, default=22)
    # parser.add_argument("--point_level", type=int, default=3)
    # parser.add_argument("--num_points", type=int, default=4096)
    # parser.add_argument("--used_interaction", type=str, default='lie on-bed')
    # parser.add_argument("--used_instance", type=str, default=None)
    # parser.add_argument("--scale_obj", type=int, default=1)
    # parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--center_type", type=str, default='obj')
    # parser.add_argument("--point_sample", type=str, default='random')
    # args = parser.parse_args()
    #
    # train_dataset = InteractionDataset(train_data, num_points=args.num_points, use_augment=True,
    #                                     used_interaction=args.used_interaction, split='train',
    #                                     center_type=args.center_type, scale_obj=args.scale_obj,
    #                                     used_instance=args.used_instance,
    #                                     data_overwrite=True, point_sample=args.point_sample,
    #                                     )
    # train_dataset.visualize()
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
    #                           drop_last=True, pin_memory=False)  # pin_memory cause warning in pytorch 1.9.0
    # for batch in train_loader:
    #     print(batch)
    #     # break
    train_dataset = PelvisFrameCanonicalDataset(split='train', num_points=4096)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True,
                              drop_last=True, pin_memory=False)  # pin_memory cause warning in pytorch 1.9.0
    for idx in range(len(train_dataset.data)):
        print(idx)
        # print(train_dataset.data[idx])
        train_dataset.__getitem__(idx, visualize=True)
    for batch in tqdm(train_loader):
        print(batch)

