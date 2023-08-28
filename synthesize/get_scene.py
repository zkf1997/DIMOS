import os
import pickle

from plyfile import PlyData, PlyElement
import json
import trimesh
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from scipy.spatial import KDTree
from pathlib import Path
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import torch.nn.functional as F

def to_open3d(trimesh_mesh):
    trimesh_mesh = deepcopy(trimesh_mesh)  # if not copy, vertex normal cannot be assigned
    o3d_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
                                         triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces))
    # as_open3d method not working for color and normal
    if hasattr(trimesh_mesh.visual, 'vertex_colors'):
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(trimesh_mesh.visual.vertex_colors[:, :3] / 255.0)
    o3d_mesh.compute_vertex_normals()  # if not compute but only assign trimesh normals, the normal rendering fails, not sure about the reason
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(trimesh_mesh.vertex_normals)
    # input_normals = trimesh_mesh.vertex_normals
    # print('normal_diff:', input_normals - np.asarray(o3d_mesh.vertex_normals))
    return o3d_mesh

unity_to_zup = np.array(
            [[-1, 0, 0, 0],
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
shapenet_to_zup = np.array(
    [[1, 0, 0, 0],
     [0, 0, -1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]]
)
from mesh_to_sdf import mesh_to_voxels, get_surface_point_cloud
import skimage
import pyrender
class ShapenetScene:
    def __init__(self, obj_path, build=False, voxel_resolution = 128):
        self.obj_path = obj_path
        self.obj_category = obj_path.parent.parent.name
        self.obj_id = obj_path.parent.name
        self.obj_folder = obj_path.parent
        self.name = obj_path.parent.name
        self.mesh = trimesh.load_mesh(self.obj_path, force='mesh')  # process=True will change vertices and cause error!
        self.mesh.apply_transform(shapenet_to_zup)
        self.floor_height = 0

        # load sdf, https: // github.com / mohamedhassanmus / POSA / blob / de21b40f22316cfb02ec43021dc5f325547c41ca / src / data_utils.py  # L99
        self.sdf_path = self.obj_folder / ('sdf.pkl')
        if build or not self.sdf_path.exists():
            extents = self.mesh.bounding_box.extents
            floor_mesh = trimesh.creation.box(extents=np.array([extents[0] + 2, extents[1] + 2, 0.5]),
                                              transform=np.array([[1.0, 0.0, 0.0, 0],
                                                                  [0.0, 1.0, 0.0, 0],
                                                                  [0.0, 0.0, 1.0, -0.25],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            scene_mesh = self.mesh + floor_mesh
            # scene_mesh.show()
            scene_extents = extents + np.array([2, 2, 1])
            scene_scale = np.max(scene_extents) * 0.5
            scene_centroid = self.mesh.bounding_box.centroid
            scene_mesh.vertices -= scene_centroid
            scene_mesh.vertices /= scene_scale
            sign_method = 'normal'
            surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method='scan', bounding_radius=3 ** 0.5, scan_count=100,
                                                          scan_resolution=400, sample_point_count=10000000, calculate_normals=(sign_method == 'normal'))

            sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth', sample_count=11, pad=False,
                                                  check_result=False, return_gradients=True)
            sdf_dict = {
                'grid': sdf_grid * scene_scale,
                'gradient_grid': gradient_grid,
                'dim': voxel_resolution,
                'centroid': scene_centroid,
                'scale': scene_scale,
            }
            with open(self.sdf_path, 'wb') as f:
                pickle.dump(sdf_dict, f)
        else:
            with open(self.sdf_path, 'rb') as f:
                sdf_dict = pickle.load(f)
        self.sdf_dict = sdf_dict


    def calc_sdf(self, vertices):
        if not hasattr(self, 'sdf_torch'):
            self.sdf_torch = torch.from_numpy(self.sdf_dict['grid']).to(dtype=torch.float32).squeeze().unsqueeze(0).unsqueeze(0) # 1x1xDxDxD
        sdf_grids = self.sdf_torch.to(vertices.device)
        sdf_centroid = torch.tensor(self.sdf_dict['centroid']).reshape(1, 1, 3).to(device=vertices.device, dtype=torch.float32)

        # vertices = torch.tensor(vertices).reshape(1, -1, 3)
        batch_size, num_vertices, _ = vertices.shape
        vertices = ((vertices - sdf_centroid) / self.sdf_dict['scale'])
        sdf_values = F.grid_sample(sdf_grids,
                                       vertices[:, :, [2, 1, 0]].view(-1, num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                       padding_mode='border',
                                       align_corners=True
                                       # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                       )
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
        return sdf_values.reshape(batch_size, num_vertices)

    def get_floor_height(self):
        return 0

    def visualize(self):
        vertices, faces, normals, _ = skimage.measure.marching_cubes(self.sdf_dict['grid'], level=0)
        vertices = vertices / self.sdf_dict['dim'] * 2 - 1
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        scene = pyrender.Scene()
        scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
        sphere = trimesh.creation.uv_sphere(radius=0.02)
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[0, :3, 3] = np.array([1, 1, 1])
        poses[1, :3, 3] = -np.array([1, 1, 1])
        scene.add(pyrender.Mesh.from_trimesh(sphere, poses=np.array(poses)))
        pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

class ObjectScene(ShapenetScene):
    def __init__(self, obj_path, obj_category='chair', obj_id=0, build=False, voxel_resolution = 128):
        self.obj_path = obj_path
        self.obj_category = obj_category
        self.obj_id = obj_id
        self.obj_folder = obj_path.parent
        self.name = '{}_{}_{}'.format(obj_path.parent.name, obj_category, obj_id)
        self.mesh = trimesh.load(self.obj_path, force='mesh')  # process=True will change vertices and cause error!
        if 'obj' in self.obj_path.name or 'glb' in self.obj_path.name:
            self.mesh.apply_transform(shapenet_to_zup)
        self.floor_height = 0

        # load sdf, https: // github.com / mohamedhassanmus / POSA / blob / de21b40f22316cfb02ec43021dc5f325547c41ca / src / data_utils.py  # L99
        self.sdf_path = self.obj_folder / ('{}_{}_sdf_grad.pkl'.format(obj_category, obj_id))
        if build or not self.sdf_path.exists():
            extents = self.mesh.bounding_box.extents
            extents = np.array([extents[0] + 2, extents[1] + 2, 0.5])
            transform = np.array([[1.0, 0.0, 0.0, 0],
                                  [0.0, 1.0, 0.0, 0],
                                  [0.0, 0.0, 1.0, -0.25],
                                  [0.0, 0.0, 0.0, 1.0],
                                  ])
            transform[:2, 3] += self.mesh.centroid[:2]
            floor_mesh = trimesh.creation.box(extents=extents,
                                              transform=transform,
                                              )
            scene_mesh = self.mesh + floor_mesh
            # scene_mesh.show()
            scene_extents = extents + np.array([2, 2, 1])
            scene_scale = np.max(scene_extents) * 0.5
            scene_centroid = self.mesh.bounding_box.centroid
            scene_mesh.vertices -= scene_centroid
            scene_mesh.vertices /= scene_scale
            sign_method = 'normal'
            surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method='scan', bounding_radius=3 ** 0.5, scan_count=100,
                                                          scan_resolution=400, sample_point_count=10000000, calculate_normals=(sign_method == 'normal'))

            sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth', sample_count=11, pad=False,
                                                  check_result=False, return_gradients=True)
            sdf_dict = {
                'grid': sdf_grid * scene_scale,
                'gradient_grid': gradient_grid,
                'dim': voxel_resolution,
                'centroid': scene_centroid,
                'scale': scene_scale,
            }
            with open(self.sdf_path, 'wb') as f:
                pickle.dump(sdf_dict, f)
        else:
            with open(self.sdf_path, 'rb') as f:
                sdf_dict = pickle.load(f)
        self.sdf_dict = sdf_dict


class GeneralScene(ShapenetScene):
    def __init__(self, scene_path, scene_name, build=False, voxel_resolution = 256):
        scene_path = Path(scene_path)
        self.obj_path = scene_path
        self.obj_folder = scene_path.parent
        self.name = scene_name
        self.mesh = trimesh.load(self.obj_path, force='mesh')  # process=True will change vertices and cause error!
        if 'obj' in self.obj_path.name or 'glb' in self.obj_path.name:
            self.mesh.apply_transform(shapenet_to_zup)
        self.floor_height = 0

        # load sdf, https: // github.com / mohamedhassanmus / POSA / blob / de21b40f22316cfb02ec43021dc5f325547c41ca / src / data_utils.py  # L99
        self.sdf_path = self.obj_folder / ('{}_sdf_grad.pkl'.format(scene_path.name.split('.')[0]))
        if build or not self.sdf_path.exists():
            extents = self.mesh.bounding_box.extents
            extents = np.array([extents[0] + 2, extents[1] + 2, 0.5])
            transform = np.array([[1.0, 0.0, 0.0, 0],
                                  [0.0, 1.0, 0.0, 0],
                                  [0.0, 0.0, 1.0, -0.25],
                                  [0.0, 0.0, 0.0, 1.0],
                                  ])
            transform[:2, 3] += self.mesh.centroid[:2]
            floor_mesh = trimesh.creation.box(extents=extents,
                                              transform=transform,
                                              )
            scene_mesh = self.mesh + floor_mesh
            # scene_mesh.show()
            scene_extents = extents + np.array([2, 2, 1])
            scene_scale = np.max(scene_extents) * 0.5
            scene_centroid = self.mesh.bounding_box.centroid
            scene_mesh.vertices -= scene_centroid
            scene_mesh.vertices /= scene_scale
            sign_method = 'normal'
            surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method='scan', bounding_radius=3 ** 0.5, scan_count=100,
                                                          scan_resolution=400, sample_point_count=10000000, calculate_normals=(sign_method == 'normal'))

            sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth', sample_count=11, pad=False,
                                                  check_result=False, return_gradients=True)
            # vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf_grid, level=0)
            # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
            # mesh.show()
            sdf_dict = {
                'grid': sdf_grid * scene_scale,
                'gradient_grid': gradient_grid,
                'dim': voxel_resolution,
                'centroid': scene_centroid,
                'scale': scene_scale,
            }
            with open(self.sdf_path, 'wb') as f:
                pickle.dump(sdf_dict, f)
        else:
            with open(self.sdf_path, 'rb') as f:
                sdf_dict = pickle.load(f)
        self.sdf_dict = sdf_dict

def quad_to_trimesh(quad_mesh_path):
    quad_mesh = PlyData.read(quad_mesh_path)
    vertices = quad_mesh['vertex']
    quad_faces = quad_mesh['face']
    vertex_indices = []
    object_id = []
    for face_idx in tqdm(range(quad_faces.count)):
        face_vertices = quad_faces['vertex_indices'][face_idx]
        if len(face_vertices) == 3:
            vertex_indices.append(face_vertices)
            object_id.append(quad_faces['object_id'][face_idx])
        elif len(face_vertices == 4):
            vertex_indices.append(face_vertices[:3])
            vertex_indices.append([face_vertices[2], face_vertices[3], face_vertices[0]])
            object_id.append(quad_faces['object_id'][face_idx])
            object_id.append(quad_faces['object_id'][face_idx])
    tri_faces = np.array(
        list(zip(vertex_indices, object_id)),
        dtype=[('vertex_indices', 'i4', (3,)),
               ('object_id', 'i4')]
    )
    tri_faces = PlyElement.describe(tri_faces, 'face')
    tri_mesh = PlyData([quad_mesh['vertex'], tri_faces],
            text=False, byte_order='<')
    tri_mesh_path = quad_mesh_path.replace('.ply', '_tri.ply')
    tri_mesh.write(tri_mesh_path)

def bbox_intersect(aabb1, aabb2):
    min_bound = np.maximum(np.asarray(aabb1[0, :]), np.asarray(aabb2[0, :]))
    max_bound = np.minimum(np.asarray(aabb1[1, :]), np.asarray(aabb2[1, :]))
    return (min_bound[:2] < max_bound[:2]).any()

class ReplicaScene:
    def __init__(self, scene_name, replica_folder, build=False, zero_floor=False):
        self.name = scene_name
        self.replica_folder = replica_folder
        self.room_folder = replica_folder / scene_name
        self.instance_folder = replica_folder / scene_name / 'instances'
        self.ply_path = replica_folder / scene_name / 'habitat' / 'mesh_semantic.ply'
        self.mesh = trimesh.load_mesh(self.ply_path, process=False)  # process=True will change vertices and cause error!

        # per instance category labels
        json_path = os.path.join(replica_folder, scene_name, 'habitat', 'info_semantic.json')
        with open(json_path, 'r') as f:
            self.semantic_mapping = json.load(f)
        self.instance_to_category = self.semantic_mapping['id_to_label']
        self.num_instances = len(self.semantic_mapping['id_to_label'])
        self.category_ids = [self.instance_to_category[instance_id] for instance_id in range(self.num_instances)]
        self.category_names = [self.semantic_mapping['classes'][category_id - 1]['name'] for category_id in
                               self.category_ids]
        self.instances = self.get_instances(build=build)



        # load sdf, https: // github.com / mohamedhassanmus / POSA / blob / de21b40f22316cfb02ec43021dc5f325547c41ca / src / data_utils.py  # L99
        replica_sdf_folder = Path(replica_folder) / 'replica_sdf'
        with open(replica_sdf_folder / scene_name / (scene_name + '.json'), 'r') as f:
            sdf_data = json.load(f)
            grid_dim = sdf_data['dim']
            grid_min = np.array(sdf_data['min']).astype(np.float32)
            grid_max = np.array(sdf_data['max']).astype(np.float32)
            voxel_size = (grid_max - grid_min) / grid_dim
        sdf = np.load(replica_sdf_folder / scene_name / (scene_name + '_sdf.npy')).astype(np.float32)
        sdf = sdf.reshape((grid_dim, grid_dim, grid_dim, 1))
        self.sdf = sdf
        self.sdf_config = {'grid_min': grid_min, 'grid_max': grid_max, 'grid_dim': grid_dim}

        self.floor_height = self.raw_floor_height = floor_height = self.get_floor_height()
        if zero_floor:
            self.mesh.vertices[:, 2] -= floor_height
            for instance in self.instances:
                instance.vertices[:, 2] -= floor_height
            self.sdf_config['grid_min'] -= np.array([0, 0, floor_height])
            self.sdf_config['grid_max'] -= np.array([0, 0, floor_height])
            self.floor_height = self.get_floor_height()

        self.mesh_with_accessory = {}


    def get_instances(self, visualize=False, build=False):
        semantic_mapping = self.semantic_mapping
        instance_to_category = semantic_mapping['id_to_label']
        num_instances = len(semantic_mapping['id_to_label'])
        if not os.path.exists(self.instance_folder) or build:
            instance_meshes = []
            os.makedirs(self.instance_folder, exist_ok=True)
            # per face instance labels
            # https: // github.com / facebookresearch / Replica - Dataset / issues / 17  # issuecomment-538757418
            file_in = PlyData.read(self.ply_path)
            faces_in = file_in.elements[1]
            objects_vertices = {}
            for f in faces_in:
                object_id = f[1]
                if not object_id in objects_vertices:
                    objects_vertices[object_id] = []
                objects_vertices[object_id] += list(f[0])
            for key in objects_vertices:
                objects_vertices[key] = list(set(objects_vertices[key]))

            for instance_id in range(num_instances):
                instance_path = os.path.join(self.instance_folder, str(instance_id) + '.ply')
                category_id = instance_to_category[instance_id]
                category_name = semantic_mapping['classes'][category_id - 1]['name']
                if category_id < 0:  # empty instance
                    instance_mesh = trimesh.Trimesh()
                else:
                    vertex_ids = np.array(objects_vertices[instance_id], dtype=np.int32)
                    vertex_valid = np.zeros(len(self.mesh.vertices))
                    vertex_valid[vertex_ids] = 1
                    face_ids = np.nonzero(vertex_valid[self.mesh.faces].sum(axis=-1) == 3)[0]
                    instance_mesh = self.mesh.submesh([face_ids], append=True)
                instance_mesh.export(instance_path)
                instance_meshes.append(instance_mesh)  # directly return submesh cannot be converted to open3d, error vertex normal not writable
                # if visualize:
                #     vis_mesh = deepcopy(instance_mesh)
                #     vis_mesh.visual.vertex_colors = np.array([255, 0, 0, 255])
                #     (self.mesh + vis_mesh).show()

        instance_meshes = []
        for instance_id in range(num_instances):
            instance_path = os.path.join(self.instance_folder, str(instance_id) + '.ply')
            scene_or_mesh = trimesh.load_mesh(instance_path, process=False)
            if isinstance(scene_or_mesh, trimesh.Scene):
                if len(scene_or_mesh.geometry) == 0:
                    instance_mesh = trimesh.Trimesh()  # empty scene
                else:
                    # we lose texture information here
                    instance_mesh = trimesh.util.concatenate(
                        tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                              for g in scene_or_mesh.geometry.values()))
            else:
                instance_mesh = scene_or_mesh
            instance_meshes.append(instance_mesh)
        return instance_meshes

    def get_floor_height(self):
        floor_pointclouds = [np.asarray(self.instances[instance_id].vertices) for instance_id in range(self.num_instances) if self.category_names[instance_id] == 'floor']
        if len(floor_pointclouds) == 0:
            floor_height = self.mesh.bounds[0, 2]
        else:
            max_idx = np.argmax(np.array(
                [pointcloud.shape[0] for pointcloud in floor_pointclouds]
            ))
            max_floor = floor_pointclouds[max_idx]
            floor_height = max_floor[:, 2].mean()  # mean of z coord of points of max sized floor
        return floor_height

    def get_mesh_with_accessory(self, instance_id, visualize=False):
        if instance_id in self.mesh_with_accessory:
            return self.mesh_with_accessory[instance_id]

        obj_mesh = self.instances[instance_id]
        category_name = self.category_names[instance_id]
        if category_name in ['sofa', 'chair', 'bed', 'table', 'cabinet', 'chest_of_drawers']:
            accessory_list = ['objects', 'object']
            if category_name == 'table':
                accessory_list += ['tv_monitor']
            if category_name in ['sofa', 'bed', 'couch', 'chair']:
                accessory_list += ['cushion', 'pillow', 'pillows', 'comforter', 'blanket']
            accessory_candidates = [instance_id for instance_id in range(self.num_instances) if self.category_names[instance_id] in accessory_list]
            if len(accessory_candidates):
                proximity = KDTree(obj_mesh.vertices)
                for candidate in accessory_candidates:
                    dists, _ = proximity.query(np.asarray(self.instances[candidate].vertices))
                    if dists.min() < 0.1:
                        obj_mesh += self.instances[candidate]

        self.mesh_with_accessory[instance_id] = obj_mesh
        if visualize:
            vis_mesh = deepcopy(obj_mesh)
            vis_mesh.visual.vertex_colors = np.array([255, 0, 0, 255])
            (self.mesh + vis_mesh).show()
        return obj_mesh

    def calc_sdf(self, vertices):
        if not hasattr(self, 'sdf_torch'):
            self.sdf_torch = torch.from_numpy(self.sdf).squeeze().unsqueeze(0).unsqueeze(0) # 1x1xDxDxD
        sdf_grids = self.sdf_torch.to(vertices.device)
        sdf_config = self.sdf_config
        sdf_max = torch.tensor(sdf_config['grid_max']).reshape(1, 1, 3).to(vertices.device)
        sdf_min = torch.tensor(sdf_config['grid_min']).reshape(1, 1, 3).to(vertices.device)

        # vertices = torch.tensor(vertices).reshape(1, -1, 3)
        batch_size, num_vertices, _ = vertices.shape
        vertices = ((vertices - sdf_min)
                         / (sdf_max - sdf_min) * 2 - 1)
        sdf_values = F.grid_sample(sdf_grids,
                                       vertices[:, :, [2, 1, 0]].view(-1, num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                       padding_mode='border',
                                       align_corners=True
                                       # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                       )
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
        return sdf_values.reshape(batch_size, num_vertices)

    def visualize(self):
        app = gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
        vis.show_settings = True
        for instance_id, instance in enumerate(self.instances):
            if type(instance) == trimesh.Scene or len(instance.vertices) == 0:
                continue
            instance_category = self.category_names[instance_id]
            # print(instance_id, instance_category)
            instance_o3d = to_open3d(instance)
            vis.add_geometry(str(instance_id), instance_o3d)
            vis.add_geometry(str(instance_id) + '_bbox', instance_o3d.get_axis_aligned_bounding_box())
            instance_bound = instance.bounds
            text_loc = (instance_bound[0] + instance_bound[1]) * 0.5
            text_loc[2] = instance_bound[1, 2]
            vis.add_3d_label(text_loc, "{}_{}".format(instance_category, instance_id))
        vis.reset_camera_to_default()

        app.add_window(vis)
        app.run()

def export_scene(scene_name, obj_list):
    scene = ReplicaScene(scene_name, replica_folder, build=False, zero_floor=True)
    scene_mesh = scene.mesh
    scene_mesh.apply_transform(zup_to_shapenet)
    scene_mesh.export(replica_folder / scene_name / (scene_name + '.obj'),
                      file_type='obj')
    for obj_id in obj_list:
        obj_mesh = scene.get_mesh_with_accessory(obj_id)
        transl = np.zeros(3)
        transl[:2] = 0.5 * (obj_mesh.bounds[0, :2] + obj_mesh.bounds[1, :2])
        transl[2] = obj_mesh.bounds[0, 2]
        obj_mesh.vertices -= transl
        obj_mesh.apply_transform(zup_to_shapenet)
        obj_mesh.export(scene.instance_folder / (scene.category_names[obj_id] + '_' + str(obj_id) + '.obj'),
                        file_type='obj')

# scene_dir = '/mnt/atlas_root/vlg-nfs/kaizhao/datasets/replica/room_0/habitat/'
# quad_to_trimesh(scene_dir + 'mesh_semantic.ply')
#
# with open(scene_dir + 'info_semantic.json') as f:
#     info = json.load(f)

replica_folder = Path('data/replica')
if __name__ == "__main__":

    scene_name = 'room_0'
    export_ids = [73, 74, 39, 41, 9, 77]
    # build from original scene
    scene = ReplicaScene(scene_name, replica_folder, build=True, zero_floor=False)
    print(scene.floor_height)
    scene_floor = deepcopy(scene.mesh)
    scene_floor.vertices[:, 2] -= scene.floor_height
    scene_floor.export(replica_folder / scene_name / 'mesh_floor.ply')
    # build from translated scene, floor at z=0
    scene = ReplicaScene(scene_name, replica_folder, build=False, zero_floor=True)
    print(scene.floor_height)
    for obj_id in export_ids:
        obj_category = scene.category_names[obj_id]
        instance_mesh = scene.get_mesh_with_accessory(obj_id)
        instance_mesh.export(replica_folder / scene_name / 'instances' / f'{obj_category}_{obj_id}.ply')

    import sys
    sys.path.append(os.getcwd())
    sys.path.append('./coins')
    sys.path.append('./coins/interaction')
    from data.scene import scenes, to_trimesh
    scene_name = 'MPH8'
    scene_dir = Path('data/PROX') / scene_name
    scene_dir.mkdir(exist_ok=True, parents=True)
    export_ids = [9]
    scene = scenes[scene_name]
    scene_floor = to_trimesh(scene.mesh)
    scene_floor.vertices[:, 2] -= scene.get_floor_height()
    scene_floor.export( scene_dir / 'scene_floor.ply')
    for obj_id in export_ids:
        obj_category = scene.object_nodes[obj_id].category_name
        instance_mesh = deepcopy(scene.get_mesh_with_accessory(obj_id))
        instance_mesh.vertices[:, 2] -= scene.get_floor_height()
        instance_mesh.export(scene_dir / f'{obj_category}_{obj_id}.ply')


