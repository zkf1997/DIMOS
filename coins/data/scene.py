import os
import sys
import time

sys.path.append('..')
from configuration.config import *

import copy
from copy import deepcopy
import json
import os
import trimesh
import pickle
import torch
from collections import defaultdict
from scipy.spatial import KDTree

import numpy as np
import torch.nn.functional as F
import open3d as o3d
from data.scene_registration import prox_to_posa

DEBUG = False

def sample_box(bbox, mesh_grid_step=0.2, z_min=-0.5, z_max=2.3):
    X, Y, Z = np.meshgrid(np.arange(bbox[0], bbox[2], mesh_grid_step),
                          np.arange(bbox[1], bbox[3], mesh_grid_step),
                          np.arange(z_min, z_max, mesh_grid_step),
                          )
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
    return points

def to_trimesh(o3d_mesh):
    """
    convert open3d mesh to trimesh mesh
    """
    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices), faces=np.asarray(o3d_mesh.triangles),
        vertex_colors=np.asarray(o3d_mesh.vertex_colors),
        vertex_normals=np.asarray(o3d_mesh.vertex_normals), process=False
    )

def to_open3d(trimesh_mesh):
    """
    convert trimesh mesh to open3d mesh
    """
    trimesh_mesh = deepcopy(trimesh_mesh)  # if not copy, vertex normal cannot be assigned
    o3d_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
                                         triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces))
    # as_open3d method not working for color and normal
    if hasattr(trimesh_mesh.visual, 'vertex_colors'):
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(trimesh_mesh.visual.vertex_colors[:, :3] / 255.0)
    o3d_mesh.compute_vertex_normals()  # if not compute but only assign trimesh normals, the normal rendering fails, not sure about the reason
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(trimesh_mesh.vertex_normals)
    return o3d_mesh

def select_face_by_vertices(faces, vertices):
    return [face for face in faces
            if set(face).issubset(set(vertices))]

def get_obb_extents(mesh):
    return trimesh.bounds.oriented_bounds(mesh)[1]

def bbox_intersect(aabb1, aabb2):
    min_bound = np.maximum(np.asarray(aabb1.min_bound), np.asarray(aabb2.min_bound))
    max_bound = np.minimum(np.asarray(aabb1.max_bound), np.asarray(aabb2.max_bound))
    return (min_bound[:2] < max_bound[:2]).any()

def get_intersection_2d(aabb1, aabb2):
    min_bound = np.maximum(aabb1[:2], aabb2[:2])
    max_bound = np.minimum(aabb1[2:], aabb2[2:])
    return np.concatenate([min_bound, max_bound])

class ObjectNode:
    """
    Class for each object instance, with serialize and deserialize functions.
    """
    def __init__(self, load=False, **kwargs):
        """
        Initializer from deserialized dict or key word arguments.
        """
        # load from serialization
        if load:
            self.__dict__.update(kwargs)
        else:
            self.id = kwargs.get('id')  #int, instance id
            self.scene = kwargs.get('scene')  #str, scene_name
            self.category = kwargs.get('category')
            self.category_name = kwargs.get('category_name') if 'category_name' in kwargs else category_dict.loc[self.category]['mpcat40']
            self.vis_color = kwargs.get('vis_color')
            self.mesh = kwargs.get('mesh')
            self.aabb = kwargs.get('aabb')
            self.pointcloud = kwargs.get('pointcloud')
            # self.trans = normalize_transformation(self.pointcloud)

            # calculate object node features if not given from serialization
            # pigraph segment features
            self.hc = None  # centroid height above ground
            self.hs = None  # oriented bounding box height extent
            self.dxy = None  # diagoanl length of xy plane of obb
            self.axy = None  # area of xy plane of obb
            self.dominant_normal = None  # dominant normal (min PCA axis)
            self.z = None  # dot product of dominant normal(min PCA axis) with upwards vector (0,0,1), pigraph seems to use the mean of per point normals as dominant normal in code
            self.features = None  # concatenate features above into a vector
            self.calc_features()
        vis_mesh = o3d.geometry.TriangleMesh(self.mesh)
        vis_mesh.paint_uniform_color(self.vis_color)
        vis_mesh.vertex_normals = self.mesh.vertex_normals
        self.vis_mesh = vis_mesh
        self.aabb.color = self.vis_color

    """
    Calculate object feature, only used by PiGraph 
    """
    def calc_features(self):
        centroid = np.asarray(self.pointcloud.points).mean(axis=0)
        obb = self.pointcloud.get_oriented_bounding_box()
        covariance = self.pointcloud.compute_mean_and_covariance()[1]
        eigen_values, eigen_vectors = np.linalg.eig(covariance)
        # pigraph segment features
        self.hc = centroid[2]   # centroid height above ground
        self.hs = obb.extent[2]  # oriented bounding box height extent
        self.dxy = np.sqrt(obb.extent[0] * obb.extent[0] + obb.extent[1] * obb.extent[1])  # diagoanl length of xy plane of obb
        self.axy = obb.extent[0] * obb.extent[1]  # area of xy plane of obb
        self.dominant_normal = eigen_vectors[-1]  # dominant normal (min PCA axis)
        self.z = eigen_vectors[-1][2] # dot product of dominant normal(min PCA axis) with upwards vector (0,0,1), pigraph seems to use the mean of per point normals as dominant normal in code
        self.features = [self.hc, self.hs, self.dxy, self.axy, self.z]

    def serialize(self):
        serialized = copy.deepcopy(self.__dict__)
        serialized['mesh'] = {'vertices': np.asarray(self.mesh.vertices),
                              'triangles': np.asarray(self.mesh.triangles),
                              'vertex_colors': np.asarray(self.mesh.vertex_colors),
                              'vertex_normals': np.asarray(self.mesh.vertex_normals)}
        del serialized['aabb']
        del serialized['pointcloud']
        del serialized['vis_mesh']
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        vertex_colors = o3d.utility.Vector3dVector(serialized['mesh']['vertex_colors'])
        vertex_normals = o3d.utility.Vector3dVector(serialized['mesh']['vertex_normals'])
        serialized['mesh'] = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(serialized['mesh']['vertices']),
            triangles=o3d.utility.Vector3iVector(serialized['mesh']['triangles'])
        )
        serialized['mesh'].vertex_colors = vertex_colors
        serialized['mesh'].compute_vertex_normals()
        serialized['mesh'].vertex_normals = vertex_normals
        serialized['aabb'] = serialized['mesh'].get_axis_aligned_bounding_box()
        serialized['pointcloud'] = o3d.geometry.PointCloud(serialized['mesh'].vertices)
        return cls(load=True, **serialized)

class Scene:
    """
    Class for each object instance, with serialize and deserialize functions to accelerate loading.
    """
    def __init__(self, scene_name, rebuild=False):
        """
        Load the scene specified by name. The scene contains a mesh and a list of object instances.
        The object instances are loaded from cache for acceleration if exist or built from segmentation labels.
        """
        scene_cache_path = os.path.join(scene_cache_folder, scene_name + '.pkl')
        # load from scene cache file if exist for acceleration
        if os.path.exists(scene_cache_path):
            print('loading from cache for:', scene_name)
            with open(scene_cache_path, 'rb') as f:
                serialized = pickle.load(f)
            vertex_colors = o3d.utility.Vector3dVector(serialized['mesh']['vertex_colors'])
            vertex_normals = o3d.utility.Vector3dVector(serialized['mesh']['vertex_normals'])
            serialized['mesh'] = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(serialized['mesh']['vertices']),
                triangles=o3d.utility.Vector3iVector(serialized['mesh']['triangles'])
            )
            serialized['mesh'].vertex_colors = vertex_colors
            serialized['mesh'].compute_vertex_normals()
            serialized['mesh'].vertex_normals = vertex_normals
            serialized['object_nodes'] = [ObjectNode.deserialize(obj_node) for obj_node in serialized['object_nodes']]
            self.__dict__.update(serialized)
        else:
            print('create scene:', scene_name)
            self.object_nodes = []
            self.name = scene_name
            # cam to world transform
            cam2world_path = os.path.join(cam2world_folder, scene_name + ".json")
            with open(cam2world_path, 'r') as f:
                self.cam2world = np.array(json.load(f))
            # scene mesh and semantic mesh
            scene_path = os.path.join(scene_folder, scene_name + '.ply')
            # semantic annotation of Werkraum is inconsistent with RGB scan
            scene_semantic_path = os.path.join(scene_folder, scene_name + '_withlabels.ply')
            # load with trimesh errors for werkraum
            self.mesh = o3d.io.read_triangle_mesh(scene_path)
            original_mesh = to_trimesh(self.mesh)
            semantic_mesh = to_trimesh(o3d.io.read_triangle_mesh(scene_semantic_path))
            segment_mesh = semantic_mesh if scene_name in ['Werkraum',
                                                           'MPH1Library'] else original_mesh  # semantic annotation of the two is inconsistent with RGB scan, we can only split the semantic mesh

            # load or build instance segmentation
            instance_segment_path = os.path.join(scene_folder, self.name + '_segment.json')
            with open(instance_segment_path, 'r') as f:
                segment_labels = json.load(f)
            vertex_category_ids, vertex_instance_ids = np.array(segment_labels['vertex_category']), np.array(
                segment_labels['vertex_instance'])
            self.get_object_nodes(vertex_category_ids, vertex_instance_ids, segment_mesh)

            # save scene cache
            with open(scene_cache_path, 'wb') as f:
                pickle.dump(self.serialize(), f)

        self.mesh_with_accessory = {}
        # build kdtree, used by pigraph
        points = np.zeros((0, 3))
        self.vertex_number_list = []  # vertex number of each object, used to query which object a point belong to
        for obj_node in self.object_nodes:
            self.vertex_number_list.append(np.asarray(obj_node.mesh.vertices).shape[0])
            points = np.concatenate((points, np.asarray(obj_node.mesh.vertices)), axis=0)
        self.vertex_number_list = np.asarray(self.vertex_number_list)
        self.vertex_number_sum = np.cumsum(self.vertex_number_list)
        self.pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.kdtree = o3d.geometry.KDTreeFlann(self.pointcloud)  # kdtree of all object instances

        # load sdf
        # https: // github.com / mohamedhassanmus / POSA / blob / de21b40f22316cfb02ec43021dc5f325547c41ca / src / data_utils.py  # L99
        with open(os.path.join(sdf_folder, self.name + '.json'), 'r') as f:
            sdf_data = json.load(f)
            grid_dim = sdf_data['dim']
            grid_min = np.array(sdf_data['min']).astype(np.float32)
            grid_max = np.array(sdf_data['max']).astype(np.float32)
            voxel_size = (grid_max - grid_min) / grid_dim
        sdf = np.load(os.path.join(sdf_folder, self.name + '_sdf.npy')).astype(np.float32)
        sdf = sdf.reshape((grid_dim, grid_dim, grid_dim, 1))
        self.sdf = sdf
        self.sdf_config = {'grid_min': grid_min, 'grid_max': grid_max, 'grid_dim': grid_dim}

    def vertex_id_to_obj_id(self, vertex_id):
        """
        Query the object ID using vertex ID.
        """
        return np.searchsorted(self.vertex_number_sum, vertex_id)

    def serialize(self):
        serialized = copy.deepcopy(self.__dict__)
        serialized['mesh'] = {'vertices': np.asarray(self.mesh.vertices),
                              'triangles': np.asarray(self.mesh.triangles),
                              'vertex_colors': np.asarray(self.mesh.vertex_colors),
                              'vertex_normals': np.asarray(self.mesh.vertex_normals)}
        serialized['object_nodes'] = [obj_node.serialize() for obj_node in serialized['object_nodes']]
        # del serialized['kdtree']
        return serialized

    def get_object_nodes(self, vertex_category_ids, vertex_instance_ids, segment_mesh):
        """
        Build the list of object instances from segmentation labels.
        """
        # build object nodes
        self.object_nodes = []
        for instance_id in np.unique(vertex_instance_ids):
            if instance_id == 0:
                continue
            vertex_ids = np.nonzero(vertex_instance_ids == instance_id)[0]
            category_id = vertex_category_ids[vertex_ids[0]]
            vis_color = np.array(category_dict.loc[category_id]['color']) / 255
            face_ids = np.nonzero((vertex_instance_ids[segment_mesh.faces] == instance_id).sum(axis=1) == 3)[0]
            instance_mesh = segment_mesh.submesh([face_ids], append=True)
            instance_mesh = to_open3d(instance_mesh)
            if self.name in ['Werkraum', 'MPH1Library']:  # use vis color for these two
                instance_mesh.paint_uniform_color(vis_color)
            aabb = instance_mesh.get_axis_aligned_bounding_box()
            aabb.color = vis_color
            self.object_nodes.append(ObjectNode(mesh=instance_mesh, aabb=aabb, scene=self.name, vis_color=vis_color,
                                                pointcloud=o3d.geometry.PointCloud(instance_mesh.vertices),
                                                category=category_id, category_name=category_dict.loc[category_id]['mpcat40'],
                                                id=instance_id - 1))  # id -1 to make instance index start from 0, this makes object nodes array indexing more simple

    def get_visualize_geometries(self, semantic=False):
        """
        Get the open3d geometries of the scene for visualization.

        Input:
            semantic: bool, configures to visualize using original mesh color or MPCAT40 object visualization color table.
        """
        geometries = []
        # visualize scene
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0])
        geometries.append(mesh_frame)
        if self.name in ['Werkraum', 'MPH1Library'] and not semantic:
            geometries.append(self.mesh)
        else:
            for object_node in self.object_nodes:
                geometries.append(object_node.vis_mesh if semantic else object_node.mesh)
                geometries.append(object_node.aabb)
        return geometries

    def visualize(self, trans=None, semantic=False):
        # render
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().line_width = 20.0

        geometries = self.get_visualize_geometries(semantic=semantic)
        for geometry in geometries:
            vis.add_geometry(geometry)

        vis.poll_events()
        vis.update_renderer()
        vis.run()

    def save(self, semantic=True):
        """
        Render the scene and save to file.
        """
        # render
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, top=0, left=0, visible=False)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().line_width = 50

        geometries = self.get_visualize_geometries(semantic=semantic)
        for geometry in geometries:
            vis.add_geometry(geometry)

        if self.cam2world is not None:
            ctr = vis.get_view_control()
            cam_param = ctr.convert_to_pinhole_camera_parameters()
            cam_param.extrinsic = np.linalg.inv(self.cam2world)
            ctr.convert_from_pinhole_camera_parameters(cam_param)

        vis.poll_events()
        vis.update_renderer()
        vis_type = 'semantic' if semantic else 'raw'
        filename = os.path.join(scene_cache_folder, self.name + '_' + vis_type + '.png')
        vis.capture_screen_image(filename, True)
        vis.destroy_window()

    def log(self):
        """
        Log the scene instance segmentation.
        """
        file_path = os.path.join(scene_cache_folder, self.name + '.txt')
        with open(file_path, 'w') as f:
            idx = 0
            for object_node in self.object_nodes:
                print(("{idx}: {obj} of {num} vertices\n".format(idx=idx, obj=object_node.category_name + '_' + str(object_node.id),
                                                      num=np.asarray(object_node.mesh.vertices).shape[0]
                                                      )))
                f.write("{idx}: {obj} of {num} vertices\n".format(idx=idx, obj=object_node.category_name + '_' + str(object_node.id),
                                                      num=np.asarray(object_node.mesh.vertices).shape[0]
                                                      ))
                idx += 1

    def get_mesh_with_accessory(self, node_idx):
        """
        return complete meshes for single instances of sofa, bed, table by adding back accessory objects like cushion and objects
        """
        if node_idx in self.mesh_with_accessory:
            return self.mesh_with_accessory[node_idx]

        obj = self.object_nodes[node_idx]
        mesh = to_trimesh(obj.mesh)
        if obj.category_name in ['sofa', 'bed', 'table', 'cabinet', 'chest_of_drawers']:
            accessory_list = ['objects', 'object']
            if obj.category_name == 'table':
                accessory_list += ['tv_monitor']
            if obj.category_name in ['sofa', 'bed', 'couch']:
                accessory_list += ['cushion', 'pillow', 'pillows']
            accessory_candidates = [obj for obj in self.object_nodes if obj.category_name in accessory_list]
            if len(accessory_candidates):
                proximity = KDTree(mesh.vertices)
                for candidate in accessory_candidates:
                    dists, _ = proximity.query(np.asarray(candidate.mesh.vertices))
                    if dists.min() < 0.1:
                        mesh += to_trimesh(candidate.mesh)
        self.mesh_with_accessory[node_idx] = mesh
        return mesh

    def support_interaction(self, query_interaction):
        """
        Check whether the scene contains the objects in query interaction.
        """
        atomic_interactions = query_interaction.split('+')
        nouns = [atomic.split('-')[1] for atomic in atomic_interactions]
        scene_objs = [node.category_name for node in self.object_nodes]
        return set(nouns).issubset(set(scene_objs))

    def get_interaction_candidate_objects(self, interaction, use_annotation=True):
        """
        Get the list of candidate object combinations in this scene for sepcified interaction.

        Input:
            interaction: str, the query interaction
            use_annotation: bool, configures to use manual candidates annotation if available
        """
        atomic_interactions = interaction.split('+')
        verbs = [atomic.split('-')[0] for atomic in atomic_interactions]
        nouns = [atomic.split('-')[1] for atomic in atomic_interactions]
        if use_annotation and self.name in candidate_combination_dict and '+'.join(nouns) in candidate_combination_dict[self.name]:
            # print('use annotation')
            candidate_combination = candidate_combination_dict[self.name]['+'.join(nouns)]
            candidate_combination = [[self.object_nodes[obj_idx] for obj_idx in obj_combination] for obj_combination in candidate_combination]
            return verbs, nouns, candidate_combination
        else:
            # print('automic filtering', interaction)
            candidate_combination = None
            for noun in nouns:
                candidate_instances = [node for node in self.object_nodes if node.category_name == noun]
                if candidate_combination is None:
                    candidate_combination = [[instance] for instance in candidate_instances]
                else:
                    updated_combination = []
                    for combination in candidate_combination:
                        for instance in candidate_instances:
                            updated_combination.append(combination + [instance])
                    candidate_combination = updated_combination
        return verbs, nouns, candidate_combination

    def translation_sample_for_interaction(self, interaction, object_combination,
                                           sample_method='posa', num_samples=512, grid_step=0.2):
        atomic_interactions = interaction.split('+')
        verbs = [atomic.split('-')[0] for atomic in atomic_interactions]
        nouns = [atomic.split('-')[1] for atomic in atomic_interactions]
        candidate_aabb = None
        for atomic_idx, object_node in enumerate(object_combination):
            offset = 0.5 if 'touch' == verbs[atomic_idx] else 0
            aabb = np.array(
                    [object_node.aabb.min_bound[0] - offset,
                    object_node.aabb.min_bound[1] - offset,
                    object_node.aabb.max_bound[0] + offset,
                    object_node.aabb.max_bound[1] + offset,]
            )
            if candidate_aabb is None:
                candidate_aabb = aabb
            else:
                intersection = get_intersection_2d(aabb, candidate_aabb)
                if (intersection[:2] < intersection[2:]).all():
                    candidate_aabb = intersection
                else:  # no intersection
                    candidate_aabb = np.zeros(4)

        floor_height = self.get_floor_height()
        z_min = floor_height + 0.7 if 'stand' in verbs else floor_height + 0.4
        z_max = floor_height + 2.5 if 'stand on-table' in interaction else floor_height + 1.2
        if sample_method == 'posa':
            translation_samples = prox_to_posa(self.name, sample_box(candidate_aabb, z_min=z_min, z_max=z_max))
            return translation_samples
        else:  # PiGraph
            soboleng = torch.quasirandom.SobolEngine(dimension=3)
            min_bound = np.array([candidate_aabb[0], candidate_aabb[1], z_min])
            max_bound = np.array([candidate_aabb[2], candidate_aabb[3], z_max])
            extent = max_bound - min_bound
            translation_samples = soboleng.draw(num_samples).numpy() * extent.reshape(
                (1, 3)) + min_bound.reshape((1, 3))
            return  translation_samples

    def get_floor_height(self):
        """
        Get the floor height in this scene.
        """
        if hasattr(self, 'floor_height'):
            return  self.floor_height
        floor_pointclouds = [np.asarray(obj_node.mesh.vertices) for obj_node in self.object_nodes if obj_node.category_name == 'floor']
        if len(floor_pointclouds) == 0:
            floor_height = self.mesh.get_min_bound()[2]
        else:
            max_idx = np.argmax(np.array(
                [pointcloud.shape[0] for pointcloud in floor_pointclouds]
            ))
            max_floor = floor_pointclouds[max_idx]
            floor_height = max_floor[:, 2].mean()  # mean of z coord of points of max sized floor
        self.floor_height = floor_height
        return floor_height

    def calc_sdf(self, vertices):
        """
        Calculate the scene SDF values of input vertices using precomputed SDF grid.
        """
        if not hasattr(self, 'sdf_torch'):
            self.sdf_torch = torch.from_numpy(self.sdf).squeeze().unsqueeze(0).unsqueeze(0) # 1x1xDxDxD
        sdf_grids = self.sdf_torch.to(vertices.device)
        sdf_config = self.sdf_config
        sdf_max = torch.tensor(sdf_config['grid_max']).reshape(1, 1, 3).to(vertices.device)
        sdf_min = torch.tensor(sdf_config['grid_min']).reshape(1, 1, 3).to(vertices.device)

        batch_size, num_vertices, _ = vertices.shape
        vertices = ((vertices - sdf_min)
                         / (sdf_max - sdf_min) * 2 - 1)
        sdf_values = F.grid_sample(sdf_grids,
                                       vertices[:, :, [2, 1, 0]].view(-1, num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                       padding_mode='border',
                                       align_corners=True
                                       )
        return sdf_values.reshape(batch_size, num_vertices)

scenes = {}
for scene_name in scene_names:
    scenes[scene_name] = Scene(scene_name=scene_name)
if __name__ == "__main__":
    for scene_name in scene_names:
        scene = scenes[scene_name]
        scene.log()
        scene.save(semantic=True)
        scene.save(semantic=False)
        print('floor height:', scene.get_floor_height())
        # make floor at z=0, and export
        scene_floor = to_trimesh(scene.mesh)
        scene_floor.vertices[:, 2] -= scene.get_floor_height()
        scene_floor.export(Path('/home/kaizhao/projects/gamma/data/scenes/PROX') / (scene.name + '_floor.ply'))




