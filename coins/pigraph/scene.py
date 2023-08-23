import copy
import json
import pickle

import numpy as np
import open3d as o3d

from pigraph_config import *

DEBUG = False

class ObjectNode:
    def __init__(self, load=False, **kwargs):
        # load from serialization
        if load:
            self.__dict__.update(kwargs)
        else:
            self.id = kwargs.get('id')
            self.scene = kwargs.get('scene')
            self.category = kwargs.get('category')
            self.category_name = category_dict.loc[self.category]['mpcat40']
            self.mesh = kwargs.get('mesh')
            self.aabb = kwargs.get('aabb')
            self.pointcloud = kwargs.get('pointcloud')

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
        if DEBUG:
            print(self.features)

    def serialize(self):
        serialized = copy.deepcopy(self.__dict__)
        serialized['mesh'] = {'vertices': np.asarray(self.mesh.vertices),
                              'triangles': np.asarray(self.mesh.triangles),
                              'vertex_colors': np.asarray(self.mesh.vertex_colors)}
        del serialized['aabb']
        del serialized['pointcloud']
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        vertex_colors = o3d.utility.Vector3dVector(serialized['mesh']['vertex_colors'])
        serialized['mesh'] = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(serialized['mesh']['vertices']),
            triangles=o3d.utility.Vector3iVector(serialized['mesh']['triangles'])
        )
        serialized['mesh'].vertex_colors = vertex_colors
        serialized['aabb'] = serialized['mesh'].get_axis_aligned_bounding_box()
        serialized['pointcloud'] = o3d.geometry.PointCloud(serialized['mesh'].vertices)
        return cls(load=True, **serialized)

class Scene:
    def __init__(self, load=False, **kwargs):
        # load from serialization
        if load:
            self.__dict__.update(kwargs)
        else:
            self.object_nodes = []
            self.name = kwargs.get('scene_name')
            self.mesh = None  # scene representation as open3d mesh
            self.semantic_label = None  # per vertex semantic label of matterport40
            self.cam2world = None  # camera to world transform for the scene, the sequences are captured by a fixed camera
            self.build_scene(self.name)

        # build kdtree
        points = np.zeros((0, 3))
        self.vertex_number_list = [] # vertex number of each object, used to query which object a point belong to
        for obj_node in self.object_nodes:
            self.vertex_number_list.append(np.asarray(obj_node.mesh.vertices).shape[0])
            points = np.concatenate((points, np.asarray(obj_node.mesh.vertices)), axis=0)
        self.vertex_number_list = np.asarray(self.vertex_number_list)
        self.vertex_number_sum = np.cumsum(self.vertex_number_list)
        self.pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.kdtree = o3d.geometry.KDTreeFlann(self.pointcloud) # kdtree of all object instances

        #  build sdf
        # self.sdf = trimesh.proximity.ProximityQuery(trimesh.Trimesh(
        #         vertices=np.asarray(self.mesh.vertices), faces=np.asarray(self.mesh.triangles)))
        # load sdf
        # https: // github.com / mohamedhassanmus / POSA / blob / de21b40f22316cfb02ec43021dc5f325547c41ca / src / data_utils.py  # L99
        with open(os.path.join(sdf_folder, self.name + '.json'), 'r') as f:
            sdf_data = json.load(f)
            grid_dim = sdf_data['dim']
            grid_min = np.array(sdf_data['min'])
            grid_max = np.array(sdf_data['max'])
            voxel_size = (grid_max - grid_min) / grid_dim
        sdf = np.load(os.path.join(sdf_folder, self.name + '_sdf.npy')).astype(np.float32)
        sdf = sdf.reshape((grid_dim, grid_dim, grid_dim, 1))
        self.sdf = sdf
        self.sdf_config = {'grid_min': grid_min, 'grid_max': grid_max, 'grid_dim': grid_dim}

    def build_scene(self, scene_name, overwrite=False):
        # no cached scene, compute scene node and save
        scene_path = os.path.join(scene_folder, scene_name + '.ply')
        scene_semantic_path = os.path.join(scene_folder, scene_name + '_withlabels.ply') if scene_name != 'N0Sofa' else os.path.join(scene_folder, scene_name + '_withlabels_old.ply')

        scene_original = o3d.io.read_triangle_mesh(scene_path)
        scene = o3d.io.read_triangle_mesh(scene_semantic_path)
        self.mesh = scene_original
        self.semantic_label = (np.asarray(scene.vertex_colors)[:, 0] * 255 / 5).astype(int)
        cam2world_path = os.path.join(cam2world_folder, scene_name + ".json")
        with open(cam2world_path, 'r') as f:
            self.cam2world = np.array(json.load(f))

        # split scene to objects
        # stores vertex indices belong to each category
        objects_idx = {}
        for idx, vertex_color in enumerate(scene.vertex_colors):
            category = int(vertex_color[0] * 255 / 5)
            if not category in objects_idx:
                objects_idx[category] = []
            objects_idx[category].append(idx)

        # 42 categories in total, exclude wall-1 and floor-2?
        instance_id = 0
        for category in range(1, 42):
            if category in objects_idx:
                if DEBUG:
                    print(category_dict.loc[category]['mpcat40'])
                if (category == 17):  # also exclude ceiling
                    # print(objects_idx[category])
                    continue

                object = scene.select_by_index(objects_idx[category])
                vis_color = np.array(category_dict.loc[category]['color']) / 255

                # instance segmentation by connected components
                (cluster_idxs, num_faces, areas) = object.cluster_connected_triangles()
                num_instances = np.asarray(num_faces).shape[0]
                instances_idx = {}
                for idx in range(num_instances):
                    instances_idx[idx] = set()
                for idx, cluster_idx in enumerate(cluster_idxs):
                    vertices = object.triangles[idx]
                    for vertex in vertices:
                        instances_idx[cluster_idx].add(vertex)
                for idx in range(num_instances):
                    instance = object.select_by_index(list(instances_idx[idx]))
                    if np.asarray(instance.vertices).shape[0] < 100 or \
                            (np.asarray(instance.vertices).shape[0] < 2000 and category <= 2) or \
                            (np.asarray(instance.vertices).shape[0] < 500 and category != 39):  # TODO maybe we need a category specific threshold, for objects, this should be smaller
                        if DEBUG:
                            print("discard an isntance of ", category_dict.loc[category]['mpcat40'], " with ", np.asarray(instance.vertices).shape[0], "points")
                        continue
                    instance.paint_uniform_color(vis_color)
                    aabb = instance.get_axis_aligned_bounding_box()
                    aabb.color = vis_color
                    # obb = instance.get_oriented_bounding_box()
                    # obb.color = vis_color
                    self.object_nodes.append(ObjectNode(mesh=instance, aabb=aabb, scene=self.name,
                                                        pointcloud=o3d.geometry.PointCloud(instance.vertices), category=category, id=instance_id))
                    # if DEBUG:
                    #     print('object ', instance_id, ':', self.object_nodes[-1].category_name, ', num vertices:',
                    #           np.asarray(instance.vertices).shape[0])
                    instance_id += 1

        if DEBUG:
            print('number of instances:', instance_id)
        # save to cache
        scene_cache_path = os.path.join(scene_cache_folder, scene_name + '.pkl')
        with open(scene_cache_path, 'wb') as f:
            pickle.dump(self.serialize(), f)

    def vertex_id_to_obj_id(self, vertex_id):
        return np.searchsorted(self.vertex_number_sum, vertex_id)

    def serialize(self):
        serialized = copy.deepcopy(self.__dict__)
        serialized['mesh'] = {'vertices': np.asarray(self.mesh.vertices),
                              'triangles': np.asarray(self.mesh.triangles),
                              'vertex_colors': np.asarray(self.mesh.vertex_colors)}
        serialized['object_nodes'] = [obj_node.serialize() for obj_node in serialized['object_nodes']]
        # del serialized['kdtree']
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        # print(serialized['mesh']['vertices'].shape)
        vertex_colors = o3d.utility.Vector3dVector(serialized['mesh']['vertex_colors'])
        serialized['mesh'] = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(serialized['mesh']['vertices']),
            triangles=o3d.utility.Vector3iVector(serialized['mesh']['triangles'])
        )
        serialized['mesh'].vertex_colors = vertex_colors

        serialized['object_nodes'] = [ObjectNode.deserialize(obj_node) for obj_node in serialized['object_nodes']]
        # points = np.zeros((0, 3))
        # for obj_node in serialized['object_nodes']:
        #     points = np.concatenate((points, np.asarray(obj_node.mesh.vertices)), axis=0)
        # serialized['kdtree'] = o3d.geometry.KDTreeFlann(points)
        return cls(load=True, **serialized)

    @classmethod
    # create from cache if available or compute using scene scan
    def create(cls, scene_name, overwrite=False):
        # check if there are precomputed scene nodes
        scene_cache_path = os.path.join(scene_cache_folder, scene_name + '.pkl')
        if os.path.exists(scene_cache_path) and not overwrite:
            with open(scene_cache_path, 'rb') as f:
                try:
                    pkl = pickle.load(f)
                    print("create with cache for:", scene_name)
                    return cls.deserialize(pkl)
                except:
                    print("load failed, try to build scene nodes from scene scan")
                    return cls(scene_name=scene_name)
        print("build scene nodes from scan for scene:", scene_name)
        return cls(scene_name=scene_name)

    def get_visualize_geometries(self, semantic=False):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0])
        geometries = [mesh_frame]
        # geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.6, origin=[1, 1, 1]))
        # visualize scene
        if semantic:
            for object_node in self.object_nodes:
                # if object_node.category_name != 'table':
                #     continue
                geometries.append(object_node.mesh)
                geometries.append(object_node.aabb)
        else:
            geometries.append(self.mesh)
        return geometries

    def visualize(self, trans=None, semantic=False):
        # render
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().line_width = 50

        geometries = self.get_visualize_geometries(semantic=semantic)
        for geometry in geometries:
            vis.add_geometry(geometry)

        vis.poll_events()
        vis.update_renderer()
        vis.run()

    def save(self, semantic=True):
        # render
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, top=0, left=0, visible=False)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().line_width = 50

        geometries = self.get_visualize_geometries(semantic=semantic)
        for geometry in geometries:
            vis.add_geometry(geometry)

        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param.extrinsic = np.linalg.inv(self.cam2world)
        ctr.convert_from_pinhole_camera_parameters(cam_param)

        vis.poll_events()
        vis.update_renderer()
        filename = os.path.join(scene_cache_folder, self.name + '.png')
        vis.capture_screen_image(filename, True)
        vis.destroy_window()

    def log(self):
        file_path = os.path.join(scene_cache_folder, self.name + '.txt')
        with open(file_path, 'w') as f:
            idx = 0
            for object_node in self.object_nodes:
                f.write("{idx}: {obj} of {num} vertices\n".format(idx=idx, obj=object_node.category_name + '_' + str(object_node.id),
                                                      num=np.asarray(object_node.mesh.vertices).shape[0]
                                                      ))
                idx += 1

if __name__ == "__main__":
    DEBUG = True
    for scene_name in scene_names:
        # if scene_name != "N0Sofa":
        #     continue
        scene = Scene.create(scene_name=scene_name, overwrite=True)
        scene.log()
        scene.save(semantic=True)
        # scene.visualize(semantic=False)

