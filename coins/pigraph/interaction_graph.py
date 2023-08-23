import json
import pickle

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from pigraph_config import *
from data.scene import Scene
from skeleton import NUM_JOINTS, Skeleton

DEBUG = False
trans = None
THRESHOLD = 0.1  # threshold for contact distance

class HumanObjectEdge:
    def __init__(self, human, object, id, features, relation='contact'):
        self.human_node_id = human
        self.object_node_id = object
        self.id = id  # edge id
        self.type = relation  # realtion type
        self.features = features  # edge feature vector

        if DEBUG:
            print(human, object, id)
            print(features)

class InteractionGraph:
    def __init__(self, scene, skeleton):
        self.scene = scene
        self.skeleton = skeleton
        self.human_object_edges = []

        self.build_graph()

    def build_graph(self):
        kdtree = self.scene.kdtree
        vertex_number_list = self.scene.vertex_number_list
        contact_edge_cnt = 0
        for idx, position in enumerate(self.skeleton.positions):
            nn_num, nn_indices, nn_dists = kdtree.search_hybrid_vector_3d(query=position, radius=THRESHOLD, max_nn=1)
            if nn_num > 0:
                nn_index = nn_indices[0]
                obj = self.scene.vertex_id_to_obj_id(nn_index)
                contact_point = self.scene.pointcloud.points[nn_index]
                if DEBUG:
                    print(contact_point)
                center_of_mass = self.skeleton.center_of_mass
                contact_vector = contact_point - center_of_mass
                rot1 = Rotation.from_rotvec(self.skeleton.relative_orientations[0]).inv()
                contact_vector_local = rot1.apply(contact_vector)
                contact_vector_normed = contact_vector / np.sqrt(np.sum(contact_vector ** 2))
                contact_vector_projection = contact_vector[:2] / np.sqrt(
                    np.sum(contact_vector[:2] ** 2))  # project to horizontal plane
                contact_vector_local_projection = contact_vector_local[:2] / np.sqrt(
                    np.sum(contact_vector_local[:2] ** 2))  # project to horizontal plane
                h = contact_point[2]  # absolute height of contact point
                r = np.sqrt(((
                                         contact_point - center_of_mass) ** 2).sum())  # radial distance from skeletal center of mass to contact point
                z = contact_point[2] - center_of_mass[2]  # vertical displacement from center of mass to contact point
                theta_xy = np.arctan2(contact_vector_local_projection[1],
                                      contact_vector_local_projection[0])  # angle of vector from center of mass to contact point in xy plane
                dot_contact = np.dot(self.scene.object_nodes[obj].dominant_normal,
                                     contact_vector_normed)  # contact segment’s dominant normal vector z dot product with direction of contact
                features = [h, r, z, theta_xy, dot_contact]
                self.human_object_edges.append(HumanObjectEdge(human=idx, object=obj,
                                                               id=contact_edge_cnt, features=features))
                if DEBUG:
                    print(self.scene.object_nodes[obj].category_name)
                contact_edge_cnt += 1

    # def build_graph(self):
    #     # find contact edges by nearest neighbor, each joint has at most one contact edge
    #     object_pointclouds = [object.pointcloud for object in self.scene.object_nodes]
    #     if DEBUG:
    #         print("constructing object sdfs")
    #     object_sdfs = [trimesh.proximity.ProximityQuery(trimesh.Trimesh(
    #         vertices=np.asarray(object.mesh.vertices), faces=np.asarray(object.mesh.triangles))) for object in self.scene.object_nodes]
    #     if DEBUG:
    #         print("construct object sdfs finish")
    #     joints = self.skeleton.positions[:NUM_JOINTS]
    #     joints_pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(joints))
    #     num_obj = len(object_pointclouds)
    #     num_joints = NUM_JOINTS
    #     # calculate distance matrix
    #     dists = np.zeros((num_joints, num_obj))  # distance from each joint to each object instance
    #     # for idx, object_sdf in enumerate(object_pointclouds):
    #     # dists[:, idx] = np.asarray(joints_pointcloud.compute_point_cloud_distance(object))
    #     if DEBUG:
    #         print("calculating dist matrix")
    #     for idx, object_sdf in enumerate(object_sdfs):
    #         dists[:, idx] = -np.asarray(object_sdf.signed_distance(joints))  #trimesh return negative value for points outside surface
    #     if DEBUG:
    #         print("dists:", dists)
    #     nearest_neighbors = dists.argmin(axis=1)
    #     if DEBUG:
    #         print(nearest_neighbors)
    #
    #     contact_edge_cnt = 0
    #     for idx, obj in enumerate(nearest_neighbors):
    #         if dists[idx][obj] < THRESHOLD:  # only if distance to nearest object is within a thresold
    #             features = []  # contact edge features
    #             # contact_point_idx = ((np.asarray(object_pointclouds[obj].points) - self.skeleton.positions[idx])**2)\
    #             #     .sum(axis=1).argmin()  # find contact point which is closest to joint
    #             # contact_point = np.asarray(object_pointclouds[obj].points)[contact_point_idx]
    #             obj_trimesh = trimesh.Trimesh(vertices=np.asarray(self.scene.object_nodes[obj].mesh.vertices),
    #                                           faces=np.asarray(self.scene.object_nodes[obj].mesh.triangles))
    #             (closest_points,
    #              distances,
    #              triangle_id) = obj_trimesh.nearest.on_surface(self.skeleton.positions[idx:idx+1])
    #             contact_point = closest_points[0]
    #             if DEBUG:
    #                 print(contact_point)
    #             center_of_mass = self.skeleton.center_of_mass
    #             contact_vector = contact_point - center_of_mass
    #             contact_vector_normed = contact_vector / np.sqrt(np.sum(contact_vector**2))
    #             contact_vector_projection = contact_vector[:2] / np.sqrt(np.sum(contact_vector[:2]**2))  # project to horizontal plane
    #             h = contact_point[2]  # absolute height of contact point
    #             r = np.sqrt(((contact_point - center_of_mass)**2).sum())  # radial distance from skeletal center of mass to contact point
    #             z = contact_point[2] - center_of_mass[2]  # vertical displacement from center of mass to contact point
    #             theta_xy = np.arctan2(contact_vector_projection[1], contact_vector_projection[0])  #angle of vector from center of mass to contact point in xy plane
    #             dot_contact = np.dot(self.scene.object_nodes[obj].dominant_normal, contact_vector_normed)  # contact segment’s dominant normal vector z dot product with direction of contact
    #             features = [h, r, z, theta_xy, dot_contact]
    #             self.human_object_edges.append(HumanObjectEdge(human=idx, object=obj,
    #                                                            id=contact_edge_cnt, features=features))
    #             if DEBUG:
    #                 print(self.scene.object_nodes[obj].category_name)
    #             contact_edge_cnt += 1

    # calculate the penetration by fit skeleton to smpl and query per vertex sdf value
    def penetration(self):
        # t1 = time.time()
        body_mesh = self.skeleton.to_smplx_mesh()
        # print(time.time() - t1)
        vertices = np.asarray(body_mesh.vertices)
        sdf_config = self.scene.sdf_config
        device = torch.device("cuda")
        sdf_grids = torch.tensor(self.scene.sdf, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 4, 1, 2, 3)  # B*C*D*D*D
        grid_dim, grid_min, grid_max = sdf_config['grid_dim'], sdf_config['grid_min'], sdf_config['grid_max']
        normed_vertices = torch.tensor((vertices - grid_min) / (grid_max - grid_min) * 2 - 1, dtype=torch.float32, device=device)
        x = F.grid_sample(sdf_grids,
                          normed_vertices[:, [2, 1, 0]].view(1, vertices.shape[0], 1, 1, 3),
                          padding_mode='border', mode='bilinear', align_corners=True)
        x = x.permute(0, 2, 3, 4, 1)  # B*V*1*1*C
        x = x[x < 0]
        penetration_score = (x**2).sum().cpu().numpy()  # penalize points inside scene meshes
        return penetration_score
        # print(self.scene.sdf)
        # print(np.asarray(body_mesh.vertices).shape)
        # dists = np.asarray(-self.scene.sdf.signed_distance(np.asarray(body_mesh.vertices[:100])))  #  trimesh sdf return negative dists for points outside surfaces, negate it
        # penetration_score = np.sum(dists[dists < 0]**2)  # penalize points inside scene meshes
        # return penetration_score

    def log(self):
        print("igraph logging:")
        for edge in self.human_object_edges:
            print("joint:", edge.human_node_id, " object: ", self.scene.object_nodes[edge.object_node_id].category_name)
            print("features[h, r, z, theta_xy, dot_contact]:", edge.features)
        print("penetration: ", self.penetration())

    def get_visualize_geometries(self, use_smplx=False, semantic=False):
        geometries = self.scene.get_visualize_geometries(semantic) + self.skeleton.get_visualize_geometries(use_smplx=use_smplx)
        lines = []
        positions = [self.skeleton.center_of_mass]
        for human_object_edge in self.human_object_edges:
            joint_id = human_object_edge.human_node_id
            object_node = self.scene.object_nodes[human_object_edge.object_node_id]
            positions.append(object_node.aabb.get_center())
            lines.append([0, len(positions) - 1])
        if len(lines) > 0 :
            colors = [[0, 0, 1]] * len(lines)
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(positions),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)

        return geometries

    def visualize(self, trans=None, use_smplx=False, semantic=False):
        # render
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().line_width = 50

        geometries = self.get_visualize_geometries(use_smplx=use_smplx, semantic=semantic)
        for geometry in geometries:
            vis.add_geometry(geometry)

        if trans is not None:
            ctr = vis.get_view_control()
            cam_param = ctr.convert_to_pinhole_camera_parameters()
            cam_param.extrinsic = np.linalg.inv(trans)
            ctr.convert_from_pinhole_camera_parameters(cam_param)

        vis.poll_events()
        vis.update_renderer()
        vis.run()

    def save(self, filename, use_smplx=False, vis=None, semantic=False):
        # render
        if vis == None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            try:
                vis.get_render_option().mesh_show_back_face = True
                vis.get_render_option().line_width = 50
            except:
                print("visualizer init failed")
        # print("clear geometries")
        vis.clear_geometries()
        # print("getting geometry")
        geometries = self.get_visualize_geometries(use_smplx=use_smplx, semantic=semantic)
        for geometry in geometries:
            vis.add_geometry(geometry)
        # print("geometry added")

        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param.extrinsic = np.linalg.inv(self.scene.cam2world)
        ctr.convert_from_pinhole_camera_parameters(cam_param)

        vis.poll_events()
        vis.update_renderer()
        # print("saving image")
        vis.capture_screen_image(filename, True)
        # print("image saved")
        # vis.destroy_window()

    def export(self, filepath):
        serialized = {'scene': self.scene.name, 'skeleton': self.skeleton}
        with open(filepath, 'wb') as f:
            pickle.dump(serialized, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            pkl = pickle.load(f)
        scene = Scene.create(scene_name=pkl['scene'])
        skeleton = pkl['skeleton']
        return cls(scene, skeleton)


if __name__ == "__main__":
    DEBUG = True
    print("load scene")
    scene = Scene.create("MPH16")
    print("scene loaded")

    from load_human import load_human
    human_path = "/local/home/proxe/PROX_temporal/PROXD_temp_v2/MPH16_00157_01/results.pkl"
    smplx_output, body_model = load_human(human_path)
    joints = smplx_output.joints.detach().cpu().numpy()
    full_poses = smplx_output.full_pose.detach().cpu().numpy()
    # cam2world tranform
    cam2world_path = r"/local/home/proxe/cam2world/MPH16.json"
    with open(cam2world_path, 'r') as f:
        trans = np.array(json.load(f))
    skeleton = Skeleton(positions=np.asarray(joints[1400][:NUM_JOINTS]), relative_orientations=np.asarray(full_poses[1400][:NUM_JOINTS * 3]).reshape((-1, 3)))
    skeleton.transform(trans)
    # skeleton.positions -= np.array([0, 0, 0.2])

    igraph = InteractionGraph(scene, skeleton)
    # t1 = time.time()
    # print(igraph.penetration())
    # print(time.time() - t1)
    igraph.log()
    igraph.visualize(trans, use_smplx=True, semantic=False)
    igraph.export(os.path.join(results_folder, 'igraph.pkl'))
    # igraph.save(os.path.join(rendering_folder, 'igraph.png'), use_smplx=True)

    loaded_igraph = InteractionGraph.load(os.path.join(results_folder, 'igraph.pkl'))
    loaded_igraph.visualize(trans, use_smplx=True, semantic=False)
