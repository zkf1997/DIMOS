import sys
sys.path.append('..')

import json
import pickle

import numpy as np
import open3d as o3d
import smplx
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from data.scene import scenes, to_trimesh
from configuration.config import *
import time

body_model = smplx.create(smplx_model_folder, model_type='smplx',
                              gender='neutral', ext='npz',
                              num_pca_comps=12,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=1,
                              ).to(torch.device('cuda'))
def get_smplx_vertices(seq_data):
    T = len(seq_data['transl'])
    torch_param = {}
    torch_param['betas'] = torch.tensor(seq_data['betas']).repeat(T, 1).to(torch.device('cuda'))
    torch_param['global_orient'] = torch.tensor(seq_data['global_orient']).to(torch.device('cuda'))
    torch_param['transl'] = torch.tensor(seq_data['transl']).to(torch.device('cuda'))
    torch_param['left_hand_pose'] = torch.tensor(seq_data['left_hand_pose']).to(torch.device('cuda'))
    torch_param['right_hand_pose'] = torch.tensor(seq_data['right_hand_pose']).to(torch.device('cuda'))
    torch_param['jaw_pose'] = torch.tensor(seq_data['jaw_pose']).to(torch.device('cuda'))
    torch_param['leye_pose'] = torch.tensor(seq_data['leye_pose']).to(torch.device('cuda'))
    torch_param['reye_pose'] = torch.tensor(seq_data['reye_pose']).to(torch.device('cuda'))
    torch_param['expression'] = torch.tensor(seq_data['expression']).to(torch.device('cuda'))
    torch_param['body_pose'] = torch.tensor(seq_data['body_pose']).to(torch.device('cuda'))
    smplx_output = body_model(return_verts=True, **torch_param)
    vertices = smplx_output.vertices.detach().cpu().numpy()  # [n_frames, 10475, 3]
    joints = smplx_output.joints.detach().cpu().numpy()

    return vertices, joints, body_model

def visualize(interaction_data, full_scene=True, skip_frame=1, start_frame=0):
    """
    Visualize interaction data with configurable options.

    Input:
        interaction_data: list of interaction frames
        full_scene: bool, choose to visualize the full scene or only interaction objects
        skip_frame: int, control the frame rate of visualization
        start_frame: int, specify the index of the starting frame for visualization
    """
    # renderer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().line_width = 50
    frame_scene = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    # vis.add_geometry(frame_scene)

    last_scene = None
    for record_idx in tqdm(range(start_frame, len(interaction_data), skip_frame)):
        record = interaction_data[record_idx]
        # scene_name, sequence, frame_idx, smplx_param, interaction_labels, interaction_obj_idx = record
        print(record['interaction_labels'], record['interaction_obj_idx'], record['sequence'], record['frame_idx'])
        scene = scenes[record['scene_name']]
        scene_mesh, trans = scene.mesh, scene.cam2world

        T = 1
        vertices, joints, body_model = get_smplx_vertices(record['smplx_param'])
        # add scene mesh
        # if record['scene_name'] != last_scene:
        #     last_scene = record['scene_name']
        #     vis.clear_geometries()
        if full_scene:
            vis.add_geometry(scene_mesh)
        if not full_scene:
            frame_objs = []
            for idx in record['interaction_obj_idx']:
                vis.add_geometry(scene.object_nodes[idx].mesh)
                # frame_objs.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
                #     size=0.6, origin=[0, 0, 0]).transform(scene.object_nodes[idx].trans))
                # vis.add_geometry(frame_objs[-1])

        body = o3d.geometry.TriangleMesh()
        body.vertices = o3d.utility.Vector3dVector(vertices[0])
        body.triangles = o3d.utility.Vector3iVector(body_model.faces)
        body.compute_vertex_normals()
        # body.transform(trans)  # camera to world coordinate

        vis.add_geometry(body)
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        # cam_param = update_cam(cam_param, trans)
        cam_param.extrinsic = np.linalg.inv(trans)
        ctr.convert_from_pinhole_camera_parameters(cam_param)

        vis.poll_events()
        vis.update_renderer()
        # vis.run()
        vis.remove_geometry(body)

        if full_scene:
            vis.remove_geometry(scene_mesh)
        else:
            for idx in record['interaction_obj_idx']:
                vis.remove_geometry(scene.object_nodes[idx].mesh)
            # for frame_obj in frame_objs:
            #     vis.remove_geometry(frame_obj)

    vis.remove_geometry(frame_scene)
    vis.destroy_window()

def have_interaction(interaction_labels, query_interaction, mode='verb-noun', exact_match=False):
    """
    Check if a given interaction labels matches query.

    Input:
        interaction_labels: list of interaction labels in the format of verb or verb-noun
        query_interaction: the queried interaction combination, list of interaction labels in the format of verb or verb-noun
        mode: str, specify using verb or verb-noun mode
        exact_match: bool, define matching as exactly the same or containing the queried interactions

    Output:
        result: whether the interaction_labels match query_interaction
    """
    if mode == 'verb':
        interaction_labels = [interaction.split('-')[0] for interaction in interaction_labels]

    result = (set(query_interaction) == set(interaction_labels)) if exact_match else set(query_interaction).issubset(set(interaction_labels))
    return result

def get_interaction_segments(query_interaction, interaction_data, mode='verb-noun', exact_match=False):
    """
    Filter interaction frames that match the queried interaction
    """
    results = [record for record in interaction_data if have_interaction(record['interaction_labels'], query_interaction, mode=mode, exact_match=exact_match)]

    return results

if __name__ == "__main__":
    with open(Path.joinpath(project_folder, "data", 'train.pkl'), 'rb') as data_file:
        train_data = pickle.load(data_file)
    with open(Path.joinpath(project_folder, "data", 'test.pkl'), 'rb') as data_file:
        test_data = pickle.load(data_file)
    # load interactions containing specified interactions and visualize
    # data = get_interaction_segments(['sit on-sofa', 'touch-table'], train_data, mode='verb-noun')
    # print(len(data))
    # visualize(data, full_scene=False, skip_frame=1, start_frame=0)
    data = get_interaction_segments(['lie on'], train_data, mode='verb')
    print(len(data))
    visualize(data, full_scene=False, skip_frame=1)
