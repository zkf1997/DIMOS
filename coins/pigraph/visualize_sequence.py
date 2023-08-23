import numpy as np
import open3d as o3d
import smplx
from tqdm import tqdm
import os
import pickle
import torch
import argparse
import time

from pigraph_config import *
from load_human import load_sequence
from data.scene import Scene, scenes

DEBUG = False

def visualize_sequence(recording_name, start_frame, end_frame):
    print(recording_name, start_frame, end_frame)
    smplx_output, body_model = load_sequence(recording_name, start_frame, end_frame)
    joints = smplx_output.joints.detach().cpu().numpy()
    full_poses = smplx_output.full_pose.detach().cpu().numpy()
    scene_name = recording_name.split('_')[0]
    scene = Scene.create(scene_name=scene_name)
    # render
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().line_width = 50
    geometries = scene.get_visualize_geometries()
    for geometry in geometries:
        vis.add_geometry(geometry)

    # for idx in tqdm(range(end_frame - start_frame + 1)):
    #     skeleton = Skeleton(positions=np.asarray(joints[idx][:NUM_JOINTS]),
    #                         relative_orientations=np.asarray(full_poses[idx][:NUM_JOINTS * 3]).reshape((-1, 3)),
    #                         transform=scene.cam2world)
    #     body = skeleton.get_visualize_geometries(use_smplx=True)
    #     for geometry in body:
    #         vis.add_geometry(geometry)
    #     ctr = vis.get_view_control()
    #     cam_param = ctr.convert_to_pinhole_camera_parameters()
    #     cam_param.extrinsic = np.linalg.inv(scene.cam2world)
    #     ctr.convert_from_pinhole_camera_parameters(cam_param)
    #     vis.poll_events()
    #     vis.update_renderer()
    #     for geometry in body:
    #         vis.remove_geometry(geometry)
    for idx in tqdm(range(end_frame - start_frame + 1)):
        body = o3d.geometry.TriangleMesh()
        body.vertices = o3d.utility.Vector3dVector(smplx_output.vertices.detach().cpu().numpy()[idx])
        body.paint_uniform_color([0.8, 0.8, 0.8])
        body.triangles = o3d.utility.Vector3iVector(body_model.faces)
        body.compute_vertex_normals()
        body.transform(scene.cam2world)

        vis.add_geometry(body)
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param.extrinsic = np.linalg.inv(scene.cam2world)
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()
        # time.sleep(1)
        vis.remove_geometry(body)

def visualize_results(pkl_path):
    pkl_dir, interaction_gender = os.path.split(pkl_path)
    interaction = interaction_gender.split(':')[0]
    gender = interaction_gender.split(':')[1].split('.')[0]
    scene_name = os.path.split(pkl_dir)[1]
    scene = scenes[scene_name]
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().line_width = 50
    scene_meshes = scene.get_visualize_geometries()
    for mesh in scene_meshes:
        vis.add_geometry(mesh)

    with open(pkl_path, 'rb') as pkl_file:
        seq_data = pickle.load(pkl_file)
    body_model = smplx.create(smplx_model_folder, model_type='smplx',
                              gender=gender, ext='npz',
                              use_pca=False).to(torch.device('cuda'))
    smplx_output = body_model(**seq_data)
    vertices = smplx_output.vertices.detach().cpu().numpy()
    for idx in range(len(seq_data['transl'])):
        body = o3d.geometry.TriangleMesh()
        body.vertices = o3d.utility.Vector3dVector(vertices[idx])
        body.paint_uniform_color([0.8, 0.8, 0.8])
        body.triangles = o3d.utility.Vector3iVector(body_model.faces)
        body.compute_vertex_normals()

        vis.add_geometry(body)
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param.extrinsic = np.linalg.inv(scene.cam2world)
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
        vis.remove_geometry(body)

    for mesh in scene_meshes:
        vis.remove_geometry(mesh)
    vis.destroy_window()

if __name__ == "__main__":
    # visualize_sequence('N0Sofa_00034_01', 545, 565)
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    visualize_results(args.file)
