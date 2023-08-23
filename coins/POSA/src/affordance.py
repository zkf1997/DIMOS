# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../human_body_prior/src')

import os.path as osp
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
import random
import trimesh
import glob
import yaml
import pickle
import torchgeometry as tgm
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
from src.optimizers import optim_factory
from src.cmd_parser import parse_config
from src import posa_utils, eulerangles, viz_utils, misc_utils, data_utils, opt_utils

from data.scene import scenes
from configuration.config import *
from data.scene_registration import POSA_to_PROX_transform

from scipy import stats
import time

if __name__ == '__main__':
    args, args_dict = parse_config()
    args_dict['batch_size'] = 1
    # args_dict['semantics_w'] = 0.01
    batch_size = args_dict['batch_size']
    if args.use_clothed_mesh:
        args_dict['opt_pose'] = False
    args_dict['base_dir'] = osp.expandvars(args_dict.get('base_dir'))
    args_dict['ds_us_dir'] = osp.expandvars(args_dict.get('ds_us_dir'))
    args_dict['affordance_dir'] = osp.expandvars(args_dict.get('affordance_dir'))
    args_dict['pkl_file_path'] = osp.expandvars(args_dict.get('pkl_file_path'))
    args_dict['model_folder'] = osp.expandvars(args_dict.get('model_folder'))
    args_dict['rp_base_dir'] = osp.expandvars(args_dict.get('rp_base_dir'))
    base_dir = args_dict.get('base_dir')
    ds_us_dir = args_dict.get('ds_us_dir')
    scene_name = args_dict.get('scene_name')

    # Create results folders
    affordance_dir = osp.join(args_dict.get('affordance_dir'))
    os.makedirs(affordance_dir, exist_ok=True)
    pkl_folder = osp.join(affordance_dir, 'pkl', args_dict.get('scene_name'))
    os.makedirs(pkl_folder, exist_ok=True)
    physical_metric_folder = osp.join(affordance_dir, 'physical_metric', args_dict.get('scene_name'))
    os.makedirs(physical_metric_folder, exist_ok=True)
    rendering_folder = osp.join(affordance_dir, 'renderings', args_dict.get('scene_name'))
    os.makedirs(rendering_folder, exist_ok=True)
    os.makedirs(osp.join(affordance_dir, 'meshes', args_dict.get('scene_name')), exist_ok=True)
    os.makedirs(osp.join(affordance_dir, 'meshes_clothed', args_dict.get('scene_name')), exist_ok=True)

    device = torch.device("cuda" if args_dict.get('use_cuda') else "cpu")
    dtype = torch.float32
    args_dict['device'] = device
    args_dict['dtype'] = dtype

    A_1, U_1, D_1 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 1, args_dict['use_cuda'])
    down_sample_fn = posa_utils.ds_us(D_1).to(device)
    up_sample_fn = posa_utils.ds_us(U_1).to(device)

    A_2, U_2, D_2 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 2, args_dict['use_cuda'])
    down_sample_fn2 = posa_utils.ds_us(D_2).to(device)
    up_sample_fn2 = posa_utils.ds_us(U_2).to(device)

    faces_arr = trimesh.load(osp.join(ds_us_dir, 'mesh_{}.obj'.format(0)), process=False).faces
    model = misc_utils.load_model_checkpoint(**args_dict).to(device)

    # load 3D scene
    scene = vis_o3d = None
    if args.viz or args.show_init_pos:
        scene = o3d.io.read_triangle_mesh(osp.join(base_dir, 'scenes', args_dict.get('scene_name') + '.ply'))
    scene_data = data_utils.load_scene_data(name=scene_name, sdf_dir=osp.join(base_dir, 'sdf'),
                                            **args_dict)
    pkl_file_path = args_dict.pop('pkl_file_path')
    if osp.isdir(pkl_file_path):
        pkl_file_dir = pkl_file_path
        pkl_file_paths = glob.glob(osp.join(pkl_file_dir, '*.pkl'))
        random.shuffle(pkl_file_paths)
    else:
        pkl_file_paths = [pkl_file_path]

    # scene specification
    specified_objs = []
    if args.obj_name == 'scene':
        specified_objs = ['scene']
    else:
        for obj in args.obj_name.split('+'):
            specified_objs.append(category_dict[category_dict['mpcat40'] == obj].index[0])
    specified_objs = set(specified_objs)
    print(specified_objs)
    atomic_interactions = args.interaction.split('+')
    instance_idx = args.object_combination.split('+')
    instance_nodes = [scenes[args.scene_name].object_nodes[int(idx)] for idx in instance_idx]
    atomic_names = [atomic_interactions[atomic_idx] + '-' + instance_idx[atomic_idx] for atomic_idx in range(len(atomic_interactions))]
    basename = '+'.join(atomic_names)
    result_path = Path.joinpath(results_folder, args.export_dir, args.interaction, args.scene_name,  basename + '.pkl')
    result_path.parent.mkdir(exist_ok=True, parents=True)
    print(result_path)
    synthesis_results = []

    # Loading VPoser Body Pose Prior
    # from human_body_prior.tools.model_loader import load_model
    # from human_body_prior.models.vposer_model import VPoser
    #
    # expr_dir = str(project_folder / 'human_body_prior' / 'V02_05')
    # vp, ps = load_model(expr_dir, model_code=VPoser,
    #                     remove_words_in_model_weights='vp_model.',
    #                     disable_grad=False)
    # vp = vp.to('cuda')

    for idx, pkl_file_path in enumerate(pkl_file_paths):
        # if platform.node() == 'cnb-d102-22' and idx >= 10:
        #     break
        pkl_file_basename = osp.splitext(osp.basename(pkl_file_path))[0]
        vertices_clothed = None
        if args.use_clothed_mesh:
            clothed_mesh = trimesh.load(osp.join(args.rp_base_dir, pkl_file_basename + '.obj'), process=False)
            vertices_clothed = clothed_mesh.vertices

        print('file_name: {}'.format(pkl_file_path))
        # load pkl file
        vertices_org, vertices_can, faces_arr, body_model, R_can, pelvis, torch_param, vertices_clothed = data_utils.pkl_to_canonical(
            pkl_file_path, vertices_clothed=vertices_clothed, **args_dict)
        # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_can.detach().cpu().numpy())),
        #                                    o3d.geometry.TriangleMesh.create_coordinate_frame(
        #                                        size=0.6, origin=[0, 0, 0])
        #                                    ])
        # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_org.detach().cpu().numpy())),
        #                                    o3d.geometry.TriangleMesh.create_coordinate_frame(
        #                                        size=0.6, origin=[0, 0, 0]),
        #                                    # scene
        #                                    ])

        pelvis_z_offset = - vertices_org.detach().cpu().numpy().squeeze()[:, 2].min()
        pelvis_z_offset = pelvis_z_offset.clip(min=0.5)
        init_body_pose = body_model.body_pose.detach().clone()
        # DownSample
        vertices_org_ds = down_sample_fn.forward(vertices_org.unsqueeze(0).permute(0, 2, 1))
        vertices_org_ds = down_sample_fn2.forward(vertices_org_ds).permute(0, 2, 1).squeeze()
        vertices_can_ds = down_sample_fn.forward(vertices_can.unsqueeze(0).permute(0, 2, 1))
        vertices_can_ds = down_sample_fn2.forward(vertices_can_ds).permute(0, 2, 1).squeeze()

        if args.use_semantics:
            # load feature generated from IPOSA
            if args.load_feature:
                # fc = torch.tensor(torch_param['feature']['fc'], dtype=torch.float32, device=device)
                # fs = torch.tensor(torch_param['feature']['fs'], dtype=torch.float32, device=device)
                # gen_batches = torch.cat([fc.unsqueeze(1), fs], dim=1).unsqueeze(0)
                gen_batches = torch.tensor(torch_param['feature'], dtype=torch.float32, device=device).unsqueeze(0)
            else:
                scene_semantics = scene_data['scene_semantics']
                scene_obj_ids = np.unique(scene_semantics.nonzero().detach().cpu().numpy().squeeze()).tolist()
                n = 100
                print('Generating feature map')
                selected_batch = None
                z = torch.tensor(np.random.normal(0, 1, (n, args.z_dim)).astype(np.float32)).to(device)
                gen_batches = model.decoder(z, vertices_can_ds.unsqueeze(0).expand(n, -1, -1)).detach()

                for i in range(gen_batches.shape[0]):
                    x, x_semantics = data_utils.batch2features(gen_batches[i], **args_dict)
                    x_semantics = np.argmax(x_semantics, axis=-1)
                    # print(x_semantics.shape)  # (655, )

                    # if contain all specified objs, choose it
                    if args.obj_name != 'scene':

                        # objs = [category_dict.loc[id]['mpcat40'] for id in np.unique(x_semantics)]
                        objs = [id for id in np.unique(x_semantics)]
                        valid_contacts = 0
                        for obj in specified_objs:
                            valid_contacts += len(x_semantics[x_semantics == obj])

                        if set(objs).issuperset(set(specified_objs)) and valid_contacts > 10:
                            selected_batch = i
                            print(objs, valid_contacts)
                            break
                    else:
                        modes = stats.mode(x_semantics[x_semantics != 0])
                        if len(modes.mode) == 0:
                            print('no contacts')
                            continue
                        most_common_obj_id = modes.mode[0]
                        if most_common_obj_id not in scene_obj_ids:
                            continue
                        selected_batch = i
                        break

                if selected_batch is not None:
                    gen_batches = gen_batches[i].unsqueeze(0)
                else:
                    print('No good semantic feat found - Results might be suboptimal')
                    gen_batches = gen_batches[0].unsqueeze(0)
        else:
            z = torch.tensor(np.random.normal(0, 1, (args.num_rendered_samples, args.z_dim)).astype(np.float32)).to(
                device)
            gen_batches = model.decoder(z,
                                        vertices_can_ds.unsqueeze(0).expand(args.num_rendered_samples, -1, -1)).detach()

        for sample_id in range(gen_batches.shape[0]):
            result_filename = pkl_file_basename + '_' + args_dict.get('obj_name') + '_{:02d}'.format(sample_id)
            gen_batch = gen_batches[sample_id, :, :].unsqueeze(0)

            if args.show_gen_sample:
                gen = gen_batch.clone()
                gen_batch_us = up_sample_fn2.forward(gen.transpose(1, 2))
                gen_batch_us = up_sample_fn.forward(gen_batch_us).transpose(1, 2)
                if args.viz:
                    gen = viz_utils.show_sample(vertices_org, gen_batch_us, faces_arr, **args_dict)
                    o3d.visualization.draw_geometries(gen)
                if args.render:
                    gen_sample_img = viz_utils.render_sample(gen_batch_us, vertices_org, faces_arr, **args_dict)[0]
                    gen_sample_img.save(osp.join(affordance_dir, 'renderings', args_dict.get('scene_name'),
                                                 pkl_file_basename + '_' + args.obj_name + '_gen.png'))

            # continue

            # Create init points grid
            # basic_pose = None
            # if 'sit' in args.interaction:
            #     basic_pose = 'sit'
            # elif 'stand' in args.interaction:
            #     basic_pose = 'stand'
            # elif 'lie' in args.interaction:
            #     basic_pose = 'lie'

            # init_pos = torch.tensor(
            #     misc_utils.create_init_points(scene_data['bbox'].detach().cpu().numpy(), args.affordance_step,
            #                                   pelvis_z_offset), dtype=dtype, device=device).reshape(-1, 1, 3)
            init_pos = torch.tensor(scenes[args.scene_name].translation_sample_for_interaction(args.interaction, instance_nodes, sample_method='posa'),
                                    dtype=dtype, device=device).reshape(-1, 1, 3)
            # init_pos = torch.tensor([0.8, 3.0, 1.2], dtype=dtype, device=device).reshape(-1, 1, 3)

            # try to constrain init points around specified objects
            # if args.obj_name == 'scene':
            # else:
            #     min_bound = np.array([1.3, 1.3, 0.2])
            #     max_bound = np.array([2, 2, 1])
            #     x_offset = 0.25
            #     y_offset = 0.25
            #     z_offset = 0.5
            #     X, Y, Z = np.meshgrid(np.arange(min_bound[0] - x_offset, max_bound[0] + x_offset, args.affordance_step),
            #                           np.arange(min_bound[1] - y_offset, max_bound[1] + y_offset, args.affordance_step),
            #                           np.arange(min_bound[2], max_bound[2] + z_offset, args.affordance_step))
            #     points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
            #     init_pos = torch.tensor(points, dtype=dtype, device=device).reshape(-1, 1, 3)
            #     # obj_nodes = scenes[args.scene_name].object_nodes
            #     # valid_nodes = [node for node in obj_nodes if node.category_name == args.obj_name]
            #     # init_pos = []
            #     # for node in valid_nodes:
            #     #     x_offset = 0.25
            #     #     y_offset = 0.25
            #     #     z_offset = 0.5
            #     #     max_bound = node.aabb.max_bound
            #     #     min_bound = node.aabb.min_bound
            #     #     print(min_bound, max_bound)
            #     #     X, Y, Z = np.meshgrid(np.arange(min_bound[0] - x_offset, max_bound[0] + x_offset, args.affordance_step),
            #     #                           np.arange(min_bound[1] - y_offset, max_bound[1] + y_offset, args.affordance_step),
            #     #                           np.arange(min_bound[2], max_bound[2] + z_offset, args.affordance_step))
            #     #     points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
            #     #     init_pos.append(points)
            #     # init_pos = torch.tensor(np.stack(init_pos, axis=0), dtype=dtype, device=device).reshape(-1, 1, 3)
            #     # if (init_pos.shape[0] == 0):
            #     #     print('no specified object in this scene, search placement in whole scene')
            #     #     init_pos = torch.tensor(
            #     #         misc_utils.create_init_points(scene_data['bbox'].detach().cpu().numpy(), args.affordance_step,
            #     #                                       pelvis_z_offset), dtype=dtype, device=device).reshape(-1, 1, 3)

            if args.show_init_pos:
                points = [scene]
                points.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0]))
                points.append(
                    o3d.geometry.PointCloud(
                        points=o3d.utility.Vector3dVector(init_pos.detach().cpu().numpy().squeeze())
                    )
                )
                # for i in range(len(init_pos)):
                #     points.append(
                #         viz_utils.create_o3d_sphere(init_pos[i].detach().cpu().numpy().squeeze(), radius=0.03))
                o3d.visualization.draw_geometries(points)
            # Eval init points
            init_pos, init_ang = opt_utils.init_points_culling(init_pos=init_pos, vertices=vertices_org_ds,
                                                               scene_data=scene_data, gen_batch=gen_batch, **args_dict)

            if args.show_init_pos:
                points = []
                vertices_np = vertices_org.detach().cpu().numpy()
                bodies = []
                for i in range(len(init_pos)):
                    points.append(
                        viz_utils.create_o3d_sphere(init_pos[i].detach().cpu().numpy().squeeze(), radius=0.03))
                    rot_aa = torch.cat((torch.zeros((1, 2), device=device), init_ang[i].reshape(1, 1)), dim=1)
                    rot_mat = tgm.angle_axis_to_rotation_matrix(rot_aa.reshape(-1, 3))[:, :3,
                              :3].detach().cpu().numpy().squeeze()
                    v = np.matmul(rot_mat, vertices_np.transpose()).transpose() + init_pos[i].detach().cpu().numpy()
                    body = viz_utils.create_o3d_mesh_from_np(vertices=v, faces=faces_arr)
                    bodies.append(body)

                o3d.visualization.draw_geometries(points + [scene])

            ###########################################################################################################
            #####################            Start of Optimization Loop                  ##############################
            ###########################################################################################################
            results = []
            results_clothed = []
            losses = []
            rot_mats = []
            t_frees = []
            body_poses = []

            # geometries = [scene]
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
            # vis.set_full_screen(True)
            # vis.add_geometry(scene)
            world2cam = np.matmul(np.linalg.inv(scenes[args.scene_name].cam2world),
                                  POSA_to_PROX_transform[args.scene_name])

            for i in tqdm(range(init_pos.shape[0])):
                body_model.reset_params(**torch_param)
                t_free = init_pos[i].reshape(1, 1, 3).clone().detach().requires_grad_(True)
                ang_free = init_ang[i].reshape(1, 1).clone().detach().requires_grad_(True)

                free_param = [t_free, ang_free]
                if args.opt_pose:
                    free_param += [body_model.body_pose]
                optimizer, _ = optim_factory.create_optimizer(free_param, optim_type='lbfgsls',
                                                              lr=args_dict.get('affordance_lr'), ftol=1e-9,
                                                              gtol=1e-9,
                                                              max_iter=args.max_iter)

                opt_wrapper_obj = opt_utils.opt_wrapper(vertices=vertices_org_ds.unsqueeze(0),
                                                        vertices_can=vertices_can_ds, pelvis=pelvis,
                                                        scene_data=scene_data,
                                                        down_sample_fn=down_sample_fn, down_sample_fn2=down_sample_fn2,
                                                        optimizer=optimizer, gen_batch=gen_batch, body_model=body_model,
                                                        # vposer=vp,
                                                        init_body_pose=init_body_pose, **args_dict)

                closure = opt_wrapper_obj.create_fitting_closure(t_free, ang_free)
                for _ in range(10):
                    curr_results, rot_mat = opt_wrapper_obj.compute_vertices(t_free, ang_free,
                                                                             vertices=vertices_org.unsqueeze(0),
                                                                             down_sample=False)
                    # visualize optimization process
                    # body = viz_utils.create_o3d_mesh_from_np(vertices=curr_results.squeeze().detach().cpu().numpy(),
                    #                                          faces=faces_arr)
                    # vis.add_geometry(body)
                    # ctr = vis.get_view_control()
                    # cam_param = ctr.convert_to_pinhole_camera_parameters()
                    # cam_param.extrinsic = world2cam
                    # ctr.convert_from_pinhole_camera_parameters(cam_param)
                    # vis.poll_events()
                    # vis.update_renderer()
                    # # time.sleep(0.25)
                    # vis.remove_geometry(body)

                    loss = optimizer.step(closure)

                # Get body vertices after optimization
                curr_results, rot_mat = opt_wrapper_obj.compute_vertices(t_free, ang_free,
                                                                         vertices=vertices_org.unsqueeze(0),
                                                                         down_sample=False)
                rot_mats.append(rot_mat)
                t_frees.append(t_free)
                body_poses.append(body_model.body_pose.clone())

                if torch.is_tensor(loss):
                    loss = float(loss.detach().cpu().squeeze().numpy())
                losses.append(loss)
                results.append(curr_results.squeeze().detach().cpu().numpy())

                # Get clothed body vertices after optimization
                if vertices_clothed is not None:
                    curr_results_clothed, rot_mat = opt_wrapper_obj.compute_vertices(t_free, ang_free,
                                                                                     vertices=vertices_clothed.unsqueeze(
                                                                                         0),
                                                                                     down_sample=False)
                    results_clothed.append(curr_results_clothed.squeeze().detach().cpu().numpy())

            ###########################################################################################################
            #####################            End of Optimization Loop                  ################################
            ###########################################################################################################

            losses = np.array(losses)
            if len(losses > 0):
                idx = losses.argmin()
                print('minimum final loss = {}'.format(losses[idx]))
                sorted_ind = np.argsort(losses)
                for i in range(min(args.num_rendered_samples, len(losses))):
                    ind = sorted_ind[i]
                    cm = mpl_cm.get_cmap('Reds')
                    norm = mpl_colors.Normalize(vmin=0.0, vmax=1.0)
                    colors = cm(norm(losses))

                    # rot_mat and t_free also have multiple values like result, need to select suing index
                    rot_mat = rot_mats[ind]
                    t_free = t_frees[ind]

                    ## Save pickle
                    R_smpl2scene = torch.tensor(eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz'), dtype=dtype,
                                                device=device)
                    Rcw = torch.matmul(rot_mat.reshape(3, 3), R_smpl2scene)

                    # tranform POSA scene to prox scene
                    transform = torch.zeros((4, 4), dtype=torch.float32)
                    transform[3, 3] = 1
                    transform[:3, :3] = Rcw.clone()
                    transform[:3, 3] = t_free.clone().reshape(1, 3)
                    # print(transform, POSA_to_PROX_transform[args.scene_name])
                    transform = torch.tensor(POSA_to_PROX_transform[args.scene_name], dtype=torch.float32).matmul(transform)
                    # print(transform)
                    Rcw = transform[:3, :3].to(torch.device('cuda'))
                    t_free = transform[:3, 3].to(torch.device('cuda'))

                    torch_param['body_pose'] = body_poses[ind]
                    torch_param = misc_utils.smpl_in_new_coords(torch_param, Rcw, t_free.reshape(1, 3),
                                                                rotation_center=pelvis, **args_dict)
                    param = {}
                    for key in torch_param.keys():
                        if key == 'feature':
                            continue
                        param[key] = torch_param[key].detach().cpu() if key != 'gender' else torch_param[key]
                    with open(osp.join(pkl_folder, '{}.pkl'.format(result_filename)), 'wb') as f:
                        pickle.dump(param, f)
                    synthesis_results.append(param)

                    # Evaluate Physical Metric
                    gen_batch_us = up_sample_fn2.forward(gen_batch.transpose(1, 2))
                    gen_batch_us = up_sample_fn.forward(gen_batch_us).transpose(1, 2)

                    non_collision_score, contact_score = misc_utils.eval_physical_metric(
                        torch.tensor(results[ind], dtype=dtype, device=device).unsqueeze(0),
                        scene_data)
                    with open(osp.join(physical_metric_folder, '{}.yaml'.format(result_filename)), 'w') as f:
                        yaml.dump({'non_collision_score': non_collision_score,
                                   'contact_score': contact_score},
                                  f)

                    if args.viz:
                        bodies = [scene]
                        if args.use_clothed_mesh:
                            body = viz_utils.create_o3d_mesh_from_np(vertices=results_clothed[ind],
                                                                     faces=clothed_mesh.faces)

                        else:
                            body = viz_utils.create_o3d_mesh_from_np(vertices=results[ind], faces=faces_arr)
                        bodies.append(body)
                        o3d.visualization.draw_geometries(bodies)

                    if args.render or args.save_meshes:
                        default_vertex_colors = np.ones((results[ind].shape[0], 3)) * np.array(viz_utils.default_color)
                        body = trimesh.Trimesh(results[ind], faces_arr, vertex_colors=default_vertex_colors,
                                               process=False)
                        clothed_body = None
                        if args.use_clothed_mesh:
                            clothed_body = trimesh.Trimesh(results_clothed[ind], clothed_mesh.faces, process=False)

                        if args.save_meshes:
                            body.export(osp.join(affordance_dir, 'meshes', args_dict.get('scene_name'),
                                                 '{}_{}.obj'.format(result_filename, i)))

                            if args.use_clothed_mesh:
                                clothed_body.export(
                                    osp.join(affordance_dir, 'meshes_clothed', args_dict.get('scene_name'),
                                             '{}_{}.obj'.format(result_filename, i)))

                        if args.render:
                            scene_mesh = trimesh.load(
                                osp.join(base_dir, 'scenes', args_dict.get('scene_name') + '.ply'))
                            img_collage = viz_utils.render_interaction_snapshot(body, scene_mesh, clothed_body,
                                                                                body_center=True,
                                                                                collage_mode='horizantal', **args_dict)
                            img_collage.save(osp.join(rendering_folder, '{}.png'.format(result_filename)))
                            print('snapshot:', osp.join(rendering_folder, '{}.png'.format(result_filename)))

    with open(result_path, 'wb') as result_file:
        pickle.dump(synthesis_results, result_file)