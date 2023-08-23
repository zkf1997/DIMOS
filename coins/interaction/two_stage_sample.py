import os
os.environ['PYOPENGL_PLATFORM'] = "osmesa"

import numpy as np
import pytorch3d.transforms.rotation_conversions
import torch
import trimesh.util
import pylab
from tqdm import tqdm

from transform_trainer import *
from interaction_trainer import LitInteraction
from pointnet2 import farthest_point_sample
from viz_util import *
from sample_pelvis import to_pointcloud, composition_sample
from sample_interaction import get_composition_mask
from chamfer_distance import chamfer_dists

def rotation_matrix_vector_multiply(rot_mat, rot_vec):
    """Multiply rotations represented as matrix and axis-angle vector, return as axis-angle vector."""
    rotation = torch.matmul(rot_mat, pytorch3d.transforms.axis_angle_to_matrix(rot_vec))
    return pytorch3d.transforms.rotation_conversions.matrix_to_axis_angle(rotation)

def calc_interaction_loss(body, contact, object_pointclouds, scene, return_full=False):
    """
    Calculate interaction losses. Suppose all points in scene coordinate frame
    """
    batch_size = body.shape[0]
    contact_semantic, contact_scene = contact[:, :, 0], contact[:, :, 1]
    dists = chamfer_dists(body, object_pointclouds.reshape(batch_size, -1, 3))
    contact_dists = dists[contact_semantic > 0.5]
    loss_contact_semantic = torch.mean(torch.sqrt(contact_dists)) if len(contact_dists) else torch.tensor(0.0, device=body.device)
    # loss_penetration = calc_penetration_loss(scene_sdfs, body, thresh=0)
    sdf_values = scene.calc_sdf(body)
    loss_contact_scene = torch.tensor(0.0, device=body.device) if (contact_scene > 0.5).sum().item() < 1 else torch.mean(
        sdf_values[(contact_scene > 0.5)].abs())
    loss_penetration = torch.tensor(0.0, device=body.device) if sdf_values.lt(0.0).sum().item() < 1 else torch.mean(
        sdf_values[sdf_values < 0].abs())
    loss = loss_contact_scene * args.weight_contact_scene + loss_contact_semantic * args.weight_contact_semantic + loss_penetration * args.weight_penetration
    # print(loss_contact, loss_penetration)
    if return_full:
        return loss, loss_contact_semantic, loss_contact_scene, loss_penetration
    else:
        return loss

def posa_optimize(smplx_param, contact, pelvis_init, object_points, scene):
    """Optimize body pose, global translation, and global orientation guided by interaction-based terms"""
    batch_size = contact.shape[0]
    device = contact.device
    rotation_init = rot6d_to_mat(pelvis_init[:, :6])  # 1x3x3
    translation_init = pelvis_init[:, 6:]  # 1x3
    angle_z = torch.zeros((batch_size, 1), dtype=torch.float32, device=contact.device).requires_grad_(True)
    translation = translation_init.detach().clone().requires_grad_(True)
    body_model.reset_params(**smplx_param)
    init_body_pose = body_model.body_pose.detach().clone()
    body_model.body_pose.requires_grad_(True)

    param_list = [angle_z, translation, body_model.body_pose] if args.opt_pose else [angle_z, translation]
    optimizer = torch.optim.LBFGS(params=param_list, lr=args.lr_posa, max_iter=30) if args.optimizer == 'lbfgs' else torch.optim.Adam(params=param_list, lr=args.lr_posa)

    def closure(verbose=0):
        optimizer.zero_grad()
        rotation_z = pytorch3d.transforms.axis_angle_to_matrix(torch.cat((torch.zeros((batch_size, 2), device=device), angle_z), dim=1))
        rotation = torch.matmul(rotation_z, rotation_init)
        body_model_output = body_model(return_verts=True)
        pelvis = body_model_output.joints[:, 0, :].reshape(batch_size, 3)
        vertices_local = body_mesh.downsample(body_model_output.vertices - pelvis.unsqueeze(1))
        vertices_scene = torch.matmul(vertices_local, rotation.transpose(1, 2)) + translation.unsqueeze(1)
        loss_pose = F.mse_loss(body_model.body_pose, init_body_pose)
        loss_init = F.l1_loss(translation[:, :2], translation_init[:, :2]) + F.l1_loss(translation[:, 2], translation_init[:, 2]) * args.weight_z + angle_z.abs().mean()
        loss, loss_contact_semantic, loss_contact_scene, loss_penetration = calc_interaction_loss(vertices_scene, contact, object_points, scene, return_full=True)
        if args.annealing:
            # print('annealing', step / args.max_step_body)
            annealing_weight = 0 if step < args.max_step_body // 4 else ((step / args.max_step_body))
            loss_penetration = loss_penetration * annealing_weight
        loss_total = loss_contact_semantic * args.weight_contact_semantic + loss_contact_scene * args.weight_contact_scene + loss_penetration * args.weight_penetration + loss_pose * args.weight_pose + loss_init * args.weight_init
        loss_total.backward(retain_graph=True)
        if verbose:
            return loss_total, loss_contact_semantic, loss_contact_scene, loss_penetration, loss_pose, loss_init, rotation, vertices_scene
        else:
            return loss_total
    for step in range(args.max_step_body):
        optimizer.step(closure)

    loss_total, loss_contact_semantic, loss_contact_scene, loss_penetration, loss_pose, loss_init, rotation, vertices_scene = closure(verbose=1)
    if args.debug:
        print('total', loss_total.item(), 'semantic', loss_contact_semantic.item(), 'contact', loss_contact_scene.item(), 'penne', loss_penetration.item(), 'pose', loss_pose.item(), 'init', loss_init.item())
    smplx_param['body_pose'] = body_model.body_pose.detach().clone()
    smplx_param['global_orient'] = rotation_matrix_vector_multiply(rotation, smplx_param['global_orient'])
    smplx_param['transl'] = smplx_param['transl'] + translation
    # for key in smplx_param:
    #     smplx_param[key] = smplx_param[key].detach().clone()
    return loss_total.item(), smplx_param, vertices_scene.detach().clone()

def two_stage_sample(method='direct'):
    """
    Sample interactions in a given scene with the semantics of specified action-object pairs.
    When 'used_scene_names' and 'used_interactions' are set to 'all', this function iterates over all specified scenes and all action-object combinations and synthesize interactions for large scale evaluation.
    You can choose to synthesize a specific interaction in a specific scene by configuring 'used_scene_names' and 'used_interactions'.
    """
    for scene_name in tqdm(used_scene_names):
        # scene_sdfs = get_scene_sdfs([scene_name], device=device)
        scene = scenes[scene_name]
        scene_mesh = to_trimesh(scene.mesh)
        for interaction in tqdm(used_interactions):
            if scene.support_interaction(interaction):
                verbs, nouns, obj_combinations = scene.get_interaction_candidate_objects(interaction)
                verb_ids = [action_names.index(verb) for verb in verbs]
                if len(verb_ids) < maximum_atomics:
                    verb_ids = verb_ids + [-1] * (maximum_atomics - len(verb_ids))
                verb_ids = torch.tensor(verb_ids, device=device).unsqueeze(0)  # Bx2

                for combination in obj_combinations:
                    obj_meshes = []
                    combination_name = []
                    for atomic_idx, instance in enumerate(combination):
                        combination_name.append(verbs[atomic_idx] + '-' + nouns[atomic_idx] + '-' + str(instance.id))
                        obj_mesh = scene.get_mesh_with_accessory(instance.id)
                        obj_meshes.append(obj_mesh)
                    combination_name = '+'.join(combination_name)
                    # scene_mesh = trimesh.util.concatenate(obj_meshes)
                    print(combination_name)
                    object_points_scene = torch.from_numpy(np.stack(
                        [np.concatenate(pointcloud, axis=1) for pointcloud in
                         to_pointcloud(obj_meshes, num_points=interaction_model.args.num_obj_points, sample_surface=False)],
                        axis=0)).to(device).unsqueeze(0).expand(1, 2, -1, -1).clone()  # B x 2 * P * 9
                    object_points_floor = object_points_scene.clone()
                    offset = object_points_floor[:, :, :, :3].reshape(-1, 3).mean(axis=0)
                    offset[2] = scene.get_floor_height()
                    object_points_floor[:, :, :, :3] -= offset  # recenter around the objects, without this breaks on scannet scenes

                    pelvis_batch_input = {
                        'num_atomics': torch.ones(1, device=device) * len(verbs),
                        'object_pointclouds': object_points_floor,
                        'verb_ids': verb_ids,
                    }

                    # sample pelvis and body
                    samples = []
                    for sample_idx in range(args.num_sample):
                        if args.decode:
                            z_pelvis = z_pelvis_global[sample_idx]
                            z_body = z_body_global[sample_idx]
                        else:
                            z_pelvis = np.random.randn(args.num_try, transform_model.args.latent_dim).astype(np.float32)
                            z_body = np.random.randn(args.num_try, interaction_model.args.latent_dim).astype(np.float32)

                        best_loss = None
                        best_sample = None
                        for try_idx in range(args.num_try):
                            with torch.no_grad():
                                # sample pelvis frame
                                x, _ = transform_model.model.decode(pelvis_batch_input,
                                                                    z_sample=torch.from_numpy(
                                                                        z_pelvis[[try_idx], :]).to(device))
                                # x, _ = transform_model.model.sample(batch)
                                x = x.squeeze(1).detach()
                                x[:, 6:] += offset  # add floor height, back to PROX scene coords

                            if method in ['direct', 'optimization_after_get_body']:
                                with torch.no_grad():
                                    # sample body
                                    rotation = rot6d_to_mat(x[:, :6])
                                    pelvis = x[:, 6:]
                                    object_points_local = object_points_scene.clone()  # 1x2xPx9
                                    object_points_local[:, :, :, :3] = object_points_local[:, :, :, :3] - pelvis.reshape(-1, 1, 1,
                                                                                                                   3)
                                    object_points_local[:, :, :, :3] = torch.matmul(object_points_local[: , :, :, :3],
                                                                              rotation.reshape(-1, 1, 3, 3))  # coord
                                    object_points_local[:, :, :, 6:] = torch.matmul(object_points_local[: , :, :, 6:],
                                                                              rotation.reshape(-1, 1, 3, 3))  # normal
                                    body_batch_input = {
                                        'num_atomics': torch.ones(1, device=device) * len(verbs),
                                        'object_pointclouds': object_points_local,
                                        'verb_ids': verb_ids,
                                    }
                                    bodies, _ = interaction_model.model.decode(body_batch_input, z_sample=torch.from_numpy(z_body[[try_idx], :]).to(device))
                                    # bodies, _ = interaction_model.model.sample(batch)
                                    bodies, contact = bodies[:, :, :3], bodies[:, :, 3:]
                                    smplx_param, smplx_vertices = interaction_model.regress_smplx(bodies)
                                    bodies = interaction_model.mesh.downsample(smplx_vertices)

                                if method == 'direct':
                                    # transform back to scene coord frame
                                    bodies = torch.matmul(bodies, rotation.transpose(1, 2))
                                    bodies = bodies + pelvis.reshape(-1, 1, 3)
                                    smplx_param['global_orient'] = rotation_matrix_vector_multiply(rotation, smplx_param['global_orient'])
                                    smplx_param['transl'] = smplx_param['transl'] + pelvis
                                    loss = calc_interaction_loss(bodies, contact, object_points_scene[:, :, :, :3],
                                                                 scene)
                                    loss = loss.item()
                                else:
                                    loss, smplx_param, bodies = posa_optimize(smplx_param, contact, x, object_points_scene[:, :, :, :3], scene)

                            contact_semantic, contact_scene = contact[:, :, 0], contact[:, :, 1]
                            if np.isnan(loss):
                                loss = 2333  # result can be nan
                            if best_loss is None or loss < best_loss:
                                best_loss = loss
                                best_sample = {'body': bodies, 'contact': contact, 'smplx_param': smplx_param,
                                               'init_pelvis_frame': x}
                        samples.append(best_sample)
                        # visualize
                        if args.visualize:
                            frame = create_frame(best_sample['init_pelvis_frame'][0])
                            if body_type == 'mesh':
                                bodies, contact = best_sample['body'], best_sample['contact'][:, :, args.contact_dimension]
                                colors = np.array([[0.8, 0.8, 0.8]] * bodies.shape[1])
                                if interaction_model.args.use_contact_feature:
                                    colors[contact[0].detach().cpu().numpy() > 0.5] = (1., 1., 0.)
                                body_mesh = trimesh.Trimesh(
                                    vertices=bodies[0].detach().cpu().numpy(),
                                    faces=interaction_model.mesh.faces,
                                    vertex_colors=colors,
                                )
                            else:
                                body_mesh = skeleton_to_mesh(bodies[0].detach().cpu().numpy(), color=np.array(color_map(sample_idx / args.sample_num)))
                            body_meshes = [frame, body_mesh]

                            # render synthesis results
                            base_name = scene_name + '_' + combination_name + '_' + str(sample_idx) + '_' + method + '.png'
                            export_file = Path(args.save_dir, args.exp_name, method, interaction, base_name)
                            export_file.parent.mkdir(exist_ok=True, parents=True)
                            img_collage = render_interaction_multview(body=trimesh.util.concatenate(body_meshes),
                                                                      smooth_body=False,
                                                                      obj_points_coord=object_points_scene[0, 0, :, :3].detach().cpu().numpy(),
                                                                      static_scene=scene_mesh if args.full_scene else trimesh.util.concatenate(obj_meshes))
                            img_collage.save(str(export_file))
                            base_name = scene_name + '_' + combination_name + '_' + str(
                                sample_idx) + '_' + method + '_body.png'
                            export_file = Path(args.save_dir, args.exp_name, method, interaction, base_name)
                            img_collage = render_body_multview(body=body_mesh,)
                            img_collage.save(str(export_file))
                            # export ply mesh
                            smplx_vertices = interaction_model.body_model(**best_sample['smplx_param']).vertices.detach().cpu().numpy()[0]
                            body_mesh = trimesh.Trimesh(
                                vertices=smplx_vertices,
                                faces=interaction_model.mesh.meshes[0].faces,
                                vertex_colors=colors,
                            )
                            base_name = scene_name + '_' + combination_name + '_' + str(
                                sample_idx) + '_' + method + '.ply'
                            mesh_path = Path(args.save_dir, args.exp_name, method, interaction, base_name)
                            body_mesh.export(mesh_path)

                    # write smplx results
                    synthesis_results = [sample['smplx_param'] for sample in samples]
                    result_path = Path(args.save_dir, args.exp_name, method, interaction, scene_name, combination_name + '.pkl')
                    result_path.parent.mkdir(exist_ok=True, parents=True)
                    print(result_path)
                    with open(result_path, 'wb') as result_file:
                        pickle.dump(synthesis_results, result_file)

def two_stage_composition_sample(method='direct', mask_type='learned_by_part'):
    """
    Sample composite interactions in a given scene with the semantics of specified action-object pairs.
    """
    used_scenes = test_scenes if args.scene_name == 'test' else [args.scene_name]
    for scene_name in tqdm(used_scenes):
        scene = scenes[scene_name]
        scene_mesh = to_trimesh(scene.mesh)
        used_interactions = composed_interaction_names if args.interaction == 'all' else [args.interaction]
        for interaction in tqdm(used_interactions):
            if scenes[scene_name].support_interaction(interaction):
                verbs, nouns, obj_combinations = scenes[scene_name].get_interaction_candidate_objects(interaction)
                verb_ids = [[action_names.index(verb), -1] for verb in verbs]  # 2x2
                verb_ids = torch.tensor(verb_ids, device=device).unsqueeze(0)  # Bx2x2
                verb_ids_composite = [action_names.index(verb) for verb in verbs]
                verb_ids_composite = torch.tensor(verb_ids_composite, device=device).unsqueeze(0)  # Bx2

                for combination in obj_combinations:
                    obj_meshes = []
                    combination_name = []
                    for atomic_idx, instance in enumerate(combination):
                        combination_name.append(verbs[atomic_idx] + '-' + nouns[atomic_idx] + '-' + str(instance.id))
                        obj_mesh = scenes[scene_name].get_mesh_with_accessory(instance.id)
                        obj_meshes.append(obj_mesh)
                    combination_name = '+'.join(combination_name)
                    print(combination_name)
                    pointcloud_list = to_pointcloud(obj_meshes, num_points=args.num_points)

                    object_points_padded = np.zeros((maximum_atomics, maximum_atomics, args.num_points, 9), dtype=np.float32)
                    for obj_idx, obj_pointcloud in enumerate(pointcloud_list):
                        object_points_padded[obj_idx, 0, :, :] = np.concatenate(obj_pointcloud, axis=1)
                        object_points_padded[obj_idx, 1, :, :] = object_points_padded[obj_idx, 0, :, :]  # copy last padding
                    object_points_padded = torch.tensor(object_points_padded, device=device).unsqueeze(0).contiguous() # 1x2x2xPx9
                    object_points_scene = object_points_padded[:, :, 0, :, :].clone()
                    object_points_padded[:, :, :, :, 2] -= scene.get_floor_height()

                    batch_list = [{
                        'num_atomics': torch.ones(1, device=device),
                        'object_pointclouds': object_points_padded[:, 0, :, :, :],  # Bx2xPx9
                        'verb_ids': verb_ids[:, 0, :],  # Bx2
                    }, {
                        'num_atomics': torch.ones(1, device=device),
                        'object_pointclouds': object_points_padded[:, 1, :, :, :],  # Bx2xPx9
                        'verb_ids': verb_ids[:, 1, :],  # Bx2
                    }]

                    # sample pelvis and body
                    samples = []
                    for sample_idx in range(args.num_sample):
                        if args.decode:
                            z_pelvis = z_pelvis_global[sample_idx]
                            z_body = z_body_global[sample_idx]
                        else:
                            z_pelvis = np.random.randn(args.num_try, transform_model.args.latent_dim).astype(np.float32)
                            z_body = np.random.randn(args.num_try, interaction_model.args.latent_dim).astype(np.float32)

                        best_loss = None
                        best_sample = None
                        for try_idx in range(args.num_try):
                            # sample pelvis frame
                            x = composition_sample(transform_model, 1, batch_list,
                                                   # z=z_pelvis,
                                                   lr=args.lr_pelvis,
                                                   max_step=args.max_step_pelvis,
                                                   weight_prob=args.weight_prob_pelvis)
                            x = x.squeeze(1).detach()
                            x[:, -1] += scene.get_floor_height()

                            with torch.no_grad():
                                rotation = rot6d_to_mat(x[:, :6])
                                pelvis = x[:, 6:]
                                object_points_local = object_points_scene.clone()  # Bx2xPx9
                                object_points_local[:, :, :, :3] = object_points_local[:, :, :, :3] - pelvis.reshape(-1, 1, 1, 3)
                                object_points_local[:, :, :, :3] = torch.matmul(object_points_local[:, :, :, :3],
                                                                          rotation.reshape(-1, 1, 3, 3))  # coord
                                object_points_local[:, :, :, 6:] = torch.matmul(object_points_local[:, :, :, 6:],
                                                                          rotation.reshape(-1, 1, 3, 3))  # normal

                                # sample body
                                batch = {
                                    'num_atomics': torch.ones(1, device=device) * len(verbs),
                                    'object_pointclouds': object_points_local,
                                    'verb_ids': verb_ids_composite,
                                }
                                bodies, _ = interaction_model.model.decode(
                                    batch, z_sample=torch.from_numpy(z_body[[try_idx], :]).to(device),
                                    composition_mask=False if mask_type == 'naive' else get_composition_mask(mask_type,
                                                                                                             scene_name,
                                                                                                             interaction.split(
                                                                                                                 '+'),
                                                                                                             interaction_model))
                                # bodies, _ = interaction_model.model.sample(batch)
                                # transform back to scene coord frame
                                bodies, contact = bodies[:, :, :3].detach(), bodies[:, :, 3:]
                                smplx_param, smplx_vertices = interaction_model.regress_smplx(bodies)
                                bodies = interaction_model.mesh.downsample(smplx_vertices)

                            if method == 'direct':
                                bodies = torch.matmul(bodies, rotation.transpose(1, 2))
                                bodies = bodies + x[:, 6:].reshape(-1, 1, 3)
                                smplx_param['global_orient'] = rotation_matrix_vector_multiply(rotation,
                                                                                               smplx_param['global_orient'])
                                smplx_param['transl'] = smplx_param['transl'] + pelvis
                                loss = calc_interaction_loss(bodies, contact, object_points_scene[:, :, :, :3],
                                                             scene)
                                loss = loss.item()
                            else:
                                loss, smplx_param, bodies = posa_optimize(smplx_param, contact, x,
                                                                          object_points_scene[:, :, :, :3], scene)

                            contact_semantic, contact_scene = contact[:, :, 0], contact[:, :, 1]
                            if np.isnan(loss):
                                loss = 2333  # result can be nan
                            if best_loss is None or loss < best_loss:
                                best_loss = loss
                                best_sample = {'body': bodies, 'contact': contact, 'smplx_param': smplx_param,
                                               'init_pelvis_frame': x}
                        samples.append(best_sample)

                        # visualize
                        if args.visualize:
                            frame = create_frame(best_sample['init_pelvis_frame'][0])
                            if body_type == 'mesh':
                                bodies, contact = best_sample['body'], best_sample['contact'][:, :, args.contact_dimension]
                                colors = np.array([[0.8, 0.8, 0.8]] * bodies.shape[1])
                                if interaction_model.args.use_contact_feature:
                                    colors[contact[0].detach().cpu().numpy() > 0.5] = (1., 1., 0.)
                                body_mesh = trimesh.Trimesh(
                                    vertices=bodies[0].detach().cpu().numpy(),
                                    faces=interaction_model.mesh.faces,
                                    vertex_colors=colors,
                                )
                            else:
                                body_mesh = skeleton_to_mesh(bodies[0].detach().cpu().numpy(),
                                                             color=np.array(color_map(sample_idx / args.num_sample)))
                            body_meshes = [frame, body_mesh]

                            base_name = scene_name + '_' + combination_name + '_' + str(sample_idx) + '_' + method + '.png'
                            export_file = Path(args.save_dir, args.exp_name, method + '_' + mask_type, interaction, base_name)
                            export_file.parent.mkdir(exist_ok=True, parents=True)
                            img_collage = render_interaction_multview(body=trimesh.util.concatenate(body_meshes),
                                                                      smooth_body=False,
                                                                      static_scene=scene_mesh if args.full_scene else trimesh.util.concatenate(obj_meshes))
                            img_collage.save(str(export_file))
                            base_name = scene_name + '_' + combination_name + '_' + str(
                                sample_idx) + '_' + method + '_body.png'
                            export_file = Path(args.save_dir, args.exp_name, method + '_' + mask_type, interaction, base_name)
                            img_collage = render_body_multview(body=body_mesh, )
                            img_collage.save(str(export_file))
                            # export ply mesh
                            smplx_vertices = interaction_model.body_model(**best_sample['smplx_param']).vertices.detach().cpu().numpy()[0]
                            body_mesh = trimesh.Trimesh(
                                vertices=smplx_vertices,
                                faces=interaction_model.mesh.meshes[0].faces,
                                vertex_colors=colors,
                            )
                            base_name = scene_name + '_' + combination_name + '_' + str(
                                sample_idx) + '_' + method + '.ply'
                            mesh_path = Path(args.save_dir, args.exp_name, method + '_' + mask_type, interaction, base_name)
                            body_mesh.export(mesh_path)

                    # write smplx results
                    synthesis_results = [sample['smplx_param'] for sample in samples]
                    result_path = Path(args.save_dir, args.exp_name, method + '_' + mask_type, interaction, scene_name,
                                       combination_name + '.pkl')
                    result_path.parent.mkdir(exist_ok=True, parents=True)
                    print(result_path)
                    with open(result_path, 'wb') as result_file:
                        pickle.dump(synthesis_results, result_file)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--transform_checkpoint", type=str, default='pelvis.ckpt')
    parser.add_argument("--interaction_checkpoint", type=str, default='body.ckpt')
    parser.add_argument("--save_dir", type=str, default="two_stage")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--scene_name", type=str, default="test")
    parser.add_argument("--interaction", type=str, default="all")
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--num_sample", type=int, default=32)
    parser.add_argument("--num_try", type=int, default=10)
    parser.add_argument("--decode", type=int, default=0)
    parser.add_argument("--visualize", type=int, default=0)
    parser.add_argument("--full_scene", type=int, default=0)
    parser.add_argument("--composition", type=int, default=0)
    parser.add_argument("--contact_dimension", type=int, default=1, help="0:semantic, 1:scene")

    parser.add_argument("--lr_pelvis", type=float, default=0.1)
    parser.add_argument("--weight_prob_pelvis", type=float, default=0.5)
    parser.add_argument("--max_step_pelvis", type=int, default=100)

    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr_body", type=float, default=1e-2)
    parser.add_argument("--lr_posa", type=float, default=1.0)
    parser.add_argument("--weight_prob_body", type=float, default=0.)
    parser.add_argument("--weight_init", type=float, default=0)
    parser.add_argument("--weight_z", type=float, default=10)
    parser.add_argument("--weight_contact_semantic", type=float, default=1.0)
    parser.add_argument("--weight_contact_scene", type=float, default=0.0)
    parser.add_argument("--weight_penetration", type=float, default=10.0)
    parser.add_argument("--weight_pose", type=float, default=100.0)
    parser.add_argument("--opt_pose", type=int, default=1)
    parser.add_argument("--max_step_body", type=int, default=100)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--annealing", type=int, default=0)
    args = parser.parse_args()
    args.save_dir = results_folder / args.save_dir

    device = torch.device('cuda')
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    transform_model = LitTransformNet.load_from_checkpoint(checkpoint_folder / args.transform_checkpoint).to(device)
    interaction_model = LitInteraction.load_from_checkpoint(checkpoint_folder / args.interaction_checkpoint).to(device)
    interaction_model.eval()
    transform_model.eval()
    body_model = interaction_model.body_model = smplx.create(smplx_model_folder, model_type='smplx',
                                                   gender='neutral', ext='npz',
                                                   num_pca_comps=num_pca_comps, batch_size=1).to(device)
    body_mesh = interaction_model.mesh
    body_type = interaction_model.args.body_type
    color_map = pylab.get_cmap('gist_yarg')

    if args.scene_name == 'test':
        used_scene_names = test_scenes
    else:
        used_scene_names = [args.scene_name]

    if args.interaction == 'all':
        used_interactions = interaction_names
    else:
        used_interactions = [args.interaction]

    torch.manual_seed(233)
    np.random.seed(233)
    z_pelvis_global = np.random.randn(args.num_sample, args.num_try, transform_model.args.latent_dim).astype(np.float32)
    z_body_global = np.random.randn(args.num_sample, args.num_try, interaction_model.args.latent_dim).astype(np.float32)
    if args.composition:
        two_stage_composition_sample(method='optim', mask_type='learned_by_part')
    else:
        two_stage_sample(method='optimization_after_get_body')
        two_stage_sample(method='direct')






