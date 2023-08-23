import numpy as np
import torch
import trimesh
from tqdm import tqdm
import io
from PIL import Image
import glob

# from skeleton_trainer import *
from transform_trainer import *
from pointnet2 import farthest_point_sample
from viz_util import *
from utils import *

def to_pointcloud(obj_meshes, num_points, sample_surface=False):
    """Convert object meshes to pointclouds with specified number of points"""
    obj_pointcloud = []
    for obj_mesh in obj_meshes:
        if sample_surface:
            o3d_mesh = to_open3d(deepcopy(obj_mesh))
            o3d_pointcloud = o3d_mesh.sample_points_uniformly(number_of_points=num_points * 16)
            obj_vertices = np.asarray(o3d_pointcloud.points, dtype=np.float32)
            obj_vertex_colors = np.asarray(o3d_pointcloud.colors,
                                           dtype=np.float32)
            if obj_vertex_colors.shape[0] == 0:
                obj_vertex_colors = np.ones_like(obj_vertices) * 0.5
            obj_vertex_normals = np.asarray(o3d_pointcloud.normals, dtype=np.float32)
            # print(obj_vertex_colors.shape, obj_vertex_normals.shape)
        else:
            obj_vertices = np.asarray(obj_mesh.vertices, dtype=np.float32).copy()
            obj_vertex_colors = np.asarray(obj_mesh.visual.vertex_colors[:, :3], dtype=np.float32).copy() / 255.0  # [0, 255]
            obj_vertex_normals = np.asarray(obj_mesh.vertex_normals, dtype=np.float32).copy()

        if obj_vertices.shape[0] > num_points:
            idx = np.squeeze(farthest_point_sample(torch.from_numpy(obj_vertices[None, :, :]), npoint=num_points))
            obj_vertices = obj_vertices[idx, :]
            obj_vertex_colors = obj_vertex_colors[idx, :]
            obj_vertex_normals = obj_vertex_normals[idx, :]
        elif obj_vertices.shape[0] < num_points:
            idx = np.random.choice(np.arange(obj_vertices.shape[0]), num_points, replace=True)
            obj_vertices = obj_vertices[idx, :]
            obj_vertex_colors = obj_vertex_colors[idx, :]
            obj_vertex_normals = obj_vertex_normals[idx, :]
        obj_pointcloud.append((obj_vertices, obj_vertex_colors, obj_vertex_normals))

    return obj_pointcloud

def composition_sample(model, batch_size, batch_list, optimizer=None, lr=3e-4, max_step=100, weight_prob=1.0, z=None):
    """Compositional pelvis sampling. Sample separate latent codes for each atomic interaction, then optimize the latent codes to make the decoded pelvis frames to be as close as possible."""
    assert len(batch_list) == 2
    device = model.device
    prior = torch.distributions.normal.Normal(
        loc=torch.zeros((batch_size, model.args.latent_dim), device=device),
        scale=torch.ones((batch_size, model.args.latent_dim), device=device))
    if z is None:
        z0 = nn.Parameter(torch.randn((batch_size, model.args.latent_dim), device=device))
        z1 = nn.Parameter(torch.randn((batch_size, model.args.latent_dim), device=device))
    else:
        z0 = z.detach().clone()
        z1 = z.detach().clone()
    # print(z0, z1)
    params = [z0, z1]
    optimizer = torch.optim.Adam(params=params,
                                  lr=lr)
    for step in range(max_step):
        optimizer.zero_grad()
        sample0, _ = model.model.decode(batch_list[0], z0)
        sample1, _ = model.model.decode(batch_list[1], z1)
        consistency_loss = F.l1_loss(sample0, sample1)
        log_prob = (prior.log_prob(z0) + prior.log_prob(z1)).mean() / 2.0
        # print(consistency_loss, log_prob)
        loss = consistency_loss - weight_prob * log_prob
        loss.backward()
        optimizer.step()
    result = ((sample0 + sample1) / 2.0).detach().clone()
    return result

def visualize_distribution():
    """Sample multiple pelvis frames for each combination of action-object, render the pelvis frames to illustrate the pelvis distribution."""
    used_scenes = test_scenes if args.scene_name == 'test' else [args.scene_name]
    for scene_name in tqdm(used_scenes):
        # for scene_name in tqdm(['MPH16']):
        scene_mesh = deepcopy(to_trimesh(scenes[scene_name].mesh))
        scene_mesh.vertices -= np.array([0.0, 0.0, scenes[scene_name].get_floor_height()])  #convert to height relative to floor
        used_interactions = interaction_names if args.interaction == 'all' else [args.interaction]
        for interaction in tqdm(used_interactions):
            if scenes[scene_name].support_interaction(interaction):
                verbs, nouns, obj_combinations = scenes[scene_name].get_interaction_candidate_objects(interaction)
                verb_ids = [action_names.index(verb) for verb in verbs]
                if len(verb_ids) < maximum_atomics:
                    verb_ids = verb_ids + [-1] * (maximum_atomics - len(verb_ids))
                verb_ids = torch.tensor(verb_ids, device=device).unsqueeze(0).expand(args.sample_num, -1)  # Bx2

                for combination in obj_combinations:
                    obj_meshes = []
                    combination_name = ''
                    for atomic_idx, instance in enumerate(combination):
                        combination_name += verbs[atomic_idx] + '-' + nouns[atomic_idx] + '_' + str(instance.id)
                        obj_mesh = deepcopy(scenes[scene_name].get_mesh_with_accessory(instance.id))
                        obj_mesh.vertices -= np.array([0.0, 0.0, scenes[scene_name].get_floor_height()])  #convert to height relative to floor
                        obj_meshes.append(obj_mesh)
                    pointcloud_list = to_pointcloud(obj_meshes, num_points=transform_model.args.num_obj_points)
                    print(combination_name)

                    object_points = np.zeros((maximum_atomics, transform_model.args.num_obj_points, 9), dtype=np.float32)
                    for obj_idx, obj_pointcloud in enumerate(pointcloud_list):
                        object_points[obj_idx, :, :] = np.concatenate(obj_pointcloud, axis=1)
                    # copy last padding
                    if len(pointcloud_list) < maximum_atomics:
                        object_points[1, :, :] = object_points[0, :, :]
                    object_points = torch.tensor(object_points, device=device).unsqueeze(0).expand(args.sample_num, -1,
                                                                                                   -1, -1)  # Bx2xPx9

                    batch = {
                        'num_atomics': torch.ones(args.sample_num, device=device) * len(verbs),
                        'object_pointclouds': object_points,
                        'verb_ids': verb_ids,
                    }
                    # sample pelvis frame
                    x, attention = transform_model.model.sample(batch)
                    x = x.squeeze(1)

                    body_meshes = []
                    for sample_idx in range(args.sample_num):
                        body_meshes.append(create_frame(x[sample_idx]))
                    body_mesh = trimesh.util.concatenate(body_meshes)

                    img_collage = img_collage = render_interaction_multview(body=body_mesh,
                                                                      static_scene=scene_mesh if args.full_scene else trimesh.util.concatenate(obj_meshes))
                    file_name = scene_name + '_' + combination_name + '.png'
                    file_path = Path(args.save_dir, args.exp_name, 'distribution', interaction, file_name)
                    file_path.parent.mkdir(exist_ok=True, parents=True)
                    img_collage.save(file_path)

def visualize_composite_sample_distribution():
    """Sample multiple pelvis frames by calling composition_sample, render the pelvis frames to illustrate the pelvis distribution."""
    used_scenes = test_scenes if args.scene_name == 'test' else [args.scene_name]
    for scene_name in tqdm(used_scenes):
        scene_mesh = deepcopy(to_trimesh(scenes[scene_name].mesh))
        scene_mesh.vertices -= np.array(
        [0.0, 0.0, scenes[scene_name].get_floor_height()])  # convert to height relative to floor
        used_interactions = composed_interaction_names if args.interaction == 'all' else [args.interaction]
        for interaction in tqdm(used_interactions):
            if '+' in interaction and scenes[scene_name].support_interaction(interaction):
                verbs, nouns, obj_combinations = scenes[scene_name].get_interaction_candidate_objects(interaction)
                verb_ids = [[action_names.index(verb), -1] for verb in verbs]  # 2x2
                verb_ids = torch.tensor(verb_ids, device=device).unsqueeze(0)  # Bx2x2

                for combination in obj_combinations:
                    obj_meshes = []
                    combination_name = ''
                    for atomic_idx, instance in enumerate(combination):
                        combination_name += verbs[atomic_idx] + '-' + nouns[atomic_idx] + '_' + str(instance.id)
                        obj_mesh = deepcopy(scenes[scene_name].get_mesh_with_accessory(instance.id))
                        obj_mesh.vertices -= np.array(
                            [0.0, 0.0, scenes[scene_name].get_floor_height()])  # convert to height relative to floor
                        obj_meshes.append(obj_mesh)
                    pointcloud_list = to_pointcloud(obj_meshes, num_points=transform_model.args.num_obj_points)
                    print(combination_name)

                    object_points = np.zeros((maximum_atomics, maximum_atomics, transform_model.args.num_obj_points, 9), dtype=np.float32)
                    for obj_idx, obj_pointcloud in enumerate(pointcloud_list):
                        object_points[obj_idx, 0, :, :] = np.concatenate(obj_pointcloud, axis=1)
                        object_points[obj_idx, 1, :, :] = object_points[obj_idx, 0, :, :] # copy last padding
                    object_points = torch.tensor(object_points, device=device).unsqueeze(0)  # Bx2x2xPx9

                    batch_list = [{
                        'num_atomics': torch.ones(1, device=device),
                        'object_pointclouds': object_points[:, 0, :, :, :],  # Bx2xPx9
                        'verb_ids': verb_ids[:, 0, :],  # Bx2
                    }, {
                        'num_atomics': torch.ones(1, device=device),
                        'object_pointclouds': object_points[:, 1, :, :, :],  # Bx2xPx9
                        'verb_ids': verb_ids[:, 1, :],  # Bx2
                    }]
                    # sample pelvis frame
                    results = []
                    for _ in range(args.sample_num):
                        x = composition_sample(transform_model, 1, batch_list,
                                               lr=args.lr,
                                               max_step=args.max_step,
                                               weight_prob=args.weight_prob)
                        x = x.squeeze(1).detach().clone()
                        results.append(x)
                    x = torch.cat(results, dim=0)

                    body_meshes = []
                    for sample_idx in range(args.sample_num):
                        body_meshes.append(create_frame(x[sample_idx]))
                    body_mesh = trimesh.util.concatenate(body_meshes)

                    img_collage = render_interaction_multview(body=body_mesh,
                                                                      static_scene=scene_mesh if args.full_scene else trimesh.util.concatenate(obj_meshes))
                    file_name = scene_name + '_' + combination_name + '.png'
                    file_path = Path(args.save_dir, args.exp_name, 'composite_distribution', interaction, file_name)
                    file_path.parent.mkdir(exist_ok=True, parents=True)
                    img_collage.save(file_path)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--transform_checkpoint", type=str, default="/mnt/scratch/scene_graph/results/transform/evaluation_atomic/version_2/checkpoints/last.ckpt")
    # parser.add_argument("--transform_checkpoint", type=str, default="/local/home/zkf/scene_graph/results/transform/test/version_31/checkpoints/last.ckpt")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--save_dir", type=str, default="/local/home/zkf/scene_graph/render/sample_pelvis")
    # parser.add_argument("--num_points", type=int, default=512)

    parser.add_argument("--scene_name", type=str, default="test")
    parser.add_argument("--interaction", type=str, default="all")
    parser.add_argument("--sample_num", type=int, default=32)
    parser.add_argument("--full_scene", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--weight_prob", type=float, default=0.1)
    parser.add_argument("--max_step", type=int, default=500)
    args = parser.parse_args()

    device = torch.device('cuda')

    transform_model = LitTransformNet.load_from_checkpoint(args.transform_checkpoint).to(device)
    transform_model.args.mask_body = 0

    with torch.no_grad():
        visualize_distribution()

    visualize_composite_sample_distribution()






