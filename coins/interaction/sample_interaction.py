import sys

import torch

sys.path.append('..')
from tqdm import tqdm

from interaction.interaction_trainer import *
from interaction.utils import *
from interaction.viz_util import *

def get_data_loader():
    """
    Create a interaction data loader where we select one frame from each action-object combination.
    """
    with open(Path.joinpath(project_folder, "data", 'test.pkl'), 'rb') as data_file:
        test_data = pickle.load(data_file)
    with open(Path.joinpath(project_folder, "data", 'train.pkl'), 'rb') as data_file:
        train_data = pickle.load(data_file)
    test_dataset = InteractionDataset(test_data + train_data,
                                      num_points=interaction_model.args.num_obj_points, use_augment=False,
                                      used_interaction='all', split='test', use_composite=True,
                                      center_type=interaction_model.args.center_type,
                                      data_overwrite=args.data_overwrite, point_sample='uniform',
                                      )
    # single frame test
    if args.decode:
        selected_frames = [
            'MPH1Library_00034_01_200',
            'MPH1Library_00034_01_1049',
            'N3OpenArea_03301_01_201',
            'MPH16_00157_01_1226', 'MPH1Library_00034_01_798',
            'N3OpenArea_00158_01_831'
        ]
        test_dataset.data = [record for record in test_dataset.data if (record['sequence'] + '_' + str(record['frame_idx'])) in selected_frames]

    data = []
    instances_set = set()
    for record in test_dataset.data:
        if args.composite_only and not '+' in record['interaction']:
            continue
        scene_name = record['scene_name']
        atomics = record['interaction'].split('+')
        obj_ids = [record['interaction_obj_idx'][record['interaction_labels'].index(atomic)] for atomic in atomics]
        instances_id = scene_name
        for atomic_idx in range(len(atomics)):
            instances_id += '_' + atomics[atomic_idx] + '_' + str(int(obj_ids[atomic_idx]))

        if instances_id not in instances_set:
            # print(instances_id)
            # print(record['sequence'], record['frame_idx'])
            instances_set.add(instances_id)
            data = data + [record] * args.sample_num
    test_dataset.data = data
    test_loader = DataLoader(test_dataset, batch_size=args.sample_num, num_workers=4, shuffle=False,
                             drop_last=True, pin_memory=False)
    return test_loader

def sample(sample_num=1024):
    """Sample interactions using inputs in data loader"""
    sample_dict = {}
    for batch in tqdm(data_loader):
        scene_name, num_atomics = batch['scene_name'][0], batch['num_atomics'][0]
        atomics = batch['interaction'][0].split('+')
        obj_ids = batch['interaction_obj_ids'][0][:num_atomics]
        base_name = scene_name
        for atomic_idx in range(num_atomics):
            base_name += '_' + atomics[atomic_idx] + '_' + str(int(obj_ids[atomic_idx].item()))
        print(base_name)
        sample_dict[base_name] = []
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)

        for _ in range(sample_num // args.sample_num):
            bodies, attention_list = interaction_model.model.sample(batch)
            bodies = transform_back(bodies, batch['centroid'], batch['rotation'])
            sample_dict[base_name].append(bodies)
        if len(atomics) == 2:
            sample_dict[base_name + '_composition'] = []
            for _ in range(sample_num // args.sample_num):
                bodies, attention_list = interaction_model.model.sample_composition(batch)
                bodies = transform_back(bodies, batch['centroid'], batch['rotation'])
                sample_dict[base_name + '_composition'].append(bodies)
        # sample_dict[base_name] = torch.cat(sample_dict[base_name], dim=0)
    return sample_dict

with open(project_folder / 'data' / 'contact_statistics.json', 'r') as f:
    contact_statistics = json.load(f)
def get_composition_mask(composition_mask_type, scene_name, atomics, interaction_model,
                         mask_thresh_by_vertex=-0.2, mask_thresh_by_part=-10):
    num_atomics = len(atomics)
    verbs = [atomic.split('-')[0] for atomic in atomics]
    Pb, Po = interaction_model.args.num_body_points, interaction_model.args.num_obj_keypoints
    contact_probability = np.asarray([contact_statistics['probability'][verb] for verb in verbs])  # 2xPb
    contact_probability = contact_probability - np.max(contact_probability, axis=0, keepdims=True)

    if composition_mask_type == 'learned_by_vertex':
        composition_mask = torch.zeros((Pb + Po * 2, Pb + Po * 2), dtype=torch.bool)
        # diagonal
        composition_mask[Pb:Pb + Po, Pb + Po:] = True
        composition_mask[Pb + Po:, Pb:Pb + Po] = True
        for atomic_idx in range(len(atomics)):
            mask_vertices = np.nonzero(contact_probability[atomic_idx, :] < mask_thresh_by_vertex)[0]
            composition_mask[mask_vertices, Pb + Po * atomic_idx: Pb + Po * (atomic_idx+1)] = True
        return composition_mask
    if composition_mask_type == 'learned_by_part':
        composition_mask = torch.zeros((Pb + Po * 2, Pb + Po * 2), dtype=torch.bool)
        # diagonal
        composition_mask[Pb:Pb + Po, Pb + Po:] = True
        composition_mask[Pb + Po:, Pb:Pb + Po] = True
        # print(atomics)
        for atomic_idx in range(len(atomics)):
            for part_index, part_vertices in enumerate(interaction_model.args.body_segment):
                if contact_probability[atomic_idx, part_vertices].sum() < mask_thresh_by_part:
                    # print(atomic_idx, part_index, contact_probability[atomic_idx, part_vertices].sum())
                    composition_mask[part_vertices, Pb + Po * atomic_idx: Pb + Po * (atomic_idx + 1)] = True
        return composition_mask
    else:
        return composition_mask_type

def render(z=None):
    """Sample interactions using inputs in data loader and render results"""
    for batch in tqdm(data_loader):
        scene_name, num_atomics = batch['scene_name'][0], batch['num_atomics'][0]
        atomics = batch['interaction'][0].split('+')
        obj_ids = batch['interaction_obj_ids'][0][:num_atomics]
        base_name = scene_name
        for atomic_idx in range(num_atomics):
            base_name += '_' + atomics[atomic_idx] + '_' + str(int(obj_ids[atomic_idx].item())) + '_' + str(int(batch['frame_idx'][0]))
        print(base_name)
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)

        # print('sample start')
        if z is None:
            bodies, attention_list = interaction_model.model.sample(batch) if (len(atomics) == 1 or args.composition_sample == 'no') else interaction_model.model.sample(batch, composition_mask=args.composition_sample)
        else:
            bodies, attention_list = interaction_model.model.decode(batch, z_sample=z) if (
                    len(atomics) == 1 or args.composition_sample == 'no') else interaction_model.model.decode(
                batch, z_sample=z, composition_mask=get_composition_mask(args.composition_sample, scene_name, atomics,
                                                                         interaction_model,
                                                                         mask_thresh_by_vertex=args.mask_thresh_by_vertex,
                                                                         mask_thresh_by_part=args.mask_thresh_by_part))
        if interaction_model.args.use_contact_feature:
            bodies, contact = bodies[:, :, :3], bodies[:, :, 3]
        # print(bodies.shape)
        # print('sample finished')
        bodies = transform_back(bodies, batch['centroid'], batch['rotation'])
        obj_points_coord = transform_back(batch['object_pointclouds'][:, :, :, :3].reshape(batch_size, -1, 3),
                                          batch['centroid'],
                                          batch['rotation']).reshape(batch_size, maximum_atomics, -1, 3).cpu().numpy()
        # print('transform back')
        obj_meshes = []
        if args.full_scene:
            obj_meshes.append(to_trimesh(scenes[scene_name].mesh))
        else:
            for obj_idx in obj_ids:
                if obj_idx != -1:
                    obj_mesh = scenes[scene_name].get_mesh_with_accessory(int(obj_idx))
                    obj_meshes.append(obj_mesh)
        for idx in range(args.sample_num):
            body_mesh = None
            if body_type == 'mesh':
                colors = np.array([[0.8, 0.8, 0.8]] * bodies.shape[1])
                if interaction_model.args.use_contact_feature:
                    colors[contact[idx].detach().cpu().numpy() > 0.5] = (1., 1., 0.)
                body_mesh = trimesh.Trimesh(
                    vertices=bodies[idx].detach().cpu().numpy(),
                    faces=interaction_model.mesh.faces,
                    vertex_colors=colors
                )
                # body_mesh.show()
                export_file = Path(args.save_dir, args.exp_name, batch['interaction'][0],
                                   base_name + '_' + str(idx) + '_body_' + args.model_name + '.png')
                export_file.parent.mkdir(exist_ok=True, parents=True)
                img_collage = render_body_multview(body=body_mesh)
                img_collage.save(str(export_file))
            else:
                body_mesh = skeleton_to_mesh(bodies[idx].detach().cpu().numpy(), np.array((0.8, 0.1, 0.1)))
            # print('add body mesh')
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = np.linalg.inv(batch['rotation'][idx].detach().cpu().numpy())
            transform[:3, 3] = batch['centroid'][idx].detach().cpu().numpy()
            # body_mesh += trimesh.creation.axis(transform=transform, origin_color=(1.0, 0.0, 0.0))

            if args.draw_attention:
                body_points = bodies[idx].detach().cpu().numpy()  # J x 3
                attention = attention_list[-1][idx, :, :]  # J x B+P*2
                # print( attention.sum(dim=1))
                object_centroids = []
                for obj_mesh in obj_meshes:
                    object_centroids.append(obj_mesh.vertices.mean(axis=0))
                lines = []
                obj_points = obj_points_coord[idx, :, :, :].reshape((-1, 3))

                for joint_idx in range(num_body_points):
                    values, indices = attention[joint_idx, num_body_points:].topk(5, largest=True)
                    for value, index in zip(values, indices):
                        point_idx = index.item()
                        attention_line = np.array([body_points[joint_idx], obj_points[point_idx]])
                        lines.append(trimesh.creation.cylinder(min(0.015, value.item()), segment=attention_line,
                                                                          vertex_colors=(0.8, 0.8, 0.1) if point_idx < num_obj_keypoints else (0.8, 0.1, 0.8)))

                obj_meshes = obj_meshes + lines

            # body_mesh.show()
            if args.visualize:
                export_file = Path(args.save_dir, args.exp_name, batch['interaction'][0],
                                   base_name + '_' + str(idx) + '_' + args.model_name + '.png')
                export_file.parent.mkdir(exist_ok=True, parents=True)
                # print('start render')
                # (body_mesh + trimesh.util.concatenate(obj_meshes)).show()
                img_collage = render_interaction_multview(body=body_mesh, static_scene=trimesh.util.concatenate(obj_meshes),
                                                          obj_points_coord=obj_points_coord[idx, :batch['num_atomics'][idx],
                                                                           :, :].reshape((-1, 3))
                                                          )
                # print('render finish')
                img_collage.save(str(export_file))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--interaction_checkpoint", type=str, default="/mnt/scratch/scene_graph/results/interaction/two_contact/version_1/checkpoints/last.ckpt")
    parser.add_argument("--save_dir", type=str, default="/local/home/zkf/scene_graph/render/interaction")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="default")
    parser.add_argument("--center_type", type=str, default="human")
    parser.add_argument("--full_scene", type=int, default=0)
    parser.add_argument("--sample_num", type=int, default=8)
    parser.add_argument("--data_overwrite", type=int, default=0)
    parser.add_argument("--composite_only", type=int, default=0)
    parser.add_argument("--composition_sample", type=str, default='no')
    parser.add_argument("--decode", type=int, default=0)
    parser.add_argument("--draw_attention", type=int, default=0)
    parser.add_argument("--visualize", type=int, default=1)

    parser.add_argument("--mask_thresh_by_vertex", type=float, default=-0.2)
    parser.add_argument("--mask_thresh_by_part", type=float, default=-10)
    args = parser.parse_args()

    device = torch.device('cuda')

    interaction_model = LitInteraction.load_from_checkpoint(args.interaction_checkpoint).to(device)
    # interaction_model.args.use_contact_feature = 0
    body_type = interaction_model.args.body_type
    batch_size = args.sample_num
    num_obj_keypoints = interaction_model.args.num_obj_keypoints
    num_body_points = interaction_model.args.num_body_points
    # print('mask_body', interaction_model.args.mask_body)

    torch.manual_seed(777)
    np.random.seed(777)
    data_loader = get_data_loader()
    z = torch.randn((args.sample_num, interaction_model.args.latent_dim), dtype=torch.float32, device=device) if args.decode else None
    with torch.no_grad():
        args.composition_sample = 'no'
        args.model_name = 'naive'
        render(z)
        args.composition_sample = 'learned_by_part'
        args.model_name = 'learned_by_part'
        render(z)




