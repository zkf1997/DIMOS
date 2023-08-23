import argparse
import copy
import os
import pickle
import random
from copy import deepcopy
import time

import numpy as np
import open3d as o3d
import smplx
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.multiprocessing import Pool

from pigraph_config import *
from data.load_interaction import get_interaction_segments
from interaction_graph import InteractionGraph
from prototypical_interaction_graph import PrototypicalInteractionGraph
from data.scene import Scene, scenes

# disable progress on cluster
from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

def error_handler(e):
    print('error')
    print(dir(e), "\n")
    print("-->{}<--".format(e.__cause__))

def test(scene, filename):
    print(scene.name, filename + '.png')

def skeleton_to_smpl(skeletons, device=torch.device('cuda')):
    # t1 = time.time()
    batch_size = len(skeletons)
    thetas = np.asarray([skeleton.relative_orientations for skeleton in skeletons])  # S*J*3
    translations = torch.from_numpy(np.asarray([skeleton.positions[0] for skeleton in skeletons]))  # S*3
    betas = torch.from_numpy(np.asarray([skeleton.shape for skeleton in skeletons]))
    # print(time.time()-t1)

    torch_param = {}
    torch_param['global_orient'] = torch.tensor(thetas[:, 0, :].reshape((batch_size, 3)),
                                                dtype=torch.float32).to(device)
    torch_param['body_pose'] = torch.tensor(thetas[:, 1:22, :].reshape((batch_size, 63)),
                                            dtype=torch.float32).to(device)
    if NUM_JOINTS == 55:
        torch_param['jaw_pose'] = torch.tensor(thetas[:, 22:23, :].reshape((batch_size, 3)),
                                               dtype=torch.float32).to(device)
        torch_param['leye_pose'] = torch.tensor(thetas[:, 23:24, :].reshape((batch_size, 3)),
                                                dtype=torch.float32).to(device)
        torch_param['reye_pose'] = torch.tensor(thetas[:, 24:25, :].reshape((batch_size, 3)),
                                                dtype=torch.float32).to(device)
        torch_param['left_hand_pose'] = torch.tensor(thetas[:, 25:40, :].reshape((batch_size, 45)),
                                                     dtype=torch.float32).to(device)
        torch_param['right_hand_pose'] = torch.tensor(thetas[:, 40:55, :].reshape((batch_size, 45)),
                                                      dtype=torch.float32).to(device)

    torch_param['betas'] = torch.tensor(betas, dtype=torch.float32).reshape((batch_size, 10)).to(device)
    torch_param['expression'] = torch.zeros([batch_size, body_model.num_expression_coeffs], dtype=torch.float32).to(device)

    # t2 = time.time()
    # print(t2 - t1)
    smplx_output = body_model(return_verts=True, return_fullpose=True, **torch_param)
    # print(time.time() - t2)
    # print(smplx_output.joints.detach().cpu().numpy()[0][0])

    transl = translations - smplx_output.joints.detach().cpu()[:, 0, :]
    torch_param['transl'] = torch.tensor(transl, dtype=torch.float32).to(device)
    return torch_param, smplx_output.vertices.detach().cpu() + transl.reshape((batch_size, 1, 3))

def query_sdf(scene, vertices, device=torch.device("cuda")):
    sdf_config = scene.sdf_config
    sdf_grids = torch.from_numpy(scene.sdf)
    sdf_grids = sdf_grids.to(device)
    sdf_grids = sdf_grids.unsqueeze(0)
    sdf_grids = sdf_grids.permute(0, 4, 1, 2, 3)  # 1*C*D*D*D
    grid_dim, grid_min, grid_max = sdf_config['grid_dim'], sdf_config['grid_min'], sdf_config['grid_max']
    normed_vertices = torch.tensor((vertices - grid_min) / (grid_max - grid_min) * 2 - 1, dtype=torch.float32,
                                   device=device)
    # print("to sample sdf")
    x = F.grid_sample(sdf_grids,
                      normed_vertices[:, :, [2, 1, 0]].view(1, vertices.shape[0], vertices.shape[1], 1, 3),
                      padding_mode='border', mode='bilinear', align_corners=True)  # 1*C*B*V*1
    # print("sdf sampled")
    x = x.permute(0, 2, 3, 4, 1)  # 1*B*V*1*C
    x = x.squeeze()  # B*V
    return x

def calc_penetration(new_skeletons, scene, batch_size=512):
    device = torch.device("cuda")  # cannot combine cuda with multiprocessing?
    num_skeletons = len(new_skeletons)
    penetration_scores = []
    for batch_left in range(0, num_skeletons, batch_size):
        batch_right = min(batch_left + batch_size, num_skeletons)
        _, vertices = skeleton_to_smpl(new_skeletons[batch_left:batch_right], device)
        # print("got vertices")
        x = query_sdf(scene, vertices)
        # print(x.shape)
        x[x > 0] = 0  # remove positive dists
        penetration_scores.append((x ** 2).sum(dim=1).cpu().numpy())  # penalize points inside scene meshes
    penetration_scores = np.concatenate(penetration_scores)
    return penetration_scores * args.penetration_weight

def synthesize(scene_name, interaction, pigraph, object_combination, filename, retarget=False, composition=False, pigraph2=None):
    start_time = time.time()
    scene = scenes[scene_name]
    # pigraph = pigraphs[pigraph_name] if not retarget else retargeted_pigraphs[pigraph_name]
    print("start synthesize for:", scene.name, ' ', filename)
    translations = scene.translation_sample_for_interaction(interaction=interaction,
                                                            object_combination=object_combination,
                                                            sample_method='pigraph',
                                                            num_samples=args.num_translations * 2)
    print('query sdf', time.time() - start_time)
    dists = query_sdf(scene, translations[np.newaxis, ...]).cpu().numpy().flatten()
    translations = translations[dists > -0.2]
    print("sampled valid transaltions:", translations.shape, time.time() - start_time)
    idx = np.random.choice(translations.shape[0], min(translations.shape[0], args.num_translations))
    translations = translations[idx]
    # rot_angles = torch.quasirandom.SobolEngine(dimension=1).draw(10).numpy().flatten() * 2 * np.pi
    rot_angles = np.linspace(0, 2 * np.pi, args.num_rotations)

    # init
    bestk_igraphs = []
    bestk_scores = []
    if args.debug:
        print('start sampling interactions')
    if composition:
        skeletons = pigraph.skeleton_distribution.sample(
            num_samples=args.num_skeletons * args.num_results * 10,
            topk=args.num_skeletons * args.num_results // 2,
            gender=args.gender) + \
                    pigraph2.skeleton_distribution.sample(
                        num_samples=args.num_skeletons * args.num_results * 10,
                        topk=args.num_skeletons * args.num_results // 2,
                        gender=args.gender)
    else:
        skeletons = pigraph.skeleton_distribution.sample(num_samples=args.num_skeletons * args.num_results * 10,
                                                         topk=args.num_skeletons * args.num_results,
                                                         gender=args.gender)  # number of skeletons: beamSize = 10 https://github.com/msavva/pigraphs/blob/ee794f3acef4eac418ca0f69bb410fef34b99246/bin/parameters_default.txt#L247

    skeletons = random.sample(skeletons, k=len(skeletons))
    print('get base skeletons', time.time() - start_time)

    for result_idx in tqdm(range(args.num_results)):
        new_skeletons = []
        for skeleton in skeletons[result_idx * args.num_skeletons: (result_idx+1) * args.num_skeletons]:
            for translation in translations:
                for rot_angle in rot_angles:
                    transform = np.zeros((4, 4))
                    transform[:3, 3] = translation
                    transform[:2, :2] = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                                                  [np.sin(rot_angle), np.cos(rot_angle)]])
                    transform[2, 2] = 1.0
                    transform[3, 3] = 1.0
                    # rot1 = Rotation.from_rotvec(new_skeleton.orientations[0])
                    # rot2 = Rotation.from_matrix(transform[:3, :3])
                    # rot = rot2 * rot1.inv()
                    # transform[:3, :3] = rot.as_matrix()

                    new_skeleton = copy.deepcopy(skeleton)
                    new_skeleton.positions -= new_skeleton.positions[0]
                    new_skeleton.transform(transform)
                    new_skeletons.append(new_skeleton)

        # build igraph cost most time
        igraphs = [InteractionGraph(scene, new_skeleton) for new_skeleton in new_skeletons]
        # print('built igraphs')
        # for igraph in igraphs:
        #     igraph.visualize(use_smplx=True)
        if composition:
            scores = np.asarray([(pigraph.similarity(igraph) + pigraph2.similarity(igraph)) * 0.5 for igraph in igraphs]).astype(np.float64)
        else:
            scores = np.asarray([pigraph.similarity(igraph) for igraph in igraphs]).astype(np.float64)
        # print('got similarity')
        if args.use_penetration:
            # print("calc penetration")
            penetration_scores = calc_penetration(new_skeletons, scene).astype(np.float64)
            # print("got penetration")
            scores -= penetration_scores
        else:
            penetration_scores = np.zeros_like(scores)

        best_idx = np.argmax(scores)
        bestk_igraphs.append(igraphs[best_idx])
        bestk_scores.append(scores[best_idx])

    # pelvis_positions = [igraph.skeleton.positions[0] for igraph in bestk_igraphs]
    # full_poses = [igraph.skeleton.relative_orientations.flatten() for igraph in bestk_igraphs]
    # results = {
    #     'pelvis_positions': np.asarray(pelvis_positions),
    #     'thetas': np.asarray(full_poses)
    # }
    print('skeleton to smpl', time.time() - start_time)
    results, _ = skeleton_to_smpl([igraph.skeleton for igraph in bestk_igraphs])
    with open(filename + '.pkl', 'wb') as result_file:
        pickle.dump(results, result_file, protocol=2)
    if args.visualize:
        bestk_igraphs[0].save(filename + '.png', vis=vis)
        bestk_igraphs[0].save(filename + '_smpl.png', use_smplx=True, vis=vis)
    print("finish synthesize for:", scene.name, ' ', filename)
    return bestk_igraphs, bestk_scores

def build_pigraph(interaction, interaction_dataset):
    print('build for:', interaction)
    interaction_data = get_interaction_segments(interaction.split('+'), interaction_dataset)
    if len(interaction_data) == 0:
        print('no data for:', interaction)
        return None
    return PrototypicalInteractionGraph(interaction=interaction, interaction_data=interaction_data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--num_process', type=int, default=4)
    parser.add_argument('--use_penetration', type=int, default=1)
    parser.add_argument('--penetration_weight', type=float, default=10.0)
    parser.add_argument('--num_translations', type=int, default=512)
    parser.add_argument('--num_skeletons', type=int, default=64)
    parser.add_argument('--num_rotations', type=int, default=12)
    parser.add_argument('--num_joints', type=int, default=55)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--scene_name', type=str, default='all')
    parser.add_argument('--interaction', type=str, default='all')
    parser.add_argument('--object_combination', type=str, default='')
    parser.add_argument('--composition', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='pigraph')
    parser.add_argument('--num_results', help='number of synthesis results for pair of scene and pigraph',
                        type=int, default=100)
    args = parser.parse_args()
    # NUM_JOINTS = args.num_joints

    if args.visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().line_width = 50
        vis.clear_geometries()

    body_model = smplx.create(smplx_model_folder, model_type='smplx',
                              gender=args.gender, ext='npz', use_pca=False,
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
                              # batch_size=args.num_skeletons * args.num_rotations
                              ).to(torch.device("cuda"))

    pigraphs = {}
    retargeted_pigraphs = {}
    print("building pigraphs")
    # load data
    with open(Path.joinpath(project_folder, "data", 'train.pkl'), 'rb') as data_file:
        train_data = pickle.load(data_file)
    # set interactions to fit pigraph
    # interactions = []
    # interactions += ['sit on-chair', 'sit on-sofa', 'sit on-bed', 'lie on-sofa', 'lie on-bed',
    #                 'stand on-bed', 'stand on-table', 'stand on-floor']
    # interactions += ['sit on-chair+touch-table']
    # interactions += ['stand on-floor+touch-table']
    # interactions += interaction_names

    # multi process to accelerate
    # for interaction in interactions:
    #     print(interaction)
    #     interaction_data = get_interaction_segments(interaction.split('+'), train_data)
    #     if len(interaction_data) == 0:
    #         print('no data for:', interaction)
    #         continue
    #     pigraphs[interaction] = PrototypicalInteractionGraph(interaction=interaction, interaction_data=interaction_data)
        # pigraphs[interaction] = PrototypicalInteractionGraph(interaction=interaction, interaction_data=interaction_data)
        # pigraphs[interaction].igraphs[0].log()
        # pigraphs[interaction].igraphs[0].visualize()
        # break

    # print('sample skeleton distributions')
    # if not os.path.exists(os.path.join(rendering_folder, 'skeleton')):
    #     os.makedirs(os.path.join(rendering_folder, 'skeleton'))
    # for pigraph_name in tqdm(pigraphs):
    #     pigraph = pigraphs[pigraph_name]
    #     t1 = time.time()
    #     skeletons = pigraph.skeleton_distribution.sample(num_samples=1000, topk=10)
    #     print(time.time() - t1)
    #     for idx, skeleton in enumerate(skeletons):
    #         skeleton.save(filename=os.path.join(rendering_folder, "skeleton", str(pigraph_name) + str(idx) + '.png'), use_smplx=False)
    #         skeleton.save(filename=os.path.join(rendering_folder, "skeleton", str(pigraph_name) + str(idx) + '_smplx.png'), use_smplx=True)

    if args.composition:
        print("synthesize using composed pigraphs")
        composed_pigraphs = {}
        used_scene_names = test_scenes if args.scene_name == 'all' else [args.scene_name]
        for scene_name in used_scene_names:
            scene = scenes[scene_name]
            used_interaction = composed_interaction_names if args.interaction == 'all' else [args.interaction]
            for interaction in tqdm(used_interaction):
                if scene.support_interaction(interaction):
                    atomic_interactions = interaction.split('+')
                    if interaction not in composed_pigraphs:
                        for atomic_interaction in atomic_interactions:
                            if atomic_interaction not in pigraphs:
                                pigraphs[atomic_interaction] = build_pigraph(atomic_interaction, train_data)
                        # composed_pigraphs[interaction] = PrototypicalInteractionGraph.compose(interaction,
                        #                                                                       pigraphs[atomic_interactions[0]],
                        #                                                                       pigraphs[atomic_interactions[1]])

                    verbs, nouns, candidate_combinations = scene.get_interaction_candidate_objects(interaction)
                    for object_combination in tqdm(candidate_combinations):
                        output_dir = os.path.join(results_folder, args.save_dir)
                        output_dir = os.path.join(output_dir, interaction, scene_name)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        atomic_names = []
                        for atomic_idx, noun in enumerate(nouns):
                            atomic_names.append(
                                verbs[atomic_idx] + '-' + nouns[atomic_idx] + '-' + str(object_combination[atomic_idx].id))
                        base_name = '+'.join(atomic_names)
                        # synthesize(scene_name, composed_pigraphs[interaction], object_combination,
                        #            os.path.join(output_dir, base_name))
                        synthesize(scene_name, interaction, pigraphs[atomic_interactions[0]], object_combination,
                                   os.path.join(output_dir, base_name), composition=True, pigraph2=pigraphs[atomic_interactions[1]])
    else:
        print("synthesize in given scenes")
        # for pigraph_name in tqdm(interaction_names):
        used_scene_names = test_scenes if args.scene_name == 'all' else [args.scene_name]
        for scene_name in used_scene_names:
            scene = scenes[scene_name]
            used_interaction = interaction_names if args.interaction == 'all' else [args.interaction]
            for interaction in tqdm(used_interaction):
                if scene.support_interaction(interaction):
                    if interaction not in pigraphs:
                        pigraphs[interaction] = build_pigraph(interaction, train_data)

                    verbs, nouns, candidate_combinations = scene.get_interaction_candidate_objects(interaction)
                    for object_combination in tqdm(candidate_combinations):
                        output_dir = os.path.join(results_folder, args.save_dir)
                        output_dir = os.path.join(output_dir, interaction, scene_name)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        atomic_names = []
                        for atomic_idx, noun in enumerate(nouns):
                            atomic_names.append(verbs[atomic_idx] + '-' + nouns[atomic_idx] + '-' + str(
                                object_combination[atomic_idx].id))
                        base_name = '+'.join(atomic_names)
                        synthesize(scene_name, interaction, pigraphs[interaction], object_combination,
                                   os.path.join(output_dir, base_name))
            #     break
            # break

    # print("retarget pigraphs")
    # retargeted_pigraphs['sit-chair'] = pigraphs['sit-bed'].retarget('sit-chair')
    # retargeted_pigraphs['stand-chair'] = pigraphs['stand-table'].retarget('stand-chair')
    # retargeted_pigraphs['lie-bed'] = pigraphs['lie-sofa'].retarget('lie-bed')
    # print("retarget synthesize in all scenes")
    # if not os.path.exists(os.path.join(rendering_folder, 'retarget')):
    #     os.makedirs(os.path.join(rendering_folder, 'retarget'))
    # for pigraph_name in tqdm(retargeted_pigraphs):
    #     print(pigraph_name)
    #     for scene_name in tqdm(scenes):
    #         # print(scene_name)
    #         # p.apply_async(synthesize, args=(scene_name, pigraph_name,
    #         #                                 os.path.join(rendering_folder, 'retarget', scene_name + '_' + pigraph_name), True),
    #         #               # error_callback=error_handler
    #         #               )
    #         synthesize(scene_name, pigraph_name,
    #                    os.path.join(rendering_folder, 'retarget', scene_name + '_' + pigraph_name), True)

    # p.close()
    # p.join()
    print("synthesize all finished")