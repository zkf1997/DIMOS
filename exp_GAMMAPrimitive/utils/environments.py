import os, sys
import pickle
import random

import numpy as np
import pytorch3d.transforms
import torch
import trimesh

sys.path.append(os.getcwd())

from models.baseops import SMPLXParser
from exp_GAMMAPrimitive.utils.batch_gen_amass import *

rest_pose = torch.cuda.FloatTensor(
[0.0, 0.0, 0.0, -0.011472027748823166, 1.2924634671010859e-26, 2.5473026963570487e-18, -0.0456559844315052, -0.0019564421381801367, -0.08563289791345596, 0.11526273936033249, 0.0, -2.5593469423841883e-17, 0.06192377582192421, -1.2932950836510723e-26, -1.3749840337845367e-17, 0.07195857912302017, 0.00617849500849843, 1.4564738304301272e-11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.09871290624141693, 6.4515848478496824e-18, 7.851343602724672e-17, -0.008369642309844494, -0.12677378952503204, -0.3995564579963684, 0.0013758527347818017, 0.01013219729065895, 0.23814785480499268, 0.277565598487854, -1.5771439149242302e-17, -6.061879960787066e-17, -0.10060133039951324, 0.1710081696510315, -0.8297445774078369, 0.016900330781936646, -0.03264763951301575, 0.9994331002235413, -0.11047029495239258, -0.4468419551849365, -0.17531509697437286, -0.15802216529846191, 0.4728464186191559, 0.023101171478629112, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11167845129966736, 0.04289207234978676, -0.41644084453582764, 0.10881128907203674, -0.06598565727472305, -0.756219744682312, -0.0963931530714035, -0.09091583639383316, -0.18845966458320618, -0.11809506267309189, 0.050943851470947266, -0.5295845866203308, -0.14369848370552063, 0.055241718888282776, -0.704857349395752, -0.019182899966835976, -0.0923367589712143, -0.3379131853580475, -0.45703303813934326, -0.1962839663028717, -0.6254575848579407, -0.21465237438678741, -0.06599827855825424, -0.5068942308425903, -0.36972442269325256, -0.0603446289896965, -0.07949023693799973, -0.14186954498291016, -0.08585254102945328, -0.6355276107788086, -0.3033415675163269, -0.05788097903132439, -0.6313892006874084, -0.17612087726593018, -0.13209305703639984, -0.3733545243740082, 0.850964367389679, 0.2769227623939514, -0.09154807031154633, -0.4998386800289154, 0.026556432247161865, 0.052880801260471344, 0.5355585217475891, 0.045960985124111176, -0.27735769748687744, 0.11167845129966736, -0.04289207234978676, 0.41644084453582764, 0.10881128907203674, 0.06598565727472305, 0.756219744682312, -0.0963931530714035, 0.09091583639383316, 0.18845966458320618, -0.11809506267309189, -0.050943851470947266, 0.5295845866203308, -0.14369848370552063, -0.055241718888282776, 0.704857349395752, -0.019182899966835976, 0.0923367589712143, 0.3379131853580475, -0.45703303813934326, 0.1962839663028717, 0.6254575848579407, -0.21465237438678741, 0.06599827855825424, 0.5068942308425903, -0.36972442269325256, 0.0603446289896965, 0.07949023693799973, -0.14186954498291016, 0.08585254102945328, 0.6355276107788086, -0.3033415675163269, 0.05788097903132439, 0.6313892006874084, -0.17612087726593018, 0.13209305703639984, 0.3733545243740082, 0.850964367389679, -0.2769227623939514, 0.09154807031154633, -0.4998386800289154, -0.026556432247161865, -0.052880801260471344, 0.5355585217475891, -0.045960985124111176, 0.27735769748687744]
)[3:66].reshape(1, 63)

class BatchGeneratorSceneTrain(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67',  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 scene_list=None,
                 scene_dir=None,
                 scene_type='random',
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        self.scene_list= scene_list
        self.scene_dir = Path(scene_dir)
        self.scene_idx = 0
        self.scene_type=scene_type
        # with open(os.path.join(dataset_path, 'orient.json')) as f:
        #     self.orient = np.array(json.load(f)).reshape(1, 3)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=True,
                  scene_idx=None, start_target=None, random_rotation_range=1.0,
                  clip_far=False,
                  res=32, extent=1.6):
        if scene_idx is None:
            scene_idx = torch.randint(len(self.scene_list), size=(1,)).item()
        scene_name = self.scene_list[scene_idx]
        if self.scene_type == 'prox':
            mesh_path = self.scene_dir / 'PROX' / (scene_name + '_floor.ply')
            navmesh_path = self.scene_dir / 'PROX' / (scene_name + '_navmesh.ply')
        elif self.scene_type == 'room_0':
            mesh_path = self.scene_dir / 'mesh_floor.ply'
            navmesh_path = self.scene_dir / 'navmesh_tight.ply'
            samples_path = self.scene_dir / 'samples.pkl'
        elif 'random' in self.scene_type:
            mesh_path = self.scene_dir / self.scene_type / (scene_name + '.ply')
            navmesh_path = self.scene_dir / self.scene_type / (scene_name + '_navmesh_tight.ply')
            samples_path = self.scene_dir / self.scene_type / (scene_name + '_samples.pkl')
        navmesh = trimesh.load(navmesh_path, force='mesh')
        navmesh.vertices[:, 2] = 0
        navmesh.visual.vertex_colors = np.array([0, 0, 200, 50])
        navmesh_torch = pytorch3d.structures.Meshes(
            verts=[torch.cuda.FloatTensor(navmesh.vertices)],
            faces=[torch.cuda.LongTensor(navmesh.faces)]
        )

        # import pyrender
        # scene = pyrender.Scene()
        # scene.add(pyrender.Mesh.from_trimesh(obj_mesh, smooth=False))
        # # scene.add(pyrender.Mesh.from_trimesh(navmesh, smooth=False))
        # scene.add(pyrender.Mesh.from_trimesh(navmesh_crop, smooth=False))
        # pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        """randomly specify a 3D path"""
        wpath = np.zeros((3,3))
        wpath = torch.cuda.FloatTensor(wpath) #starting point, ending point, another point to initialize the body orientation
        if start_target is not None:
            wpath[0] = torch.cuda.FloatTensor(start_target[0])  # starting point
            wpath[1] = torch.cuda.FloatTensor(start_target[1])  # ending point xy
        elif self.scene_type == 'prox':
            start_target = np.zeros((2, 3))  # pairs of start and target positions
            max_try = 32
            for try_idx in range(max_try):
                start_target[0] = trimesh.sample.sample_surface_even(navmesh, 1)[0]
                if try_idx < max_try - 1:
                    crop_box = trimesh.creation.box(extents=[sigma, sigma, 2])
                    crop_box.vertices += start_target[0]
                    navmesh_crop = navmesh.slice_plane(crop_box.facets_origin, -crop_box.facets_normal)
                    if len(navmesh_crop.vertices) >= 3:
                        start_target[1] = trimesh.sample.sample_surface_even(navmesh_crop, 1)[0]
                        break
                else:
                    start_target[1] = trimesh.sample.sample_surface_even(navmesh, 1)[0]

            if np.isnan(start_target).any():
                print('error in sampling start-target')
            wpath[0] = torch.cuda.FloatTensor(start_target[0]) #starting point
            wpath[1] = torch.cuda.FloatTensor(start_target[1])  # ending point xy
            if torch.isnan(wpath).any() or torch.isinf(wpath).any():
                print('error:wpath invalid, random sample', wpath)
        elif 'random' in self.scene_type or self.scene_type == 'room_0':
            with open(samples_path, 'rb') as f:
                sample_pairs = pickle.load(f)
            num_samples = len(sample_pairs)
            if num_samples == 0:
                print('error: zero samples, precompute')
            start, target = sample_pairs[np.random.randint(low=0, high=num_samples)]
            if clip_far and np.linalg.norm(target - start) > 1.0:
                # print('clip far pairs')
                length = np.linalg.norm(target - start).clip(min=1e-12)
                vec_dir = (target - start) / length
                l1 = np.random.uniform(low=0.0, high=length - 0.5)
                l2 = min(np.random.uniform(0.5, 1.0) + l1, length)
                target = start + vec_dir * l2
                start = start + vec_dir * l1
            wpath[0] = torch.cuda.FloatTensor(start)  # starting point
            wpath[1] = torch.cuda.FloatTensor(target)  # ending point xy
            if torch.isnan(wpath).any() or torch.isinf(wpath).any():
                print('error:wpath invalid, precompute', wpath)

        # wpath[2, :2] = wpath[0, :2] + torch.randn(2).to(device=wpath.device) #point to initialize the body orientation, not returned
        theta = torch.pi * (2 * torch.cuda.FloatTensor(1).uniform_() - 1) * random_rotation_range
        random_rotz = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]).reshape(1, 3),
                                                                  convention="XYZ")
        wpath[2] = torch.einsum('ij, j->i', random_rotz[0], wpath[1] - wpath[0]) + wpath[0]  # face the target with [-90, 90] disturbance
        if torch.norm(wpath[2] - wpath[0], dim=-1) < 1e-12:
            wpath[2] += 1e-12
        # hard code
        # wpath[0] = torch.cuda.FloatTensor([-1, 0, 0])
        # wpath[1] = torch.cuda.FloatTensor([0.5, 0, 0])
        # wpath[2] = wpath[1]

        """generate a body"""
        xbo_dict = {}
        # gender = random.choice(['male', 'female'])
        gender = random.choice(['male'])
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None,...]

        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :]  # [1,3]
        xbo_dict = self.snap_to_ground(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        wpath[0] = bm(**xbo_dict).joints[0, 0, :]
        wpath[1, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        if torch.isnan(xbo_dict['wpath']).any() or torch.isinf(xbo_dict['wpath']).any():
            print('error:wpath invalid', xbo_dict['wpath'])
        xbo_dict['scene_path'] = mesh_path
        xbo_dict['navmesh'] = navmesh
        xbo_dict['navmesh_torch'] = navmesh_torch
        xbo_dict['navmesh_path'] = navmesh_path
        xbo_dict['floor_height'] = 0
        self.index_rec += 1

        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            # floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
            #                                   transform=np.array([[1.0, 0.0, 0.0, 0],
            #                                                       [0.0, 1.0, 0.0, 0],
            #                                                       [0.0, 0.0, 1.0, -0.005],
            #                                                       [0.0, 0.0, 0.0, 1.0],
            #                                                       ]),
            #                                   )
            # floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            obj_mesh = trimesh.load(mesh_path, force='mesh')
            obj_mesh.vertices[:, 2] -= 0.02
            vis_mesh = [
                # floor_mesh,
                        init_body_mesh,
                        obj_mesh, navmesh,
                        trimesh.creation.axis()
                        ]

            for point_idx, pelvis in enumerate(wpath[:2, :].reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)

            # visualize map
            joints = bm(**xbo_dict).joints  # [b,p,3]
            x_axis = joints[:, 2, :] - joints[:, 1, :]
            x_axis[:, -1] = 0
            x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
            z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
            y_axis = torch.cross(z_axis, x_axis)
            gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
            gamma_transl = joints[0, 0, :].reshape(1, 1, 3)
            points_local, points, map = get_map(navmesh, gamma_orient, gamma_transl,
                                    res=res, extent=extent,
                                  return_type='numpy')
            # _, points, map = get_map_trimesh(navmesh, navmesh_query, gamma_orient, gamma_transl,
            #                          res=32, extent=1.6,
            #                          return_type='numpy')
            points = points[0]  # [p, 3]
            map = map[0]  #[p]
            cells = []
            import exp_GAMMAPrimitive.utils.config_env as config_env
            with open(config_env.get_body_marker_path() + '/SSM2.json') as f:
                marker_ssm_67 = json.load(f)['markersets'][0]['indices']
            body_marker_idx = list(marker_ssm_67.values())
            print(body_marker_idx)
            feet_markers = ['RHEE', 'RTOE', 'RRSTBEEF', 'LHEE', 'LTOE', 'LRSTBEEF']
            feet_marker_idx = [list(marker_ssm_67.keys()).index(marker_name) for marker_name in feet_markers]
            pene_type = 'foot'
            Y_w = bm(**xbo_dict).vertices[:, body_marker_idx, :]
            Y_l = torch.einsum('bij,bpj->bpi', gamma_orient.permute(0, 2, 1), Y_w - gamma_transl)  # [b, p, 3]
            Y_l = Y_l[:, None, :, :]
            if pene_type == 'foot':
                markers_local_xy = Y_l[:, :, feet_marker_idx, :2]  # [b, t, p, 2]
            elif pene_type == 'body':
                markers_local_xy = Y_l[:, :, :, :2]  # [b, t, p, 2]
            # get body bbox on xy plane
            box_min = markers_local_xy.amin(dim=[1, 2]).reshape(1, 1, 2)
            box_max = markers_local_xy.amax(dim=[1, 2]).reshape(1, 1, 2)
            points_local = torch.cuda.FloatTensor(points_local)
            inside_box = ((points_local[:, :, :2] >= box_min).all(-1) & (points_local[:, :, :2] <= box_max).all(
                -1)).float()
            num_pene = inside_box[0] * (1 - torch.cuda.FloatTensor(map)) * 0.5  # [D]
            num_pene = num_pene.detach().cpu().numpy()
            for point_idx in range(points.shape[0]):
                color = np.array([0, 0, 200, 100]) if map[point_idx] else np.array([200, 0, 0, 100])
                if num_pene[point_idx] != 0:
                    color = np.array([0, 0, 0, 200])
                transform = np.eye(4)
                transform[:3, 3] = points[point_idx]
                cell = trimesh.creation.box(extents=(0.05, 0.05, 1), vertex_colors=color, transform=transform)
                # cell = trimesh.creation.cylinder(radius=0.02,
                #                           segment=np.stack([points[point_idx], points[point_idx] + np.array([0, 0, 0.5])], axis=0),
                #                           vertex_colors=color)
                cells.append(cell)
            vis_mesh.append(trimesh.util.concatenate(cells))

            print(xbo_dict['wpath'])
            import pyrender
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        return xbo_dict

class BatchGeneratorSceneRandomTest(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67',  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 scene_list=None,
                 scene_dir=None,
                 scene_type='random',
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        self.scene_list= scene_list
        self.scene_dir = Path(scene_dir)
        self.scene_idx = 0
        self.scene_type=scene_type
        # with open(os.path.join(dataset_path, 'orient.json')) as f:
        #     self.orient = np.array(json.load(f)).reshape(1, 3)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=True,
                  scene_idx=None, wpath=None, path_idx=None,
                  clip_far=False,
                  res=32, extent=1.6):
        if scene_idx is None:
            scene_idx = torch.randint(len(self.scene_list), size=(1,)).item()
        scene_name = self.scene_list[scene_idx]
        if self.scene_type == 'prox':
            mesh_path = self.scene_dir / 'PROX' / (scene_name + '_floor.ply')
            navmesh_path = self.scene_dir / 'PROX' / (scene_name + '_navmesh.ply')
        elif self.scene_type == 'random':
            mesh_path = self.scene_dir / 'random_scene' / (scene_name + '.ply')
            navmesh_path = self.scene_dir / 'random_scene' / (scene_name + '_navmesh_tight.ply')
            samples_path = self.scene_dir / 'random_scene' / (scene_name + '_samples.pkl')
        elif self.scene_type == 'random_obstacle':
            mesh_path = self.scene_dir / 'random_scene_obstacle' / (scene_name + '.ply')
            navmesh_path = self.scene_dir / 'random_scene_obstacle' / (scene_name + '_navmesh_tight.ply')
            samples_path = self.scene_dir / 'random_scene_obstacle' / (scene_name + '_samples.pkl')
        navmesh = trimesh.load(navmesh_path, force='mesh')
        navmesh.vertices[:, 2] = 0
        navmesh.visual.vertex_colors = np.array([0, 0, 200, 200])
        navmesh_torch = pytorch3d.structures.Meshes(
            verts=[torch.cuda.FloatTensor(navmesh.vertices)],
            faces=[torch.cuda.LongTensor(navmesh.faces)]
        )

        # import pyrender
        # scene = pyrender.Scene()
        # scene.add(pyrender.Mesh.from_trimesh(obj_mesh, smooth=False))
        # # scene.add(pyrender.Mesh.from_trimesh(navmesh, smooth=False))
        # scene.add(pyrender.Mesh.from_trimesh(navmesh_crop, smooth=False))
        # pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        """randomly specify a 3D path"""
        if wpath is not None:
            wpath = torch.cuda.FloatTensor(wpath) #starting point, ending point, another point to initialize the body orientation
        elif self.scene_type in ['random']:
            with open(samples_path, 'rb') as f:
                sample_pairs = pickle.load(f)
            print(sample_pairs)
            paths = []
            last_point = None
            path = []
            for sample in sample_pairs:
                if last_point is not None and not np.array_equal(last_point, sample[0]):
                    paths.append(path)
                    path = [sample[0]]
                elif last_point is None:
                    path = [sample[0]]
                last_point = sample[1]
                path.append(last_point)
            if last_point is not None:
                paths.append(path)
            print('#path:', len(paths))
            path_idx = random.choice(range(len(paths))) if path_idx is None else path_idx
            path_name = 'path' + str(path_idx)
            print(paths[path_idx])
            wpath = np.stack(paths[path_idx], axis=0)
            wpath = torch.cuda.FloatTensor(wpath)
            print(wpath.shape)

        """generate a body"""
        xbo_dict = {}
        # gender = random.choice(['male', 'female'])
        gender = random.choice(['male'])
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[1])[None,...]

        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :]  # [1,3]
        xbo_dict = self.snap_to_ground(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        wpath[0] = bm(**xbo_dict).joints[0, 0, :]
        wpath[1:, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath
        if torch.isnan(xbo_dict['wpath']).any() or torch.isinf(xbo_dict['wpath']).any():
            print('error:wpath invalid', xbo_dict['wpath'])
        xbo_dict['scene_path'] = mesh_path
        xbo_dict['navmesh'] = navmesh
        xbo_dict['navmesh_torch'] = navmesh_torch
        xbo_dict['navmesh_path'] = navmesh_path
        xbo_dict['path_name'] = path_name
        xbo_dict['floor_height'] = 0
        self.index_rec += 1

        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            # floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
            #                                   transform=np.array([[1.0, 0.0, 0.0, 0],
            #                                                       [0.0, 1.0, 0.0, 0],
            #                                                       [0.0, 0.0, 1.0, -0.005],
            #                                                       [0.0, 0.0, 0.0, 1.0],
            #                                                       ]),
            #                                   )
            # floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            obj_mesh = trimesh.load(mesh_path, force='mesh')
            obj_mesh.vertices[:, 2] -= 0.2
            vis_mesh = [
                # floor_mesh,
                        init_body_mesh,
                        obj_mesh, navmesh,
                        trimesh.creation.axis()
                        ]

            for point_idx, pelvis in enumerate(wpath[:, :].reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)

            # visualize map
            joints = bm(**xbo_dict).joints  # [b,p,3]
            x_axis = joints[:, 2, :] - joints[:, 1, :]
            x_axis[:, -1] = 0
            x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
            z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
            y_axis = torch.cross(z_axis, x_axis)
            gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
            gamma_transl = joints[0, 0, :].reshape(1, 1, 3)
            _, points, map = get_map(navmesh, gamma_orient, gamma_transl,
                                    res=res, extent=extent,
                                  return_type='numpy')
            # _, points, map = get_map_trimesh(navmesh, navmesh_query, gamma_orient, gamma_transl,
            #                          res=32, extent=1.6,
            #                          return_type='numpy')
            points = points[0]  # [p, 3]
            map = map[0]  #[p]
            cells = []
            for point_idx in range(points.shape[0]):
                color = np.array([0, 0, 200, 100]) if map[point_idx] else np.array([200, 0, 0, 100])
                transform = np.eye(4)
                transform[:3, 3] = points[point_idx]
                cell = trimesh.creation.box(extents=(0.05, 0.05, 1), vertex_colors=color, transform=transform)
                # cell = trimesh.creation.cylinder(radius=0.02,
                #                           segment=np.stack([points[point_idx], points[point_idx] + np.array([0, 0, 0.5])], axis=0),
                #                           vertex_colors=color)
                cells.append(cell)
            vis_mesh.append(trimesh.util.concatenate(cells))

            print(xbo_dict['wpath'])
            import pyrender
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        return xbo_dict

class BatchGeneratorSceneTest(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67',  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)

    def interpolate_path(self, wpath, max_len=0.5):
        interpolated_path = [wpath[0]]
        last_point = wpath[0]
        for point_idx in range(1, wpath.shape[0]):
            while torch.norm(wpath[point_idx] - last_point) > max_len:
                last_point = last_point + (wpath[point_idx] - last_point) / torch.norm(wpath[point_idx] - last_point) * max_len
                interpolated_path.append(last_point)
            last_point = wpath[point_idx]
            interpolated_path.append(last_point)
        return torch.stack(interpolated_path, dim=0)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=True, use_zero_shape=True,
                  scene_path=None, floor_height=0, navmesh_path=None,
                  wpath_path=None, path_name=None,
                  last_motion_path=None,
                  clip_far=False, random_orient=False,
                  res=32, extent=1.6):

        """get navmesh"""
        if navmesh_path.exists():
            navmesh = trimesh.load(navmesh_path, force='mesh')
        else:
            from test_navmesh import create_navmesh, zup_to_shapenet
            scene_mesh = trimesh.load(scene_path, force='mesh')
            """assume the scene coords are z-up"""
            scene_mesh.vertices[:, 2] -= floor_height
            scene_mesh.apply_transform(zup_to_shapenet)
            navmesh = create_navmesh(scene_mesh, export_path=navmesh_path, agent_radius=0.01, visualize=False)
        navmesh.vertices[:, 2] = 0
        navmesh.visual.vertex_colors = np.array([0, 0, 200, 50])
        navmesh_torch = pytorch3d.structures.Meshes(
            verts=[torch.cuda.FloatTensor(navmesh.vertices)],
            faces=[torch.cuda.LongTensor(navmesh.faces)]
        )

        """get wpath"""
        with open(wpath_path, 'rb') as f:
            wpath = pickle.load(f)  # [n, 3]
        wpath = torch.cuda.FloatTensor(wpath)
        if clip_far:
            wpath = self.interpolate_path(wpath)

        """load or generate a body"""
        xbo_dict = {}
        if last_motion_path is not None:
            with open(last_motion_path, 'rb') as f:
                motion_data = pickle.load(f)  # [n, 3]
            last_primitive = motion_data['motion'][-1]
            gender = last_primitive['gender']
            betas = xbo_dict['betas'] = torch.cuda.FloatTensor(last_primitive['betas']).reshape((1, 10))
            smplx_params = torch.cuda.FloatTensor(last_primitive['smplx_params'][0, -1:])
            R0 = torch.cuda.FloatTensor(last_primitive['transf_rotmat'])
            T0 = torch.cuda.FloatTensor(last_primitive['transf_transl'])
            from models.baseops import SMPLXParser
            pconfig_1frame = {
                'n_batch': 1,
                'device': 'cuda',
                'marker_placement': 'ssm2_67'
            }
            smplxparser_1frame = SMPLXParser(pconfig_1frame)
            smplx_params = smplxparser_1frame.update_transl_glorot(R0.permute(0, 2, 1),
                                                                   -torch.einsum('bij,bkj->bki', R0.permute(0, 2, 1),
                                                                                 T0),
                                                                   betas=betas,
                                                                   gender=gender,
                                                                   xb=smplx_params,
                                                                   inplace=False,
                                                                   to_numpy=False)  # T0 must be [1, 1, 3], [1,3] leads to error
            xbo_dict['transl'] = smplx_params[:, :3]
            xbo_dict['global_orient'] = smplx_params[:, 3:6]
            xbo_dict['body_pose'] = smplx_params[:, 6:69]
            bm = self.bm_male if gender == 'male' else self.bm_female
        else:
            # gender = random.choice(['male', 'female'])
            gender = random.choice(['male'])
            xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_() if use_zero_shape else torch.cuda.FloatTensor(1,10).normal_()
            """maunal rest pose"""
            # body_pose = torch.zeros(1, 63).to(dtype=torch.float32)
            # body_pose[:, 45:48] = -torch.tensor([0, 0, 1]) * torch.pi * 0.45
            # body_pose[:, 48:51] = torch.tensor([0, 0, 1]) * torch.pi * 0.45
            # xbo_dict['body_pose'] = body_pose.to(device='cuda')
            xbo_dict['body_pose'] = torch.cuda.FloatTensor(rest_pose) if use_zero_pose else self.vposer.decode(torch.cuda.FloatTensor(1,32).normal_(), output_type='aa').view(1, -1)
            if random_orient:
                target = wpath[0] + torch.cuda.FloatTensor(3).normal_()
                target[2] = wpath[0, 2]
                # xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], target)[None,...]
                xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[0] + torch.cuda.FloatTensor([-1, 0, 0]))[None, ...]
            else:
                xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[1])[None, ...]
            """snap to the ground"""
            bm = self.bm_male if gender == 'male' else self.bm_female
            xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :]  # [1,3]
            xbo_dict = self.snap_to_ground(xbo_dict, bm) # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        init_body = bm(**xbo_dict)
        wpath[0] = init_body.joints[0, 0]
        start_markers = init_body.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]
        wpath[1:, 2] = wpath[0, 2]

        """specify output"""
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath
        if torch.isnan(xbo_dict['wpath']).any() or torch.isinf(xbo_dict['wpath']).any():
            print('error:wpath invalid', xbo_dict['wpath'])
        xbo_dict['scene_path'] = scene_path
        xbo_dict['navmesh'] = navmesh
        xbo_dict['navmesh_torch'] = navmesh_torch
        xbo_dict['navmesh_path'] = navmesh_path
        xbo_dict['path_name'] = path_name
        xbo_dict['floor_height'] = floor_height
        xbo_dict['motion_history'] = motion_data if last_motion_path is not None else None

        if visualize:
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                              transform=np.array([[1.0, 0.0, 0.0, 0],
                                                                  [0.0, 1.0, 0.0, 0],
                                                                  [0.0, 0.0, 1.0, -0.0051],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            obj_mesh = trimesh.load(scene_path, force='mesh')
            obj_mesh.vertices[:, 2] -= floor_height + 0.05
            vis_mesh = [
                floor_mesh,
                        init_body_mesh,
                        obj_mesh,
                navmesh,
                        trimesh.creation.axis()
                        ]

            marker_meshes = []
            for point_idx, pelvis in enumerate(wpath[:, :].reshape(-1, 3)):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                # sm = trimesh.creation.uv_sphere(radius=0.02)
                # sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                # sm.apply_transform(trans_mat)
                # vis_mesh.append(sm)
                point_axis = trimesh.creation.axis(transform=trans_mat, axis_radius=0, origin_size=0.05, origin_color=np.array([0, 200, 0]))
                vis_mesh.append(point_axis)
                if point_idx > 0:
                    marker_meshes.append(point_axis)


            marker_dirs = wpath[1] - start_markers
            marker_dirs = marker_dirs / torch.norm(marker_dirs, keepdim=True, dim=-1)
            for marker_idx in range(start_markers.reshape(-1, 3).shape[0]):
                marker = start_markers.reshape(-1, 3)[marker_idx]
                marker_dir = marker_dirs.reshape(-1, 3)[marker_idx]
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = marker.detach().cpu().numpy()
                sm = trimesh.creation.uv_sphere(radius=0.02)
                sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                sm.apply_transform(trans_mat)
                vis_mesh.append(sm)
                marker_meshes.append(sm)
                grad_vec = trimesh.creation.cylinder(radius=0.002, segment=np.stack(
                    [marker.detach().cpu().numpy(),
                     (marker + 0.1 * marker_dir).detach().cpu().numpy()]))
                grad_vec.visual.vertex_colors = np.array([0, 0, 255, 255])
                marker_meshes.append(grad_vec)
            # trimesh.util.concatenate(marker_meshes).show()

            # visualize map
            joints = bm(**xbo_dict).joints  # [b,p,3]
            x_axis = joints[:, 2, :] - joints[:, 1, :]
            x_axis[:, -1] = 0
            x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
            z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
            y_axis = torch.cross(z_axis, x_axis)
            gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
            gamma_transl = joints[0, 0, :].reshape(1, 1, 3)
            _, points, map = get_map(navmesh, gamma_orient, gamma_transl,
                                    res=res, extent=extent,
                                  return_type='numpy')
            # _, points, map = get_map_trimesh(navmesh, navmesh_query, gamma_orient, gamma_transl,
            #                          res=32, extent=1.6,
            #                          return_type='numpy')
            points = points[0]  # [p, 3]
            map = map[0]  #[p]
            cells = []
            for point_idx in range(points.shape[0]):
                color = np.array([0, 0, 200, 100]) if map[point_idx] else np.array([200, 0, 0, 200])
                transform = np.eye(4)
                transform[:3, 3] = points[point_idx]
                cell = trimesh.creation.box(extents=(0.05, 0.05, 1.5), vertex_colors=color, transform=transform)
                # cell = trimesh.creation.cylinder(radius=0.02,
                #                           segment=np.stack([points[point_idx], points[point_idx] + np.array([0, 0, 0.5])], axis=0),
                #                           vertex_colors=color)
                cells.append(cell)
            vis_mesh.append(trimesh.util.concatenate(cells))

            print(xbo_dict['wpath'])
            import pyrender
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        return xbo_dict

class BatchGeneratorInteractionTrain(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path, shapenet_dir, sdf_dir, data_path_list,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67'  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        self.interaction_data = []
        for data_path in data_path_list:
            with open(data_path, 'rb') as f:
                 self.interaction_data += pickle.load(f)
        self.shapenet_dir = shapenet_dir
        self.sdf_dir = sdf_dir
        if not os.path.exists(sdf_dir):
            os.mkdir(sdf_dir)
        import logging
        logger = logging.getLogger("trimesh")
        logger.setLevel(logging.ERROR)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=False,
                  interaction_id=None, hard_code=None, reverse=False):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''

        """read interaction"""
        # interaction_id = 373
        if interaction_id is None:
            interaction_id = torch.randint(len(self.interaction_data), size=(1,)).item()
            # interaction_id = torch.randint(32, size=(1,)).item()
        interaction = self.interaction_data[interaction_id]
        gender = 'male'
        bm = self.bm_male if gender == 'male' else self.bm_female
        smplx_params = deepcopy(interaction['smplx'])  # will change smplx_param inplace
        smplx_params = {k: v.cpu().cuda() if type(v)==torch.Tensor else v for k, v in smplx_params.items() }  # change cuda device to current device
        objects = deepcopy(interaction['objects'])
        assert objects['obj_num'] == 1
        shapenet_id = objects['shapenet_id'][0]
        # transform = deepcopy(objects['transform'][0])
        # print("load interaction:", shapenet_id + '-id' + str(interaction_id))
        # load or calc sdf grid
        sdf_path = Path(self.sdf_dir, shapenet_id + '_' + str(interaction['time']) + '.pkl')
        if not sdf_path.exists():
            object_mesh = trimesh.load(
                os.path.join(*([self.shapenet_dir] + shapenet_id.split('-') + ['models', 'model_normalized.obj'])),
                force='mesh'
            )
            transform = deepcopy(objects['transform'][0])
            transform[:3, 3] -= np.array([0, 0, interaction['floor_height']])
            object_mesh.apply_transform(transform)

            scene_centroid = object_mesh.bounding_box.centroid
            extents = object_mesh.bounding_box.extents
            floor_mesh = trimesh.creation.box(extents=np.array([extents[0] + 2, extents[1] + 2, 0.5]),
                                              transform=np.array([[1.0, 0.0, 0.0, scene_centroid[0]],
                                                                  [0.0, 1.0, 0.0, scene_centroid[1]],
                                                                  [0.0, 0.0, 1.0, -0.25],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            scene_mesh = object_mesh + floor_mesh
            # scene_mesh.show()
            scene_extents = extents + np.array([2, 2, 1])
            scene_scale = np.max(scene_extents) * 0.5
            scene_mesh.vertices -= scene_centroid
            scene_mesh.vertices /= scene_scale
            sign_method = 'normal'
            voxel_resolution = 128
            surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method='scan',
                                                          bounding_radius=3 ** 0.5,
                                                          scan_count=100,
                                                          scan_resolution=400, sample_point_count=10000000,
                                                          calculate_normals=(sign_method == 'normal'))

            sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth',
                                                                     sample_count=11, pad=False,
                                                                     check_result=False, return_gradients=True)
            print(sdf_grid.shape, gradient_grid.shape)
            object_sdf = {
                'grid': sdf_grid * scene_scale,
                'gradient_grid': gradient_grid,
                'dim': voxel_resolution,
                'centroid': scene_centroid,
                'scale': scene_scale,
            }
            with open(sdf_path, 'wb') as f:
                pickle.dump(object_sdf, f)
        else:
            with open(sdf_path, 'rb') as f:
                object_sdf = pickle.load(f)

        """sdf to tensor"""
        sdf_grids = torch.from_numpy(object_sdf['grid'])
        object_sdf['grid'] = sdf_grids.squeeze().unsqueeze(0).unsqueeze(0).to(device='cuda',
                                                                            dtype=torch.float32)  # 1x1xDxDxD
        if 'gradient_grid' in object_sdf:
            gradient_grids = torch.from_numpy(object_sdf['gradient_grid'])
            object_sdf['gradient_grid'] = gradient_grids.permute(3, 0, 1, 2).unsqueeze(0).to(device='cuda',
                                                                                           dtype=torch.float32)  # 1x3xDxDxD
        object_sdf['centroid'] = torch.tensor(object_sdf['centroid']).reshape(1, 1, 3).to(device='cuda',
                                                                                      dtype=torch.float32)

        """randomly specify a 3D path"""
        wpath = np.zeros((3, 3))
        # starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(
            wpath)  # starting point, ending point, another point to initialize the body orientation
        wpath[0] = 0  # starting point
        if 'sit' in interaction['interaction']:
            # r = torch.cuda.FloatTensor(1).uniform_() * 0.4 + 0.6
            r = 1.5
            theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
            body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
            forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
            forward_dir[2] = 0
            forward_dir = forward_dir / torch.norm(forward_dir)
            random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]), convention="XYZ")
            forward_dir = torch.matmul(random_rot, forward_dir)
            # wpath[1, :2] = sigma*(2*torch.cuda.FloatTensor(2).uniform_()-1) #ending point xy
            wpath[1] = -forward_dir * r
        elif 'lie' in interaction['interaction']:
            r = torch.cuda.FloatTensor(1).uniform_() * 0.4 + 0.6
            theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
            forward_dir = torch.tensor([0, 0, -1.0]).to(device='cuda', dtype=torch.float32)
            forward_dir = torch.matmul(torch.tensor(objects['transform'][0][:3, :3]).to(device='cuda', dtype=torch.float32), forward_dir)
            forward_dir[2] = 0
            forward_dir = forward_dir / torch.norm(forward_dir).clip(min=1e-12)
            # random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
            #                                                          convention="XYZ")
            # forward_dir = torch.matmul(random_rot, forward_dir)
            wpath[1] = -forward_dir * r
        # # wpath[2] for inital body orientation
        # theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        # random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
        #                                                          convention="XYZ")
        # forward_dir = torch.matmul(random_rot, forward_dir)
        # wpath[2, :2] = forward_dir[:2]  # point to initialize the body orientation, not returned
        wpath[2, :2] = torch.randn(2)  # point to initialize the body orientation, not returned

        # left
        # wpath[1, 0] = 0.5
        # wpath[1, 1] = -0.2
        # wpath[2, :2] = - wpath[1, :2]  # point to initialize the body orientation, not returned
        # back
        # wpath[1, 0] = 0
        # wpath[1, 1] = 0.75
        # wpath[2, :2] = wpath[1, :2] #point to initialize the body orientation, not returned
        # front
        # wpath[1, 0] = 0
        # wpath[1, 1] = -0.75
        # wpath[2, :2] = wpath[1, :2] #point to initialize the body orientation, not returned
        # lie
        # wpath[1, 0] = 0.75
        # wpath[1, 1] = -0.75

        """ translate object and target body to the sampled location and make floor to be the plane z=0"""
        transl_xy = wpath[1, :2] - bm(**smplx_params).joints[0, 0, :2]
        transl_z = -torch.cuda.FloatTensor([interaction['floor_height']])
        transl = torch.cat([transl_xy, transl_z])
        smplx_params['transl'] = smplx_params['transl'] + transl
        object_sdf['centroid'][:, :, :2] += transl_xy  # floor height has already been considered in calc sdf grids
        # get transformed mesh
        object_mesh = trimesh.load(
            os.path.join(*([self.shapenet_dir] + shapenet_id.split('-') + ['models', 'model_normalized.obj'])),
            force='mesh'
        )
        transform = deepcopy(objects['transform'][0])
        transform[:3, 3] = transform[:3, 3] + transl.cpu().numpy()
        object_mesh.apply_transform(transform)
        output = bm(**smplx_params)
        wpath[1] = output.joints[:, 0, :].detach()
        target_markers = output.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """generate init body"""
        xbo_dict = {}
        gender = 'male'
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None, ...]
        xbo_dict['transl'] = wpath[:1]  # [1,3]
        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict['transl'], wpath[0] = self.snap_to_ground_recenter_origin(xbo_dict,
                                                                           bm)  # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        start_markers = bm(**xbo_dict).vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """specify output"""
        # xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        xbo_dict['markers'] = torch.cat([start_markers, target_markers], dim=0)
        # xbo_dict['obj_mesh'] = object_mesh
        xbo_dict['obj_id'] = shapenet_id
        xbo_dict['obj_transform'] = torch.cuda.FloatTensor(transform)
        xbo_dict['obj_sdf'] = object_sdf
        obj_points, _ = trimesh.sample.sample_surface_even(object_mesh, 1024)
        xbo_dict['obj_points'] = torch.cuda.FloatTensor(obj_points)
        xbo_dict['target_body'] = deepcopy(smplx_params)

        """" reverse start and target body"""
        target_orient = R.from_rotvec(smplx_params['global_orient'].detach().cpu().numpy() if not reverse else xbo_dict[
            'global_orient'].detach().cpu().numpy())
        joints = bm(**(smplx_params if not reverse else xbo_dict)).joints  # [b,p,3]
        if reverse:
            for key in smplx_params:
                if key in xbo_dict:
                    xbo_dict['target_body'][key] = xbo_dict[key]
                if key != 'betas':
                    xbo_dict[key] = smplx_params[key]
            xbo_dict['wpath'] = torch.flip(xbo_dict['wpath'], [0])
            xbo_dict['markers'] = torch.flip(xbo_dict['markers'], [0])

        """target orientation"""
        xbo_dict['target_orient'] = torch.cuda.FloatTensor(target_orient.as_rotvec())  # [1, 3]
        xbo_dict['target_orient_matrix'] = torch.cuda.FloatTensor(target_orient.as_matrix())  # [1, 3, 3]
        xbo_dict['wpath_orients'] = torch.cat([xbo_dict['global_orient'], xbo_dict['target_orient']], dim=0)
        # target_forward_dir = xbo_dict['target_orient_matrix'][:, :3, 2]  # [1, 3]
        # target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        # target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        # xbo_dict['target_forward_dir'] = target_forward_dir
        x_axis = joints[:, 2, :] - joints[:, 1, :]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
        z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
        y_axis = torch.cross(z_axis, x_axis)
        gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
        target_forward_dir = y_axis  # [1, 3], y-axis
        target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        xbo_dict['target_forward_dir'] = target_forward_dir

        if visualize:
            target_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict['target_body']).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([200, 100, 100])
            )
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                         transform=np.array([[1.0, 0.0, 0.0, 0],
                                                             [0.0, 1.0, 0.0, 0],
                                                             [0.0, 0.0, 1.0, -0.005],
                                                             [0.0, 0.0, 0.0, 1.0],
                                                             ]),
                                         )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            forward_dir_segment = torch.cat([joints[:, 0, :], joints[:, 0, :] + target_forward_dir], dim=0).detach().cpu().numpy()
            forward_dir_segment = trimesh.creation.annulus(0.01, 0.03, segment=forward_dir_segment)
            forward_dir_segment.visual.vertex_colors = np.array([0, 0, 255, 255])
            # forward_dir_segment.visual.vertex_colors = np.array([255, 0, 0, 255])
            vis_mesh = [floor_mesh,
                        target_body_mesh,
                        init_body_mesh,
                        object_mesh,
                        forward_dir_segment,
                        trimesh.creation.axis(),
                        ]
            for point_idx, pelvis in enumerate(xbo_dict['wpath']):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                trans_mat[:3, :3] = R.from_rotvec(xbo_dict['wpath_orients'][point_idx].detach().cpu().numpy()).as_matrix()
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)
            for marker in xbo_dict['markers'].reshape(-1, 3):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = marker.detach().cpu().numpy()
                sm = trimesh.creation.uv_sphere(radius=0.02)
                sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                sm.apply_transform(trans_mat)
                vis_mesh.append(sm)
            print(xbo_dict['wpath'])
            # trimesh.util.concatenate(vis_mesh).show()
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        xbo_dict['betas'] = xbo_dict['betas'][0]
        # xbo_dict = self.params2numpy(xbo_dict)

        self.index_rec += 1

        return xbo_dict

class BatchGeneratorInteraction2frameTrain(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path, shapenet_dir, sdf_dir, data_path_list,
                 motion_seed_list,
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67'  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        self.interaction_data = []
        for data_path in data_path_list:
            with open(data_path, 'rb') as f:
                 self.interaction_data += pickle.load(f)
        self.motion_seed_list = motion_seed_list
        self.shapenet_dir = shapenet_dir
        self.sdf_dir = sdf_dir

        self.bm_2frame = smplx.create(body_model_path, model_type='smplx',
                                    gender='male', ext='npz',
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
                                    batch_size=2
                                    ).eval().cuda()

        if not os.path.exists(sdf_dir):
            os.mkdir(sdf_dir)
        import logging
        logger = logging.getLogger("trimesh")
        logger.setLevel(logging.ERROR)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=False,
                  interaction_id=None, hard_code=None, reverse=False):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''

        """read interaction"""
        # interaction_id = 373
        if interaction_id is None:
            interaction_id = torch.randint(len(self.interaction_data), size=(1,)).item()
            # interaction_id = torch.randint(32, size=(1,)).item()
        interaction = self.interaction_data[interaction_id]
        gender = 'male'
        bm = self.bm_male if gender == 'male' else self.bm_female
        smplx_params = deepcopy(interaction['smplx'])  # will change smplx_param inplace
        smplx_params = {k: v.cpu().cuda() if type(v)==torch.Tensor else v for k, v in smplx_params.items() }  # change cuda device to current device
        objects = deepcopy(interaction['objects'])
        assert objects['obj_num'] == 1
        shapenet_id = objects['shapenet_id'][0]
        # load or calc sdf grid
        sdf_path = Path(self.sdf_dir, shapenet_id + '_' + str(interaction['time']) + '.pkl')
        if not sdf_path.exists():
            object_mesh = trimesh.load(
                os.path.join(*([self.shapenet_dir] + shapenet_id.split('-') + ['models', 'model_normalized.obj'])),
                force='mesh'
            )
            transform = deepcopy(objects['transform'][0])
            transform[:3, 3] -= np.array([0, 0, interaction['floor_height']])
            object_mesh.apply_transform(transform)

            scene_centroid = object_mesh.bounding_box.centroid
            extents = object_mesh.bounding_box.extents
            floor_mesh = trimesh.creation.box(extents=np.array([extents[0] + 2, extents[1] + 2, 0.5]),
                                              transform=np.array([[1.0, 0.0, 0.0, scene_centroid[0]],
                                                                  [0.0, 1.0, 0.0, scene_centroid[1]],
                                                                  [0.0, 0.0, 1.0, -0.25],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            scene_mesh = object_mesh + floor_mesh
            # scene_mesh.show()
            scene_extents = extents + np.array([2, 2, 1])
            scene_scale = np.max(scene_extents) * 0.5
            scene_mesh.vertices -= scene_centroid
            scene_mesh.vertices /= scene_scale
            sign_method = 'normal'
            voxel_resolution = 128
            surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method='scan',
                                                          bounding_radius=3 ** 0.5,
                                                          scan_count=100,
                                                          scan_resolution=400, sample_point_count=10000000,
                                                          calculate_normals=(sign_method == 'normal'))

            sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth',
                                                                     sample_count=11, pad=False,
                                                                     check_result=False, return_gradients=True)
            print(sdf_grid.shape, gradient_grid.shape)
            object_sdf = {
                'grid': sdf_grid * scene_scale,
                'gradient_grid': gradient_grid,
                'dim': voxel_resolution,
                'centroid': scene_centroid,
                'scale': scene_scale,
            }
            with open(sdf_path, 'wb') as f:
                pickle.dump(object_sdf, f)
        else:
            with open(sdf_path, 'rb') as f:
                object_sdf = pickle.load(f)

        """sdf to tensor"""
        sdf_grids = torch.from_numpy(object_sdf['grid'])
        object_sdf['grid'] = sdf_grids.squeeze().unsqueeze(0).unsqueeze(0).to(device='cuda',
                                                                            dtype=torch.float32)  # 1x1xDxDxD
        if 'gradient_grid' in object_sdf:
            gradient_grids = torch.from_numpy(object_sdf['gradient_grid'])
            object_sdf['gradient_grid'] = gradient_grids.permute(3, 0, 1, 2).unsqueeze(0).to(device='cuda',
                                                                                           dtype=torch.float32)  # 1x3xDxDxD
        object_sdf['centroid'] = torch.tensor(object_sdf['centroid']).reshape(1, 1, 3).to(device='cuda',
                                                                                      dtype=torch.float32)

        """randomly specify a 3D path"""
        wpath = np.zeros((3, 3))
        # starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(
            wpath)  # starting point, ending point, another point to initialize the body orientation
        wpath[0] = 0  # starting point
        if 'sit' in interaction['interaction']:
            r = torch.cuda.FloatTensor(1).uniform_() * 0.4 + 0.6
            theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
            body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
            forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
            forward_dir[2] = 0
            forward_dir = forward_dir / torch.norm(forward_dir)
            random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]), convention="XYZ")
            forward_dir = torch.matmul(random_rot, forward_dir)
            # wpath[1, :2] = sigma*(2*torch.cuda.FloatTensor(2).uniform_()-1) #ending point xy
            wpath[1] = -forward_dir * r
        elif 'lie' in interaction['interaction']:
            r = torch.cuda.FloatTensor(1).uniform_() * 0.4 + 0.6
            theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
            forward_dir = torch.tensor([0, 0, -1.0]).to(device='cuda', dtype=torch.float32)
            forward_dir = torch.matmul(torch.tensor(objects['transform'][0][:3, :3]).to(device='cuda', dtype=torch.float32), forward_dir)
            forward_dir[2] = 0
            forward_dir = forward_dir / torch.norm(forward_dir).clip(min=1e-12)
            # random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
            #                                                          convention="XYZ")
            # forward_dir = torch.matmul(random_rot, forward_dir)
            wpath[1] = -forward_dir * r
        # # wpath[2] for inital body orientation
        # theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        # random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
        #                                                          convention="XYZ")
        # forward_dir = torch.matmul(random_rot, forward_dir)
        # wpath[2, :2] = forward_dir[:2]  # point to initialize the body orientation, not returned
        wpath[2, :2] = torch.randn(2)  # point to initialize the body orientation, not returned



        """ translate object and target body to the sampled location and make floor to be the plane z=0"""
        transl_xy = wpath[1, :2] - bm(**smplx_params).joints[0, 0, :2]
        transl_z = -torch.cuda.FloatTensor([interaction['floor_height']])
        transl = torch.cat([transl_xy, transl_z])
        smplx_params['transl'] = smplx_params['transl'] + transl
        object_sdf['centroid'][:, :, :2] += transl_xy  # floor height has already been considered in calc sdf grids
        # get transformed mesh
        object_mesh = trimesh.load(
            os.path.join(*([self.shapenet_dir] + shapenet_id.split('-') + ['models', 'model_normalized.obj'])),
            force='mesh'
        )
        transform = deepcopy(objects['transform'][0])
        transform[:3, 3] = transform[:3, 3] + transl.cpu().numpy()
        object_mesh.apply_transform(transform)
        output = bm(**smplx_params)
        wpath[1] = output.joints[:, 0, :].detach()

        """body_param_seed, prev_betas, gender, R0, T0"""
        gender = 'male'
        bm_2frame = self.bm_2frame
        """generate init body"""
        if reverse:  # repeat interaction body twice as init
            motion_seed_dict = {}
            motion_seed_dict['betas'] = smplx_params['betas'].repeat(2, 1)
            motion_seed_dict['body_pose'] = smplx_params['body_pose'].repeat(2, 1)
            motion_seed_dict['global_orient'] = smplx_params['global_orient'].repeat(2, 1)
            motion_seed_dict['transl'] = smplx_params['transl'].repeat(2, 1)
        else:  # load from walking data, take random 2 frames
            motion_seed_path = random.choice(self.motion_seed_list)
            motion_data = np.load(motion_seed_path)
            start_frame = torch.randint(0, len(motion_data['poses']) - 1, (1,)).item()
            motion_seed_dict = {}
            motion_seed_dict['betas'] = torch.cuda.FloatTensor(motion_data['betas']).reshape((1, 10)).repeat(2, 1)
            motion_seed_dict['body_pose'] = torch.cuda.FloatTensor(motion_data['poses'][start_frame:start_frame + 2, 3:66])
            motion_seed_dict['global_orient'] = torch.cuda.FloatTensor(motion_data['poses'][start_frame:start_frame + 2, :3])
            motion_seed_dict['transl'] = torch.cuda.FloatTensor(motion_data['trans'][start_frame:start_frame + 2])

            # randomly rotate around up-vec
            theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi * 2
            random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
                                                                     convention="XYZ").reshape(1, 3, 3)
            pelvis_zero = bm_2frame(betas=motion_seed_dict['betas']).joints[:1, 0, :]  # [1, 3]
            original_rot = pytorch3d.transforms.axis_angle_to_matrix(motion_seed_dict['global_orient'])
            new_rot = torch.einsum('bij,bjk->bik', random_rot, original_rot)
            new_transl = torch.einsum('bij,bj->bi', random_rot, pelvis_zero + motion_seed_dict['transl']) - pelvis_zero
            motion_seed_dict['global_orient'] = pytorch3d.transforms.matrix_to_axis_angle(new_rot)
            motion_seed_dict['transl'] = new_transl

            # translate to make the init body pelvis above origin, feet on floor
            output = bm_2frame(**motion_seed_dict)
            transl = torch.cuda.FloatTensor([output.joints[0, 0, 0], output.joints[0, 0, 1], output.joints[0, :, 2].amin()])
            motion_seed_dict['transl'] -= transl
        output = bm_2frame(**motion_seed_dict)
        start_markers = output.vertices.detach()[:, marker_ssm_67, :]  # [2, 67, 3]
        wpath[0] = output.joints[0, 0, :]


        """get target body"""
        if reverse:
            target_body_dict = {'gender':gender}
            target_body_dict['body_pose'] = self.vposer.decode(
                torch.cuda.FloatTensor(1, 32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1, 32).normal_(),
                # prone to self-interpenetration
                output_type='aa').view(1, -1)
            target_body_dict['global_orient'] = self.get_bodyori_from_wpath(torch.cuda.FloatTensor([0, 0, 0]), wpath[-1])[None, ...]
            target_body_dict['transl'] = torch.cuda.FloatTensor([[0, 0, 0]])
            """snap to the ground"""
            target_body_dict['transl'], _ = self.snap_to_ground_recenter_origin(target_body_dict, bm)  # snap foot to ground, rec
        else:
            target_body_dict = smplx_params
        output = bm(**target_body_dict)
        target_markers = output.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]
        target_joints = output.joints
        wpath[1] = output.joints[0, 0, :]

        """specify output"""
        xbo_dict = {}
        xbo_dict['gender']=gender
        xbo_dict['motion_seed'] = motion_seed_dict
        xbo_dict['betas'] = motion_seed_dict['betas'][:1, :]
        xbo_dict['wpath']=wpath[:2]
        xbo_dict['markers'] = torch.cat([start_markers, target_markers], dim=0)
        xbo_dict['obj_id'] = shapenet_id
        xbo_dict['obj_transform'] = torch.cuda.FloatTensor(transform)
        xbo_dict['obj_sdf'] = object_sdf
        obj_points, _ = trimesh.sample.sample_surface_even(object_mesh, 1024)
        xbo_dict['obj_points'] = torch.cuda.FloatTensor(obj_points)
        # xbo_dict['target_body'] = target_body_dict

        """target orientation"""
        xbo_dict['target_orient'] = target_body_dict['global_orient']  # [1, 3]
        xbo_dict['target_orient_matrix'] = pytorch3d.transforms.axis_angle_to_matrix(target_body_dict['global_orient'])  # [1, 3, 3]
        xbo_dict['wpath_orients'] = torch.cat([motion_seed_dict['global_orient'][:1], target_body_dict['global_orient']], dim=0)
        x_axis = target_joints[:, 2, :] - target_joints[:, 1, :]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
        z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
        y_axis = torch.cross(z_axis, x_axis)
        gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
        target_forward_dir = y_axis  # [1, 3], y-axis
        target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        xbo_dict['target_forward_dir'] = target_forward_dir

        if visualize:
            target_body_mesh = trimesh.Trimesh(
                vertices=bm(**target_body_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([200, 100, 100])
            )
            init_body1_mesh = trimesh.Trimesh(
                vertices=bm_2frame(**motion_seed_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            init_body2_mesh = trimesh.Trimesh(
                vertices=bm_2frame(**motion_seed_dict).vertices[1].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([150, 150, 150])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                         transform=np.array([[1.0, 0.0, 0.0, 0],
                                                             [0.0, 1.0, 0.0, 0],
                                                             [0.0, 0.0, 1.0, -0.005],
                                                             [0.0, 0.0, 0.0, 1.0],
                                                             ]),
                                         )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            forward_dir_segment = torch.cat([target_joints[:, 0, :], target_joints[:, 0, :] + target_forward_dir], dim=0).detach().cpu().numpy()
            forward_dir_segment = trimesh.creation.annulus(0.01, 0.03, segment=forward_dir_segment)
            forward_dir_segment.visual.vertex_colors = np.array([0, 0, 255, 255])
            # forward_dir_segment.visual.vertex_colors = np.array([255, 0, 0, 255])
            vis_mesh = [floor_mesh,
                        target_body_mesh,
                        init_body1_mesh,
                        init_body2_mesh,
                        object_mesh,
                        forward_dir_segment,
                        trimesh.creation.axis(),
                        ]
            for point_idx, pelvis in enumerate(xbo_dict['wpath']):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                trans_mat[:3, :3] = R.from_rotvec(xbo_dict['wpath_orients'][point_idx].detach().cpu().numpy()).as_matrix()
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)
            for marker in xbo_dict['markers'].reshape(-1, 3):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = marker.detach().cpu().numpy()
                sm = trimesh.creation.uv_sphere(radius=0.02)
                sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                sm.apply_transform(trans_mat)
                vis_mesh.append(sm)
            print(xbo_dict['wpath'])
            # trimesh.util.concatenate(vis_mesh).show()
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        xbo_dict['betas'] = xbo_dict['betas'][0]
        # xbo_dict = self.params2numpy(xbo_dict)

        self.index_rec += 1

        return xbo_dict

class BatchGeneratorInteractionTest(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path='',
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67'  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=True, use_zero_shape=True,
                  target_body_path=None, last_motion_path=None,
                  target_point_path=None, start_point_path=None,
                  sdf_path=None, mesh_path=None, scene_path=None, floor_height=0,
                  ):
        if not sdf_path.exists():
            instance_mesh = trimesh.load(mesh_path, force='mesh')
            if 'obj' in mesh_path.name:
                instance_mesh.apply_transform(shapenet_to_zup)
            instance_mesh.vertices[:, 2] -= floor_height
            scene_centroid = instance_mesh.bounding_box.centroid
            extents = instance_mesh.bounding_box.extents
            floor_mesh = trimesh.creation.box(extents=np.array([extents[0] + 2, extents[1] + 2, 0.5]),
                                              transform=np.array([[1.0, 0.0, 0.0, scene_centroid[0]],
                                                                  [0.0, 1.0, 0.0, scene_centroid[1]],
                                                                  [0.0, 0.0, 1.0, -0.25],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            scene_mesh = instance_mesh + floor_mesh
            # scene_mesh.show()
            scene_extents = extents + np.array([2, 2, 1])
            scene_scale = np.max(scene_extents) * 0.5
            scene_mesh.vertices -= scene_centroid
            scene_mesh.vertices /= scene_scale
            sign_method = 'normal'
            voxel_resolution = 128
            surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method='scan',
                                                          bounding_radius=3 ** 0.5,
                                                          scan_count=100,
                                                          scan_resolution=400, sample_point_count=10000000,
                                                          calculate_normals=(sign_method == 'normal'))

            sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth',
                                                                     sample_count=11, pad=False,
                                                                     check_result=False, return_gradients=True)
            print(sdf_grid.shape, gradient_grid.shape)
            object_sdf = {
                'grid': sdf_grid * scene_scale,
                'gradient_grid': gradient_grid,
                'dim': voxel_resolution,
                'centroid': scene_centroid,
                'scale': scene_scale,
            }
            sdf_path.parent.mkdir(exist_ok=True, parents=True)
            with open(sdf_path, 'wb') as f:
                pickle.dump(object_sdf, f)
            # visualize
            if visualize:
                import skimage
                vertices, faces, normals, _ = skimage.measure.marching_cubes(object_sdf['grid'], level=0)
                vertices = vertices / object_sdf['dim'] * 2 - 1
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
                sdf_scene = pyrender.Scene()
                sdf_scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
                sphere = trimesh.creation.uv_sphere(radius=0.02)
                poses = np.tile(np.eye(4), (2, 1, 1))
                poses[0, :3, 3] = np.array([1, 1, 1])
                poses[1, :3, 3] = -np.array([1, 1, 1])
                sdf_scene.add(pyrender.Mesh.from_trimesh(sphere, poses=np.array(poses)))
                pyrender.Viewer(sdf_scene, use_raymond_lighting=True, run_in_thread=False)
        else:
            with open(sdf_path, 'rb') as f:
                object_sdf = pickle.load(f)
        """sdf to tensor"""
        sdf_grids = torch.from_numpy(object_sdf['grid'])
        object_sdf['grid'] = sdf_grids.squeeze().unsqueeze(0).unsqueeze(0).to(device='cuda',
                                                                              dtype=torch.float32)  # 1x1xDxDxD
        if 'gradient_grid' in object_sdf:
            gradient_grids = torch.from_numpy(object_sdf['gradient_grid'])
            object_sdf['gradient_grid'] = gradient_grids.permute(3, 0, 1, 2).unsqueeze(0).to(device='cuda',
                                                                                             dtype=torch.float32)  # 1x3xDxDxD
        object_sdf['centroid'] = torch.tensor(object_sdf['centroid']).reshape(1, 1, 3).to(device='cuda',
                                                                                          dtype=torch.float32)


        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''

        """randomly specify a 3D path"""
        wpath = np.zeros((2, 3))  # starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(wpath)  # starting point, ending point, another point to initialize the body orientation

        """load init body or start point"""
        xbo_dict = {}
        if last_motion_path is not None:  # sit after locomotion
            with open(last_motion_path, 'rb') as f:
                motion_data = pickle.load(f)  # [n, 3]
            last_primitive = motion_data['motion'][-1]
            gender = last_primitive['gender']
            xbo_dict['betas'] = betas = torch.cuda.FloatTensor(last_primitive['betas']).reshape((1, 10))
            smplx_params = torch.cuda.FloatTensor(last_primitive['smplx_params'][0, -1:])
            R0 = torch.cuda.FloatTensor(last_primitive['transf_rotmat'])
            T0 = torch.cuda.FloatTensor(last_primitive['transf_transl'])
            from models.baseops import SMPLXParser
            pconfig_1frame = {
                'n_batch': 1,
                'device': 'cuda',
                'marker_placement': 'ssm2_67'
            }
            smplxparser_1frame = SMPLXParser(pconfig_1frame)
            smplx_params = smplxparser_1frame.update_transl_glorot(R0.permute(0, 2, 1),
                                                                   -torch.einsum('bij,bkj->bki', R0.permute(0, 2, 1),
                                                                                 T0),
                                                                   betas=betas,
                                                                   gender=gender,
                                                                   xb=smplx_params,
                                                                   inplace=False,
                                                                   to_numpy=False)  # T0 must be [1, 1, 3], [1,3] leads to error
            xbo_dict['transl'] = smplx_params[:, :3]
            xbo_dict['global_orient'] = smplx_params[:, 3:6]
            xbo_dict['body_pose'] = smplx_params[:, 6:69]
        else:  # only sit/lie
            gender = random.choice(['male'])
            xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_() if use_zero_shape else torch.cuda.FloatTensor(1, 10).normal_()
            # xbo_dict['body_pose'] = torch.cuda.FloatTensor(rest_pose) if use_zero_pose else self.vposer.decode(torch.cuda.FloatTensor(1,32).normal_(), output_type='aa').view(1, -1)
            xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1, 32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1, 32).normal_(), output_type='aa').view(1, -1)
            # xbo_dict['body_pose'] = self.vposer.decode(self.vposer.encode(torch.cuda.FloatTensor(rest_pose)).mean, output_type='aa').view(1, -1)
            with open(start_point_path, 'rb') as f:
                data = pickle.load(f)
            if len(data.shape) == 1:
                start_point = data
                start_point = torch.cuda.FloatTensor(start_point)
                orient_point = start_point + torch.cuda.FloatTensor(3).normal_()
            else: # 2 points, specify location and orientation
                start_point, orient_point = data
                start_point = torch.cuda.FloatTensor(start_point)
                orient_point = torch.cuda.FloatTensor(orient_point)

            wpath[0] = start_point
            orient_point[2] = start_point[2]
            xbo_dict['global_orient'] = self.get_bodyori_from_wpath(start_point, orient_point)[None, ...]

            """snap to the ground"""
            bm = self.bm_male if gender == 'male' else self.bm_female
            xbo_dict['transl'] = start_point - bm(**xbo_dict).joints[[0], 0, :]  # [1,3]
            xbo_dict = self.snap_to_ground(xbo_dict,
                                           bm)  # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        bm = self.bm_male if gender == 'male' else self.bm_female
        init_body = bm(**xbo_dict)
        wpath[0] = init_body.joints[0, 0]
        start_markers = init_body.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """read target body"""
        if target_body_path is not None:  # sit/lie down
            with open(target_body_path, 'rb') as f:
                target_body = pickle.load(f)
            # gender = target_body['gender']
            smplx_params = self.params2torch(target_body)
            smplx_params = {k: v.cpu().cuda() if type(v)==torch.Tensor else v for k, v in smplx_params.items() }  # change cuda device to current device
        else:  # stand up
            with open(target_point_path, 'rb') as f:
                data = pickle.load(f)  # [3,]
            if len(data.shape) == 1:
                target_point = data
                target_point = torch.cuda.FloatTensor(target_point)
                target_point[2] = wpath[0, 2]
                smplx_params = {'gender': 'male'}
                smplx_params['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
                smplx_params['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1, 32).zero_(), output_type='aa').view(1, -1)
                smplx_params['global_orient'] = self.get_bodyori_from_wpath(wpath[0], target_point)[None, ...]
                """snap to the ground"""
                bm = self.bm_male if gender == 'male' else self.bm_female
                smplx_params['transl'] = (target_point - bm(**smplx_params).joints[0, 0]).unsqueeze(0)
                smplx_params = self.snap_to_ground(smplx_params, bm)  # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
                # start_markers = bm(**xbo_dict).vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]
            else:  # 2 points, specify location and orientation
                target_point, next_point = data
                target_point = torch.cuda.FloatTensor(target_point)
                next_point = torch.cuda.FloatTensor(next_point)
                smplx_params = {'gender': 'male'}
                smplx_params['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
                smplx_params['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1, 32).zero_(),
                                                               output_type='aa').view(1, -1)
                smplx_params['global_orient'] = self.get_bodyori_from_wpath(target_point, next_point)[None, ...]
                """snap to the ground"""
                bm = self.bm_male if gender == 'male' else self.bm_female
                smplx_params['transl'] = (target_point - bm(**smplx_params).joints[0, 0]).unsqueeze(0)
                smplx_params = self.snap_to_ground(smplx_params, bm)  # snap foot to ground, recenter pelvis right

        bm = self.bm_male if gender == 'male' else self.bm_female
        output = bm(**smplx_params)
        wpath[1] = output.joints[:, 0, :].detach()
        target_markers = output.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """specify output"""
        # xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        xbo_dict['markers'] = torch.cat([start_markers, target_markers], dim=0)
        xbo_dict['scene_path'] = scene_path
        if 'obj' in scene_path.name:
            xbo_dict['obj_transform'] = shapenet_to_zup
        xbo_dict['floor_height'] = floor_height
        xbo_dict['obj_sdf'] = object_sdf
        xbo_dict['target_body'] = deepcopy(smplx_params)
        xbo_dict['motion_history'] = motion_data if last_motion_path is not None else None

        """" reverse start and target body"""
        target_orient = R.from_rotvec(smplx_params['global_orient'].detach().cpu().numpy())
        joints = bm(**(smplx_params)).joints  # [b,p,3]

        """target orientation"""
        xbo_dict['target_orient'] = torch.cuda.FloatTensor(target_orient.as_rotvec())  # [1, 3]
        xbo_dict['target_orient_matrix'] = torch.cuda.FloatTensor(target_orient.as_matrix())  # [1, 3, 3]
        xbo_dict['wpath_orients'] = torch.cat([xbo_dict['global_orient'], xbo_dict['target_orient']], dim=0)
        # target_forward_dir = xbo_dict['target_orient_matrix'][:, :3, 2]  # [1, 3]
        # target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        # target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        # xbo_dict['target_forward_dir'] = target_forward_dir
        x_axis = joints[:, 2, :] - joints[:, 1, :]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
        z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
        y_axis = torch.cross(z_axis, x_axis)
        gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
        target_forward_dir = y_axis  # [1, 3], y-axis
        target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        xbo_dict['target_forward_dir'] = target_forward_dir

        if visualize:
            target_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict['target_body']).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 200, 100])
            )
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                         transform=np.array([[1.0, 0.0, 0.0, 0],
                                                             [0.0, 1.0, 0.0, 0],
                                                             [0.0, 0.0, 1.0, -0.0051],
                                                             [0.0, 0.0, 0.0, 1.0],
                                                             ]),
                                         )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            forward_dir_segment = torch.cat([joints[:, 0, :], joints[:, 0, :] + target_forward_dir], dim=0).detach().cpu().numpy()
            forward_dir_segment = trimesh.creation.annulus(0.01, 0.03, segment=forward_dir_segment)
            forward_dir_segment.visual.vertex_colors = np.array([0, 0, 255, 255])
            # forward_dir_segment.visual.vertex_colors = np.array([255, 0, 0, 255])
            scene_mesh = trimesh.load(scene_path, force='mesh')
            if 'obj' in scene_path.name:
                scene_mesh.apply_transform(shapenet_to_zup)
            scene_mesh.vertices[:, 2] -= floor_height + 0.05
            vis_mesh = [floor_mesh,
                        target_body_mesh,
                        init_body_mesh,
                        scene_mesh,
                        # forward_dir_segment,
                        # trimesh.creation.axis(),
                        ]

            # for point_idx, pelvis in enumerate(xbo_dict['wpath']):
            #     trans_mat = np.eye(4)
            #     trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
            #     trans_mat[:3, :3] = R.from_rotvec(xbo_dict['wpath_orients'][point_idx].detach().cpu().numpy()).as_matrix()
            #     point_axis = trimesh.creation.axis(transform=trans_mat)
            #     vis_mesh.append(point_axis)

            marker_meshes = []
            marker_dirs = target_markers - start_markers
            marker_dirs = marker_dirs / torch.norm(marker_dirs, keepdim=True, dim=-1)
            for marker_idx in range(start_markers.reshape(-1, 3).shape[0]):
                marker = start_markers.reshape(-1, 3)[marker_idx]
                marker_dir = marker_dirs.reshape(-1, 3)[marker_idx]
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = marker.detach().cpu().numpy()
                sm = trimesh.creation.uv_sphere(radius=0.02)
                sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                sm.apply_transform(trans_mat)
                vis_mesh.append(sm)
                marker_meshes.append(sm)
                grad_vec = trimesh.creation.cylinder(radius=0.002, segment=np.stack(
                    [marker.detach().cpu().numpy(),
                     (marker + 0.1 * marker_dir).detach().cpu().numpy()]))
                grad_vec.visual.vertex_colors = np.array([0, 0, 255, 255])
                marker_meshes.append(grad_vec)
            for marker_idx in range(target_markers.reshape(-1, 3).shape[0]):
                marker = target_markers.reshape(-1, 3)[marker_idx]
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = marker.detach().cpu().numpy()
                sm = trimesh.creation.uv_sphere(radius=0.02)
                sm.visual.vertex_colors = [0.0, 1.0, 0.0]
                sm.apply_transform(trans_mat)
                vis_mesh.append(sm)
                marker_meshes.append(sm)
            # trimesh.util.concatenate(marker_meshes).show()

            # for marker in xbo_dict['markers'].reshape(-1, 3):
            #     trans_mat = np.eye(4)
            #     trans_mat[:3, 3] = marker.detach().cpu().numpy()
            #     sm = trimesh.creation.uv_sphere(radius=0.02)
            #     sm.visual.vertex_colors = [1.0, 0.0, 0.0]
            #     sm.apply_transform(trans_mat)
            #     vis_mesh.append(sm)
            print(xbo_dict['wpath'])
            # trimesh.util.concatenate(vis_mesh).show()
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        xbo_dict['betas'] = xbo_dict['betas'][0]
        # xbo_dict = self.params2numpy(xbo_dict)

        self.index_rec += 1

        return xbo_dict

class BatchGeneratorInteractionShapenetTest(BatchGeneratorReachingTarget):
    def __init__(self,
                 dataset_path='',
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67'  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        import logging
        logger = logging.getLogger("trimesh")
        logger.setLevel(logging.ERROR)

    def next_body(self, sigma=10, visualize=False, use_zero_pose=False,
                  mesh_path=Path(''), sdf_path=Path(''), target_body=None,
                  reverse=False):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''

        """read interaction"""
        object_mesh = trimesh.load_mesh(mesh_path, force='mesh')
        object_mesh.apply_transform(shapenet_to_zup)
        if not sdf_path.exists():
            extents = object_mesh.bounding_box.extents
            floor_mesh = trimesh.creation.box(extents=np.array([extents[0] + 2, extents[1] + 2, 0.5]),
                                              transform=np.array([[1.0, 0.0, 0.0, 0],
                                                                  [0.0, 1.0, 0.0, 0],
                                                                  [0.0, 0.0, 1.0, -0.25],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            scene_mesh = object_mesh + floor_mesh
            # scene_mesh.show()
            scene_extents = extents + np.array([2, 2, 1])
            scene_scale = np.max(scene_extents) * 0.5
            scene_centroid = object_mesh.bounding_box.centroid
            scene_mesh.vertices -= scene_centroid
            scene_mesh.vertices /= scene_scale
            sign_method = 'normal'
            voxel_resolution = 128
            surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method='scan',
                                                          bounding_radius=3 ** 0.5, scan_count=100,
                                                          scan_resolution=400, sample_point_count=10000000,
                                                          calculate_normals=(sign_method == 'normal'))

            sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth',
                                                                     sample_count=11, pad=False,
                                                                     check_result=False, return_gradients=True)
            object_sdf = {
                'grid': sdf_grid * scene_scale,
                'gradient_grid': gradient_grid,
                'dim': voxel_resolution,
                'centroid': scene_centroid,
                'scale': scene_scale,
            }
            with open(sdf_path, 'wb') as f:
                pickle.dump(object_sdf, f)
        else:
            with open(sdf_path, 'rb') as f:
                object_sdf = pickle.load(f)

        """sdf to tensor"""
        sdf_grids = torch.from_numpy(object_sdf['grid'])
        object_sdf['grid'] = sdf_grids.squeeze().unsqueeze(0).unsqueeze(0).to(device='cuda',
                                                                            dtype=torch.float32)  # 1x1xDxDxD
        if 'gradient_grid' in object_sdf:
            gradient_grids = torch.from_numpy(object_sdf['gradient_grid'])
            object_sdf['gradient_grid'] = gradient_grids.permute(3, 0, 1, 2).unsqueeze(0).to(device='cuda',
                                                                                           dtype=torch.float32)  # 1x3xDxDxD
        object_sdf['centroid'] = torch.tensor(object_sdf['centroid']).reshape(1, 1, 3).to(device='cuda',
                                                                                      dtype=torch.float32)

        """randomly specify a 3D path"""
        wpath = np.zeros((3, 3))
        # starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(
            wpath)  # starting point, ending point, another point to initialize the body orientation
        gender = 'male'
        bm = self.bm_male if gender == 'male' else self.bm_female
        smplx_params = self.params2torch(target_body)  # will change smplx_param inplace
        smplx_params = {k: v.cpu().cuda() if type(v) == torch.Tensor else v for k, v in
                        smplx_params.items()}  # change cuda device to current device
        output = bm(**smplx_params)
        wpath[1] = output.joints[:, 0, :].detach()
        target_markers = output.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        # starting point
        r = torch.cuda.FloatTensor(1).uniform_() * 0.4 + 0.6
        theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
        forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
        forward_dir[2] = 0
        forward_dir = forward_dir / torch.norm(forward_dir)
        random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]), convention="XYZ")
        forward_dir = torch.matmul(random_rot, forward_dir)
        wpath[0] = wpath[1] + forward_dir * r
        wpath[2, :2] = torch.randn(2).clip(min=1e-12).to(device=wpath.device) + wpath[0, :2]  # point to initialize the body orientation, not returned
        wpath[2, 2] = wpath[0, 2]



        """generate init body"""
        xbo_dict = {}
        gender = 'male'
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None, ...]
        xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[[0], 0, :] # [1,3]
        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict = self.snap_to_ground(xbo_dict, bm)  # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        init_body = bm(**xbo_dict)
        wpath[0] = init_body.joints[0, 0, :]
        start_markers = init_body.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """specify output"""
        # xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        xbo_dict['markers'] = torch.cat([start_markers, target_markers], dim=0)
        # xbo_dict['obj_mesh'] = object_mesh
        xbo_dict['scene_path'] = mesh_path
        xbo_dict['obj_transform'] = shapenet_to_zup
        xbo_dict['obj_sdf'] = object_sdf
        obj_points, _ = trimesh.sample.sample_surface_even(object_mesh, 1024)
        xbo_dict['obj_points'] = torch.cuda.FloatTensor(obj_points)
        xbo_dict['floor_height'] = 0
        xbo_dict['target_body'] = deepcopy(smplx_params)

        """" reverse start and target body"""
        target_orient = R.from_rotvec(smplx_params['global_orient'].detach().cpu().numpy() if not reverse else xbo_dict[
            'global_orient'].detach().cpu().numpy())
        joints = bm(**(smplx_params if not reverse else xbo_dict)).joints  # [b,p,3]
        if reverse:
            for key in smplx_params:
                if key in xbo_dict:
                    xbo_dict['target_body'][key] = xbo_dict[key]
                if key != 'betas':
                    xbo_dict[key] = smplx_params[key]
            xbo_dict['wpath'] = torch.flip(xbo_dict['wpath'], [0])
            xbo_dict['markers'] = torch.flip(xbo_dict['markers'], [0])

        """target orientation"""
        xbo_dict['target_orient'] = torch.cuda.FloatTensor(target_orient.as_rotvec())  # [1, 3]
        xbo_dict['target_orient_matrix'] = torch.cuda.FloatTensor(target_orient.as_matrix())  # [1, 3, 3]
        xbo_dict['wpath_orients'] = torch.cat([xbo_dict['global_orient'], xbo_dict['target_orient']], dim=0)
        # target_forward_dir = xbo_dict['target_orient_matrix'][:, :3, 2]  # [1, 3]
        # target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        # target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        # xbo_dict['target_forward_dir'] = target_forward_dir
        x_axis = joints[:, 2, :] - joints[:, 1, :]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
        z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
        y_axis = torch.cross(z_axis, x_axis)
        gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
        target_forward_dir = y_axis  # [1, 3], y-axis
        target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        xbo_dict['target_forward_dir'] = target_forward_dir

        if visualize:
            target_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict['target_body']).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([200, 100, 100])
            )
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                         transform=np.array([[1.0, 0.0, 0.0, 0],
                                                             [0.0, 1.0, 0.0, 0],
                                                             [0.0, 0.0, 1.0, -0.005],
                                                             [0.0, 0.0, 0.0, 1.0],
                                                             ]),
                                         )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            forward_dir_segment = torch.cat([joints[:, 0, :], joints[:, 0, :] + target_forward_dir], dim=0).detach().cpu().numpy()
            forward_dir_segment = trimesh.creation.annulus(0.01, 0.03, segment=forward_dir_segment)
            forward_dir_segment.visual.vertex_colors = np.array([0, 0, 255, 255])
            # forward_dir_segment.visual.vertex_colors = np.array([255, 0, 0, 255])
            vis_mesh = [floor_mesh,
                        target_body_mesh,
                        init_body_mesh,
                        object_mesh,
                        forward_dir_segment,
                        trimesh.creation.axis(),
                        ]
            for point_idx, pelvis in enumerate(xbo_dict['wpath']):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                trans_mat[:3, :3] = R.from_rotvec(xbo_dict['wpath_orients'][point_idx].detach().cpu().numpy()).as_matrix()
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)
            for marker in xbo_dict['markers'].reshape(-1, 3):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = marker.detach().cpu().numpy()
                sm = trimesh.creation.uv_sphere(radius=0.02)
                sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                sm.apply_transform(trans_mat)
                vis_mesh.append(sm)
            print(xbo_dict['wpath'])
            # trimesh.util.concatenate(vis_mesh).show()
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        xbo_dict['betas'] = xbo_dict['betas'][0]
        # xbo_dict = self.params2numpy(xbo_dict)

        self.index_rec += 1

        return xbo_dict

class BatchGeneratorInteractionReplicaTest(BatchGeneratorReachingTarget):
    def __init__(self,
                 scene,
                 dataset_path='',
                 body_model_path='/home/yzhang/body_models/VPoser',
                 body_repr='ssm2_67'  # ['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                 ):
        super().__init__(dataset_path, body_model_path, body_repr)
        self.scene = scene
        self.scene_name = scene.name
        self.replica_folder = scene.replica_folder
        self.scene_path = self.replica_folder / scene.name / 'mesh.ply'
        self.scene_mesh = scene.mesh

        # self.sdf_path = sdf_path = self.replica_folder / scene.name / 'sdf.pkl'
        # self.object_sdf = object_sdf


    def next_body(self, sigma=10, target_body=None, visualize=False, use_zero_pose=False,
                  sdf_path=None, instance_mesh=None,
                  interaction=None, reverse=False, start_point=None):
        if not sdf_path.exists():
            scene_centroid = instance_mesh.bounding_box.centroid
            extents = instance_mesh.bounding_box.extents
            floor_mesh = trimesh.creation.box(extents=np.array([extents[0] + 2, extents[1] + 2, 0.5]),
                                              transform=np.array([[1.0, 0.0, 0.0, scene_centroid[0]],
                                                                  [0.0, 1.0, 0.0, scene_centroid[1]],
                                                                  [0.0, 0.0, 1.0, -0.25],
                                                                  [0.0, 0.0, 0.0, 1.0],
                                                                  ]),
                                              )
            scene_mesh = instance_mesh + floor_mesh
            # scene_mesh.show()
            scene_extents = extents + np.array([2, 2, 1])
            scene_scale = np.max(scene_extents) * 0.5
            scene_mesh.vertices -= scene_centroid
            scene_mesh.vertices /= scene_scale
            sign_method = 'normal'
            voxel_resolution = 128
            surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method='scan',
                                                          bounding_radius=3 ** 0.5,
                                                          scan_count=100,
                                                          scan_resolution=400, sample_point_count=10000000,
                                                          calculate_normals=(sign_method == 'normal'))

            sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth',
                                                                     sample_count=11, pad=False,
                                                                     check_result=False, return_gradients=True)
            print(sdf_grid.shape, gradient_grid.shape)
            object_sdf = {
                'grid': sdf_grid * scene_scale,
                'gradient_grid': gradient_grid,
                'dim': voxel_resolution,
                'centroid': scene_centroid,
                'scale': scene_scale,
            }
            with open(sdf_path, 'wb') as f:
                pickle.dump(object_sdf, f)
            # visualize
            import skimage
            vertices, faces, normals, _ = skimage.measure.marching_cubes(object_sdf['grid'], level=0)
            vertices = vertices / object_sdf['dim'] * 2 - 1
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
            sdf_scene = pyrender.Scene()
            sdf_scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            sphere = trimesh.creation.uv_sphere(radius=0.02)
            poses = np.tile(np.eye(4), (2, 1, 1))
            poses[0, :3, 3] = np.array([1, 1, 1])
            poses[1, :3, 3] = -np.array([1, 1, 1])
            sdf_scene.add(pyrender.Mesh.from_trimesh(sphere, poses=np.array(poses)))
            pyrender.Viewer(sdf_scene, use_raymond_lighting=True, run_in_thread=False)
        else:
            with open(sdf_path, 'rb') as f:
                object_sdf = pickle.load(f)
        """sdf to tensor"""
        sdf_grids = torch.from_numpy(object_sdf['grid'])
        object_sdf['grid'] = sdf_grids.squeeze().unsqueeze(0).unsqueeze(0).to(device='cuda',
                                                                              dtype=torch.float32)  # 1x1xDxDxD
        if 'gradient_grid' in object_sdf:
            gradient_grids = torch.from_numpy(object_sdf['gradient_grid'])
            object_sdf['gradient_grid'] = gradient_grids.permute(3, 0, 1, 2).unsqueeze(0).to(device='cuda',
                                                                                             dtype=torch.float32)  # 1x3xDxDxD
        object_sdf['centroid'] = torch.tensor(object_sdf['centroid']).reshape(1, 1, 3).to(device='cuda',
                                                                                          dtype=torch.float32)


        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''

        """randomly specify a 3D path"""
        wpath = np.zeros((3, 3))
        # starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(
            wpath)  # starting point, ending point, another point to initialize the body orientation

        """read target body"""
        gender = 'male'
        bm = self.bm_male if gender == 'male' else self.bm_female
        smplx_params = self.params2torch(target_body)
        smplx_params = {k: v.cpu().cuda() if type(v)==torch.Tensor else v for k, v in smplx_params.items() }  # change cuda device to current device
        """ target body to make floor to be the plane z=0 """
        smplx_params['transl'][:, 2] -= self.scene.raw_floor_height
        output = bm(**smplx_params)
        wpath[1] = output.joints[:, 0, :].detach()
        target_markers = output.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        # starting point
        if 'sit' in interaction:
            # r = torch.cuda.FloatTensor(1).uniform_() * 0.4 + 0.6
            r = 0.6
            theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
            body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
            forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
            forward_dir[2] = 0
            forward_dir = forward_dir / torch.norm(forward_dir)
            random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]), convention="XYZ")
            # forward_dir = torch.matmul(random_rot, forward_dir)
            wpath[0] = wpath[1] + forward_dir * r
        elif 'lie' in interaction:
            wpath[0] = torch.cuda.FloatTensor(start_point)
        wpath[2, :2] = torch.randn(2)  # point to initialize the body orientation, not returned

        """generate init body"""
        xbo_dict = {}
        gender = 'male'
        # xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).normal_()
        xbo_dict['betas'] = torch.cuda.FloatTensor(1, 10).zero_()
        xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).zero_() if use_zero_pose else torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                       output_type='aa').view(1, -1)
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None, ...]
        xbo_dict['transl'] = wpath[:1] - bm(**xbo_dict).joints[0, 0, :] # [1,3]
        """snap to the ground"""
        bm = self.bm_male if gender == 'male' else self.bm_female
        xbo_dict = self.snap_to_ground(xbo_dict, bm)  # snap foot to ground, recenter pelvis right above origin, set starting point at pelvis
        init_body = bm(**xbo_dict)
        wpath[0] = init_body.joints[0, 0]
        start_markers = init_body.vertices.detach()[:, marker_ssm_67, :]  # [1, 67, 3]

        """specify output"""
        # xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        xbo_dict['markers'] = torch.cat([start_markers, target_markers], dim=0)
        xbo_dict['scene_path'] = self.scene_path
        xbo_dict['floor_height'] = self.scene.raw_floor_height
        xbo_dict['obj_sdf'] = object_sdf
        obj_points, _ = trimesh.sample.sample_surface_even(instance_mesh, 1024)
        xbo_dict['obj_points'] = torch.cuda.FloatTensor(obj_points)
        xbo_dict['target_body'] = deepcopy(smplx_params)

        """" reverse start and target body"""
        target_orient = R.from_rotvec(smplx_params['global_orient'].detach().cpu().numpy() if not reverse else xbo_dict[
            'global_orient'].detach().cpu().numpy())
        joints = bm(**(smplx_params if not reverse else xbo_dict)).joints  # [b,p,3]
        if reverse:
            for key in smplx_params:
                if key in xbo_dict:
                    xbo_dict['target_body'][key] = xbo_dict[key]
                if key != 'betas':
                    xbo_dict[key] = smplx_params[key]
            xbo_dict['wpath'] = torch.flip(xbo_dict['wpath'], [0])
            xbo_dict['markers'] = torch.flip(xbo_dict['markers'], [0])

        """target orientation"""
        xbo_dict['target_orient'] = torch.cuda.FloatTensor(target_orient.as_rotvec())  # [1, 3]
        xbo_dict['target_orient_matrix'] = torch.cuda.FloatTensor(target_orient.as_matrix())  # [1, 3, 3]
        xbo_dict['wpath_orients'] = torch.cat([xbo_dict['global_orient'], xbo_dict['target_orient']], dim=0)
        # target_forward_dir = xbo_dict['target_orient_matrix'][:, :3, 2]  # [1, 3]
        # target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        # target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        # xbo_dict['target_forward_dir'] = target_forward_dir
        x_axis = joints[:, 2, :] - joints[:, 1, :]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
        z_axis = torch.cuda.FloatTensor([[0, 0, 1]]).repeat(x_axis.shape[0], 1)
        y_axis = torch.cross(z_axis, x_axis)
        gamma_orient = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [1, 3, 3]
        target_forward_dir = y_axis  # [1, 3], y-axis
        target_forward_dir[:, 2] = 0  # set z 0, only xy direction for locomotion
        target_forward_dir = target_forward_dir / torch.norm(target_forward_dir, dim=-1, keepdim=True)
        xbo_dict['target_forward_dir'] = target_forward_dir

        if visualize:
            target_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict['target_body']).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([200, 100, 100])
            )
            init_body_mesh = trimesh.Trimesh(
                vertices=bm(**xbo_dict).vertices[0].detach().cpu().numpy(),
                faces=bm.faces,
                vertex_colors=np.array([100, 100, 100])
            )
            floor_mesh = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                         transform=np.array([[1.0, 0.0, 0.0, 0],
                                                             [0.0, 1.0, 0.0, 0],
                                                             [0.0, 0.0, 1.0, -0.005],
                                                             [0.0, 0.0, 0.0, 1.0],
                                                             ]),
                                         )
            floor_mesh.visual.vertex_colors = [0.8, 0.8, 0.8]
            forward_dir_segment = torch.cat([joints[:, 0, :], joints[:, 0, :] + target_forward_dir], dim=0).detach().cpu().numpy()
            forward_dir_segment = trimesh.creation.annulus(0.01, 0.03, segment=forward_dir_segment)
            forward_dir_segment.visual.vertex_colors = np.array([0, 0, 255, 255])
            # forward_dir_segment.visual.vertex_colors = np.array([255, 0, 0, 255])
            vis_mesh = [floor_mesh,
                        target_body_mesh,
                        init_body_mesh,
                        self.scene_mesh,
                        forward_dir_segment,
                        trimesh.creation.axis(),
                        ]
            for point_idx, pelvis in enumerate(xbo_dict['wpath']):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = pelvis.detach().cpu().numpy()
                trans_mat[:3, :3] = R.from_rotvec(xbo_dict['wpath_orients'][point_idx].detach().cpu().numpy()).as_matrix()
                point_axis = trimesh.creation.axis(transform=trans_mat)
                vis_mesh.append(point_axis)
            for marker in xbo_dict['markers'].reshape(-1, 3):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = marker.detach().cpu().numpy()
                sm = trimesh.creation.uv_sphere(radius=0.02)
                sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                sm.apply_transform(trans_mat)
                vis_mesh.append(sm)
            print(xbo_dict['wpath'])
            # trimesh.util.concatenate(vis_mesh).show()
            scene = pyrender.Scene()
            for mesh in vis_mesh:
                scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False)))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        xbo_dict['betas'] = xbo_dict['betas'][0]
        # xbo_dict = self.params2numpy(xbo_dict)

        self.index_rec += 1

        return xbo_dict
