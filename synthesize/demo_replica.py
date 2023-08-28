def room_0_sit():
    scene_name = 'room_0'
    scene = ReplicaScene(scene_name=scene_name, replica_folder=replica_folder, zero_floor=False)
    floor_height = scene.raw_floor_height
    scene_path = scene.ply_path
    navmesh_path = scene.replica_folder / scene_name / 'navmesh_tight.ply'
    navmesh_loose_path = scene.replica_folder / scene_name / 'navmesh_loose.ply'
    # get loose navmesh for path planning
    if navmesh_loose_path.exists():
        navmesh_loose = trimesh.load(navmesh_loose_path, force='mesh')
    else:
        scene_mesh = trimesh.load(scene_path, force='mesh')
        """assume the scene coords are z-up"""
        scene_mesh.vertices[:, 2] -= floor_height
        scene_mesh.apply_transform(zup_to_shapenet)
        navmesh_loose = create_navmesh(scene_mesh, export_path=navmesh_loose_path, agent_radius=0.2, visualize=True)
    navmesh_loose.vertices[:, 2] = 0
    seq_num = 20
    for seq_id in range(seq_num):
        target_interaction_path = '/home/kaizhao/projects/gamma/results/coins/two_stage/gen/optimization_after_get_body/sit on/room_0/sit on_room_0_39/1.pkl'
        action = 'sit'
        obj_category = 'stool'
        obj_id = 39
        path_name = 'door_to_stool_{}'.format(seq_id)
        interaction_name = '_'.join([action, obj_category, str(obj_id), str(seq_id)])
        wpath_path = scene.replica_folder / scene_name / 'waypoints' / (path_name + '.pkl')
        wpath_path.parent.mkdir(exist_ok=True, parents=True)
        sdf_path = scene.replica_folder / scene_name / 'sdf' / (str(obj_id) + '.pkl')
        sdf_path.parent.mkdir(exist_ok=True, parents=True)
        mesh_path = scene.replica_folder / scene_name / 'instances' / (str(obj_id) + '.ply')
        mesh_path.parent.mkdir(exist_ok=True, parents=True)
        target_point_path = Path('results', 'tmp', scene_name, interaction_name, 'target_point.pkl')
        target_point_path.parent.mkdir(exist_ok=True, parents=True)
        target_body_path = Path('results', 'tmp', scene_name, interaction_name, 'target_body.pkl')

        with open(target_interaction_path, 'rb') as f:
            target_interaction = pickle.load(f)
        smplx_params = target_interaction['smplx_param']
        del smplx_params['left_hand_pose']
        del smplx_params['right_hand_pose']
        smplx_params['transl'][:, 2] -= floor_height + 0.1
        smplx_params['gender'] = 'male'
        with open(target_body_path, 'wb') as f:
            pickle.dump(smplx_params, f)

        smplx_params = params2torch(smplx_params)
        pelvis = bm(**smplx_params).joints[0, 0, :].detach().cpu().numpy()

        start_point = np.array([0.51, -0.43, 0])
        # r = torch.cuda.FloatTensor(1).uniform_() * 0.4 + 0.6
        r = 0.6
        forward_dir = torch.cuda.FloatTensor([0, -1, 0])
        theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
        forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
        forward_dir[2] = 0
        forward_dir = forward_dir / torch.norm(forward_dir)
        # random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]), convention="XYZ")
        # forward_dir = torch.matmul(random_rot, forward_dir)
        target_point = pelvis + (forward_dir * r).detach().cpu().numpy()
        target_point[2] = 0
        start_target = np.stack([start_point, target_point])

        scene_mesh = deepcopy(scene.mesh)
        scene_mesh.vertices[:, 2] -= scene.raw_floor_height + 0.05
        wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=False, scene_mesh=scene_mesh)
        with open(wpath_path, 'wb') as f:
            pickle.dump(wpath, f)

        command = "python synthesize/gen_locomotion_unify.py --goal_thresh 0.5 --goal_thresh_final 0.2 --max_depth 60 --num_gen1 128 --num_gen2 16 --num_expand 8 " \
                  "--project_dir /mnt/atlas_root/vlg-nfs/kaizhao/gamma --cfg_policy ../../../../../../mnt/atlas_root/vlg-nfs/kaizhao/gamma/results/exp_GAMMAPrimitive/MPVAEPolicy_samp_collision/map_kl10_batchfix_pene1 " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --navmesh_path {} --floor_height {:.2f} --wpath_path {} --path_name {} " \
                  "--weight_pene 1 " \
                  "--visualize 0 --use_zero_pose 1 --use_zero_shape 1".format(seq_id, scene_path, scene_name, navmesh_path, floor_height, wpath_path, path_name)
        print(command)
        os.system(command)

        last_motion_path = 'results/locomotion/room_0/{}/MPVAEPolicy_samp_collision/map_kl10_batchfix_pene1/policy_search/seq000/results_ssm2_67_condi_marker_map_0.pkl'.format(path_name)
        """sit down"""
        command = "python synthesize/gen_interaction_unify.py --goal_thresh_final -1 --max_depth 15 --num_gen1 128 --num_gen2 32 --num_expand 4 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_babel_marker/sit_2frame_test " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                  "--target_body_path {} --interaction_name {} --last_motion_path {} " \
                  "--history_mode 2 --weight_target_dist 1 " \
                  "--visualize 0".format(seq_id, scene_path, scene_name, sdf_path, mesh_path, floor_height, target_body_path, interaction_name + '_down', last_motion_path)
        print(command)
        os.system(command)


        last_motion_path = '/home/kaizhao/projects/gamma/results/interaction/room_0/{}_down/MPVAEPolicy_babel_marker/sit_2frame_test/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'.format(interaction_name)
        with open(target_point_path, 'wb') as f:
            pickle.dump(target_point + np.array([0.3, 0, 0]), f)
        """stand up"""
        command = "python synthesize/gen_interaction_unify.py --goal_thresh_final 0.3 --max_depth 10 --num_gen1 128 --num_gen2 32 --num_expand 4 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_babel_marker/sit_2frame_test " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                  "--target_point_path {} --interaction_name {} --last_motion_path {} " \
                  "--history_mode 2 --weight_target_dist 1 " \
                  "--visualize 0".format(seq_id, scene_path, scene_name, sdf_path, mesh_path, floor_height, target_point_path, interaction_name + '_up', last_motion_path)
        print(command)
        os.system(command)

        last_motion_path = '/home/kaizhao/projects/gamma/results/interaction/room_0/{}_up/MPVAEPolicy_babel_marker/sit_2frame_test/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'.format(interaction_name)
        start_point = target_point
        target_interaction_path = '/home/kaizhao/projects/gamma/results/coins/two_stage/gen/optimization_after_get_body/sit on/room_0/sit on_room_0_74/3.pkl'
        action = 'sit'
        obj_category = 'chair'
        obj_id = 74
        path_name = 'stool_to_chair_{}'.format(seq_id)
        interaction_name = '_'.join([action, obj_category, str(obj_id), str(seq_id)])
        wpath_path = scene.replica_folder / scene_name / 'waypoints' / (path_name + '.pkl')
        wpath_path.parent.mkdir(exist_ok=True, parents=True)
        sdf_path = scene.replica_folder / scene_name / 'sdf' / (str(obj_id) + '.pkl')
        sdf_path.parent.mkdir(exist_ok=True, parents=True)
        mesh_path = scene.replica_folder / scene_name / 'instances' / (str(obj_id) + '.ply')
        mesh_path.parent.mkdir(exist_ok=True, parents=True)
        target_point_path = Path('results', 'tmp', scene_name, interaction_name, 'target_point.pkl')
        target_point_path.parent.mkdir(exist_ok=True, parents=True)
        target_body_path = Path('results', 'tmp', scene_name, interaction_name, 'target_body.pkl')

        with open(target_interaction_path, 'rb') as f:
            target_interaction = pickle.load(f)
        smplx_params = target_interaction['smplx_param']
        del smplx_params['left_hand_pose']
        del smplx_params['right_hand_pose']
        smplx_params['transl'][:, 2] -= floor_height + 0.1
        smplx_params['transl'][:, 0] += 0.05
        smplx_params['gender'] = 'male'
        with open(target_body_path, 'wb') as f:
            pickle.dump(smplx_params, f)
        smplx_params = params2torch(smplx_params)
        pelvis = bm(**smplx_params).joints[0, 0, :].detach().cpu().numpy()

        r = 0.4
        body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
        forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
        forward_dir[2] = 0
        forward_dir = forward_dir / torch.norm(forward_dir)
        theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]), convention="XYZ")
        # forward_dir = torch.matmul(random_rot, forward_dir)
        target_point = pelvis + (forward_dir * r).detach().cpu().numpy()
        target_point[2] = 0
        target_point = np.array([5.07, 0.53, 0])
        start_target = np.stack([start_point, target_point])

        scene_mesh = deepcopy(scene.mesh)
        scene_mesh.vertices[:, 2] -= scene.raw_floor_height + 0.05
        wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=False, scene_mesh=scene_mesh)
        with open(wpath_path, 'wb') as f:
            pickle.dump(wpath, f)


        command = "python synthesize/gen_locomotion_unify.py --goal_thresh 0.5 --goal_thresh_final 0.5 --max_depth 60 --num_gen1 128 --num_gen2 32 --num_expand 8 " \
                  "--project_dir /mnt/atlas_root/vlg-nfs/kaizhao/gamma --cfg_policy ../../../../../../mnt/atlas_root/vlg-nfs/kaizhao/gamma/results/exp_GAMMAPrimitive/MPVAEPolicy_samp_collision/map_kl10_batchfix_pene1 " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --navmesh_path {} --floor_height {:.2f} --wpath_path {} --path_name {} --last_motion_path {} " \
                  "--history_mode 1 --weight_pene 5 " \
                  "--visualize 0 --clip_far 1 --use_zero_pose 1 --use_zero_shape 1".format(seq_id, scene_path, scene_name, navmesh_path, floor_height, wpath_path, path_name, last_motion_path)
        print(command)
        os.system(command)

        last_motion_path = '/home/kaizhao/projects/gamma/results/locomotion/room_0/{}/MPVAEPolicy_samp_collision/map_kl10_batchfix_pene1/policy_search/seq000/results_ssm2_67_condi_marker_map_0.pkl'.format(path_name)
        """sit down"""
        command = "python synthesize/gen_interaction_unify.py --goal_thresh_final -1 --max_depth 15 --num_gen1 128 --num_gen2 32 --num_expand 4 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_babel_marker/sit_2frame_test " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                  "--target_body_path {} --interaction_name {} --last_motion_path {} " \
                  "--history_mode 2 --weight_target_dist 5 " \
                  "--visualize 0".format(seq_id, scene_path, scene_name, sdf_path, mesh_path, floor_height, target_body_path, interaction_name + '_down', last_motion_path)
        print(command)
        os.system(command)
        #
        last_motion_path = 'results/interaction/room_0/{}_down/MPVAEPolicy_babel_marker/sit_2frame_test/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'.format(interaction_name)
        with open(target_point_path, 'wb') as f:
            # pickle.dump(target_point, f)
            pickle.dump(np.array([4.66, 0.46, 0]), f)
        """stand up"""
        command = "python synthesize/gen_interaction_unify.py --goal_thresh_final 0.3 --max_depth 10 --num_gen1 128 --num_gen2 32 --num_expand 4 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_babel_marker/sit_2frame_test " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                  "--target_point_path {} --interaction_name {} --last_motion_path {} " \
                  "--history_mode 2 --weight_target_dist 5 " \
                  "--visualize 0".format(seq_id, scene_path, scene_name, sdf_path, mesh_path, floor_height, target_point_path, interaction_name + '_up', last_motion_path)
        print(command)
        os.system(command)

        last_motion_path = '/home/kaizhao/projects/gamma/results/interaction/room_0/{}_up/MPVAEPolicy_babel_marker/sit_2frame_test/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'.format(interaction_name)
        start_point = target_point
        target_interaction_path = '/home/kaizhao/projects/gamma/results/coins/two_stage/gen/optimization_after_get_body/sit on/room_0/sit on_room_0_9/1.pkl'
        action = 'sit'
        obj_category = 'sofa'
        obj_id = 9
        path_name = 'chair_to_sofa_{}'.format(seq_id)
        interaction_name = '_'.join([action, obj_category, str(obj_id), str(seq_id)])
        wpath_path = scene.replica_folder / scene_name / 'waypoints' / (path_name + '.pkl')
        wpath_path.parent.mkdir(exist_ok=True, parents=True)
        sdf_path = scene.replica_folder / scene_name / 'sdf' / (str(obj_id) + '.pkl')
        sdf_path.parent.mkdir(exist_ok=True, parents=True)
        mesh_path = scene.replica_folder / scene_name / 'instances' / (str(obj_id) + '.ply')
        mesh_path.parent.mkdir(exist_ok=True, parents=True)

        target_point_path = Path('results', 'tmp', scene_name, interaction_name, 'target_point.pkl')
        target_point_path.parent.mkdir(exist_ok=True, parents=True)
        target_body_path = Path('results', 'tmp', scene_name, interaction_name, 'target_body.pkl')

        with open(target_interaction_path, 'rb') as f:
            target_interaction = pickle.load(f)
        smplx_params = target_interaction['smplx_param']
        del smplx_params['left_hand_pose']
        del smplx_params['right_hand_pose']
        smplx_params['transl'][:, 2] -= floor_height + 0.1
        smplx_params['transl'][:, 1] += 0.1
        smplx_params['gender'] = 'male'
        with open(target_body_path, 'wb') as f:
            pickle.dump(smplx_params, f)
        smplx_params = params2torch(smplx_params)
        pelvis = bm(**smplx_params).joints[0, 0, :].detach().cpu().numpy()

        r = 0.5
        body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()
        forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
        forward_dir[2] = 0
        forward_dir = forward_dir / torch.norm(forward_dir)
        theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
        random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]), convention="XYZ")
        # forward_dir = torch.matmul(random_rot, forward_dir)
        target_point = pelvis + (forward_dir * r).detach().cpu().numpy()
        target_point[2] = 0
        target_point = np.array([3.64, 0.34, 0])
        start_target = np.stack([start_point, target_point])

        scene_mesh = deepcopy(scene.mesh)
        scene_mesh.vertices[:, 2] -= scene.raw_floor_height + 0.05
        wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=False, scene_mesh=scene_mesh)
        with open(wpath_path, 'wb') as f:
            pickle.dump(wpath, f)

        command = "python synthesize/gen_locomotion_unify.py --goal_thresh 0.5 --goal_thresh_final 0.5 --max_depth 60 --num_gen1 128 --num_gen2 32 --num_expand 8 " \
                  "--project_dir /mnt/atlas_root/vlg-nfs/kaizhao/gamma --cfg_policy ../../../../../../mnt/atlas_root/vlg-nfs/kaizhao/gamma/results/exp_GAMMAPrimitive/MPVAEPolicy_samp_collision/map_kl10_batchfix_pene1 " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --navmesh_path {} --floor_height {:.2f} --wpath_path {} --path_name {} --last_motion_path {} " \
                  "--weight_pene 1 --history_mode 1 " \
                  "--visualize 0 --use_zero_pose 1 --use_zero_shape 1".format(seq_id, scene_path, scene_name, navmesh_path, floor_height, wpath_path, path_name, last_motion_path)
        print(command)
        os.system(command)

        last_motion_path = '/home/kaizhao/projects/gamma/results/locomotion/room_0/{}/MPVAEPolicy_samp_collision/map_kl10_batchfix_pene1/policy_search/seq000/results_ssm2_67_condi_marker_map_0.pkl'.format(path_name)
        """sit down"""
        command = "python synthesize/gen_interaction_unify.py --goal_thresh_final -1 --max_depth 15 --num_gen1 128 --num_gen2 32 --num_expand 4 " \
                  "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_babel_marker/sit_2frame_test " \
                  "--gen_name policy_search --num_sequence 1 " \
                  "--random_seed {} --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                  "--target_body_path {} --interaction_name {} --last_motion_path {} " \
                  "--weight_target_dist 1 --history_mode 2 " \
                  "--visualize 0".format(seq_id, scene_path, scene_name, sdf_path, mesh_path, floor_height, target_body_path, interaction_name + '_down', last_motion_path)
        print(command)
        os.system(command)
