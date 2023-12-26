import glob
import json
import pickle

import numpy as np
import trimesh
import mesh_to_sdf
import pyrender
from pathlib import Path
from pathfinder import navmesh_baker as nmb
import pathfinder as pf
from copy import deepcopy
from tqdm import tqdm
import torch
import pytorch3d
import pytorch3d.transforms
import time
import random

zup_to_shapenet = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]]
)
shapenet_to_zup = np.array(
    [[1, 0, 0, 0],
     [0, 0, -1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]]
)

unity_to_zup = np.array(
            [[-1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )

def triangulate(vertices, polygons):
    triangle_faces = []
    for face in polygons:
        for idx in range(len(face) - 2):
            triangle_faces.append((face[0], face[idx + 1], face[idx + 2]))
    return trimesh.Trimesh(vertices=np.array(vertices),
                           faces=np.array(triangle_faces),
                           vertex_colors=np.array([0, 0, 200, 100]))

def create_navmesh(scene_mesh, export_path, agent_radius=0.2,
                    agent_max_climb=0.1, agent_max_slope=15.0,
                   visualize=False):
    baker = nmb.NavmeshBaker()
    vertices = scene_mesh.vertices.tolist()
    vertices = [tuple(vertex) for vertex in vertices]
    faces = scene_mesh.faces.tolist()
    baker.add_geometry(vertices, faces)
    # bake navigation mesh
    baker.bake(
        verts_per_poly=3,
        cell_size=0.05, cell_height=0.05,
        agent_radius=agent_radius,
        agent_max_climb=agent_max_climb, agent_max_slope=agent_max_slope
    )
    # obtain polygonal description of the mesh
    vertices, polygons = baker.get_polygonization()
    print(vertices)
    print(polygons)
    triangulated = triangulate(vertices, polygons)
    triangulated.apply_transform(shapenet_to_zup)
    triangulated = triangulated.slice_plane(np.array([0, 0, 0.1]), np.array([0, 0, -1.0]))  # cut off floor faces
    triangulated.vertices[:, 2] = 0
    triangulated.visual.vertex_colors = np.array([0, 0, 200, 100])
    export_path.parent.mkdir(exist_ok=True, parents=True)
    triangulated.export(export_path)
    if visualize:
        scene = pyrender.Scene()
        scene_mesh = deepcopy(scene_mesh)
        scene_mesh.vertices[:, 1] -= 0.05
        scene_mesh.apply_transform(shapenet_to_zup)
        scene.add(pyrender.Mesh.from_trimesh(scene_mesh))
        scene.add(pyrender.Mesh.from_trimesh(triangulated))
        pyrender.Viewer(scene, use_raymond_lighting=True)
    return triangulated

"""return can be empty when no path"""
def path_find(navmesh_zup, start_zup, finish_zup, visualize=False, scene_mesh=None):
    # to yup
    start = tuple(start_zup.squeeze().tolist())
    finish = tuple(finish_zup.squeeze().tolist())
    start = (start[0], start[2], -start[1])
    finish = (finish[0], finish[2], -finish[1])
    navmesh = deepcopy(navmesh_zup)
    navmesh.apply_transform(zup_to_shapenet)


    vertices = navmesh.vertices.tolist()
    vertices = [tuple(vertex) for vertex in vertices]
    faces = navmesh.faces.tolist()
    pathfinder = pf.PathFinder(vertices, faces)
    path = pathfinder.search_path(start, finish)
    path = np.array(path)
    if len(path) > 0:
        # to zup
        path = np.stack([path[:, 0], -path[:, 2], path[:, 1],], axis=1)

    if visualize:
        scene = pyrender.Scene()
        if scene_mesh is not None:
            scene.add(pyrender.Mesh.from_trimesh(scene_mesh))
        scene.add(pyrender.Mesh.from_trimesh(navmesh_zup, poses=np.array([[1, 0, 0, 0],
                                                                      [0, 1, 0, 0, ],
                                                                      [0, 0, 1, 0.05],
                                                                      [0, 0, 0, 1]])))
        vis_meshes = []
        point = trimesh.creation.uv_sphere(radius=0.05)
        point.visual.vertex_colors = np.array([255, 0, 0, 255])
        point.vertices += np.array(start_zup)
        vis_meshes.append(point)
        point = trimesh.creation.uv_sphere(radius=0.05)
        point.visual.vertex_colors = np.array([255, 0, 0, 255])
        point.vertices += np.array(finish_zup)
        vis_meshes.append(point)
        for idx, waypoint in enumerate(path):
            point = trimesh.creation.uv_sphere(radius=0.05)
            point.visual.vertex_colors = np.array([255, 0, 0, 255])
            point.vertices += waypoint
            vis_meshes.append(point)
            if idx > 0:
                line = trimesh.creation.cylinder(radius=0.03, segment=path[idx - 1:idx + 1, :], )
                line.visual.vertex_colors = np.array([0, 255, 0, 255])
                vis_meshes.append(line)
        scene.add(pyrender.Mesh.from_trimesh(trimesh.util.concatenate(vis_meshes)))
        pyrender.Viewer(scene, use_raymond_lighting=True)
    return path

def test_shapenet():
    # create baker object
    shapenet_dir = Path('/mnt/atlas_root/vlg-data/ShapeNetCore.v2/')
    obj_category = '03001627'
    obj_id = '3d267294b4a01900b04cb542e2c50eb4'
    obj_dir = shapenet_dir / obj_category / obj_id / 'models'
    obj_path = obj_dir / 'model_normalized.obj'
    json_path = obj_dir / 'model_normalized.json'
    with open(json_path, 'r') as f:
        meta_info = json.load(f)
    obj_mesh = trimesh.load(obj_path, force='mesh')
    obj_mesh.vertices[:, 1] -= obj_mesh.bounds[0, 1]
    # scene = pyrender.Scene()
    # scene.add(pyrender.Mesh.from_trimesh(obj_mesh))
    # pyrender.Viewer(scene, use_raymond_lighting=True)

    floor_mesh = trimesh.Trimesh(
        vertices=np.array([(-4.0, 0.0, -4.0), (-4.0, 0.0, 4.0), (4.0, 0.0, 4.0), (4.0, 0.0, -4.0)]),
        faces=np.array([[0, 1, 2, 3]])
    )
    obj_mesh = obj_mesh + floor_mesh
    obj_mesh.export('scene.obj')

    baker = nmb.NavmeshBaker()
    # add geometry, for example a simple plane
    # the first array contains vertex positions, the second array contains polygons of the geometry
    # baker.add_geometry([(-20.0, 0.0, -20.0), (-20.0, 0.0, 20.0), (20.0, 0.0, 20.0), (20.0, 0.0, -20.0)], [[0, 1, 2, 3]])  # floor
    vertices = obj_mesh.vertices.tolist()
    vertices = [tuple(vertex) for vertex in vertices]
    faces = obj_mesh.faces.tolist()
    baker.add_geometry(vertices, faces)
    # bake navigation mesh
    baker.bake(verts_per_poly=3,
        cell_size=0.05, cell_height=0.05,
        agent_radius=0.01,
        agent_max_climb=0.1, agent_max_slope=15.0)
    # obtain polygonal description of the mesh
    vertices, polygons = baker.get_polygonization()
    print(vertices)
    print(polygons)
    triangulated = triangulate(vertices, polygons)
    scene = pyrender.Scene()
    obj_mesh.vertices[:, 1] -= 0.05
    scene.add(pyrender.Mesh.from_trimesh(obj_mesh))
    scene.add(pyrender.Mesh.from_trimesh(triangulated))
    pyrender.Viewer(scene, use_raymond_lighting=True)
    baker.save_to_text('data/navmesh/' + obj_category + '_' +obj_id + '.txt')

def test_naive():
    baker = nmb.NavmeshBaker()
    # plane
    baker.add_geometry([(-4.0, 0.0, -4.0), (-4.0, 0.0, 4.0), (4.0, 0.0, 4.0), (4.0, 0.0, -4.0)],
                       [[0, 1, 2, 3]])
    # cube
    baker.add_geometry([(-1.0, 0.0, -1.0), (-1.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, -1.0)],
                       [[0, 3, 2, 1], [2, 6, 5, 1], [4, 5, 6, 7], [0, 4, 7, 3], [2, 3, 7, 6], [0, 1, 5, 4]])
    baker.bake()
    vertices, polygons = baker.get_polygonization()
    print(vertices)
    print(polygons)
    triangulated = triangulate(vertices, polygons)
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(triangulated))
    pyrender.Viewer(scene, use_raymond_lighting=True)

def test_prox():

    prox_folder = Path('/home/kaizhao/projects/gamma/data/scenes/PROX')
    scene_names = [
        # "BasementSittingBooth", "MPH11", "MPH112", "MPH16", "MPH1Library",
                   "MPH8",
                   # "N0SittingBooth", "N0Sofa", "N3Library", "N3Office", "N3OpenArea", "Werkraum"
    ]
    for scene_name in tqdm(scene_names):
        print(scene_name)
        scene_mesh = trimesh.load_mesh(prox_folder / (scene_name + '_floor.ply'), force='mesh')
        scene_mesh_yup = deepcopy(scene_mesh)
        scene_mesh_yup.apply_transform(zup_to_shapenet)
        # scene_mesh_yup.export(prox_folder / (scene_name + '_yup.obj'))

        export_path = prox_folder / (scene_name + '_navmesh0.2.ply')
        navmesh = create_navmesh(scene_mesh_yup, export_path=export_path, agent_radius=0.2, visualize=False)
        export_path = prox_folder / (scene_name + '_navmesh0.3.ply')
        navmesh = create_navmesh(scene_mesh_yup, export_path=export_path, agent_radius=0.3, visualize=False)
        export_path = prox_folder / (scene_name + '_navmesh0.05.ply')
        navmesh_tight = create_navmesh(scene_mesh_yup, export_path=export_path, agent_radius=0.05, visualize=False)

        # baker = nmb.NavmeshBaker()
        # vertices = scene_mesh_yup.vertices.tolist()
        # vertices = [tuple(vertex) for vertex in vertices]
        # faces = scene_mesh_yup.faces.tolist()
        # baker.add_geometry(vertices, faces)
        # # bake navigation mesh
        # baker.bake(
        #     verts_per_poly=3,
        #     cell_size=0.05, cell_height=0.05,
        #     agent_radius=0.2,
        #     agent_max_climb=0.1, agent_max_slope=15.0
        # )
        # # obtain polygonal description of the mesh
        # vertices, polygons = baker.get_polygonization()
        # print(vertices)
        # print(polygons)
        # triangulated = triangulate(vertices, polygons)
        # triangulated.apply_transform(shapenet_to_zup)
        # triangulated = triangulated.slice_plane(np.array([0, 0, 0.2]), np.array([0, 0, -1.0]))  # cut off floor faces
        # triangulated.vertices[:, 2] = 0
        # triangulated.visual.vertex_colors = np.array([0, 0, 200, 100])
        # triangulated.export(prox_folder / (scene_name + '_navmesh.ply'))
        # # scene = pyrender.Scene()
        # # scene.add(pyrender.Mesh.from_trimesh(scene_mesh))
        # # scene.add(pyrender.Mesh.from_trimesh(triangulated))
        # # pyrender.Viewer(scene, use_raymond_lighting=True)

def generate_paris(scene_name, scene_path, scene_transl=None, visualize=False):
    scene_mesh = trimesh.load_mesh(scene_path, force='mesh')
    scene_mesh_yup = deepcopy(scene_mesh)
    scene_mesh_yup.apply_transform(zup_to_shapenet)
    scene_mesh_yup.export(scene_path.parent / (scene_name + '_yup.obj'))
    # if visualize:
    #     scene_mesh_yup.show()

    # export_path = scene_path.parent / (scene_name + '_navmesh0.2.ply')
    # navmesh = create_navmesh(scene_mesh_yup, export_path=export_path, agent_radius=0.2, agent_max_climb=0.2, agent_max_slope=30.0, visualize=visualize)
    # if scene_transl is not None:
    #     navmesh.vertices += scene_transl
    #     navmesh.export(export_path)
    # export_path = scene_path.parent / (scene_name + '_navmesh0.3.ply')
    # navmesh = create_navmesh(scene_mesh_yup, export_path=export_path, agent_radius=0.3, agent_max_climb=0.2, agent_max_slope=30.0, visualize=visualize)
    # if scene_transl is not None:
    #     navmesh.vertices += scene_transl
    #     navmesh.export(export_path)
    export_path = scene_path.parent / (scene_name + '_navmesh0.05.ply')
    navmesh = create_navmesh(scene_mesh_yup, export_path=export_path, agent_radius=0.051, agent_max_climb=0.2, agent_max_slope=30.0, visualize=visualize)
    if scene_transl is not None:
        navmesh.vertices += scene_transl
        navmesh.export(export_path)


def test_replica():
    scene_names = [
                   # 'apartment_0',
                   'apartment_1',
                   'apartment_2',
                   # 'frl_apartment_0',
                   # 'frl_apartment_1',
                   # 'frl_apartment_2',
                   # 'frl_apartment_3',
                   # 'frl_apartment_4',
                   # 'frl_apartment_5',
                   # 'hotel_0',
                   # 'office_0',
                   # 'office_1',
                   # 'office_2',
                   # 'office_3',
                   # 'office_4',
                   # 'room_0',
                   # 'room_1',
                   # 'room_2',
                   ]
    from synthesize.get_scene import ReplicaScene, replica_folder
    for scene_name in tqdm(scene_names):
        print(scene_name)
        replica_scene = ReplicaScene(scene_name, replica_folder, build=False, zero_floor=True)
        scene_mesh_yup = deepcopy(replica_scene.mesh)
        scene_mesh_yup.apply_transform(zup_to_shapenet)
        scene_mesh_yup.export(replica_folder / scene_name  / 'mesh_floor_yup.obj')

        export_path = replica_folder / scene_name / 'navmesh_loose.ply'
        navmesh = create_navmesh(scene_mesh_yup, export_path=export_path, agent_radius=0.2, visualize=False)
        export_path = replica_folder / scene_name / 'navmesh_looser.ply'
        navmesh = create_navmesh(scene_mesh_yup, export_path=export_path, agent_radius=0.3, visualize=False)
        export_path = replica_folder / scene_name / 'navmesh_tight.ply'
        navmesh_tight = create_navmesh(scene_mesh_yup, export_path=export_path, agent_radius=0.01, visualize=False)

        navmesh = trimesh.load(export_path, force='mesh')
        start = trimesh.sample.sample_surface_even(navmesh, 1)[0]
        finish = trimesh.sample.sample_surface_even(navmesh, 1)[0]
        path = path_find(navmesh, start, finish)
        print(path.shape)



def test_random(visualize=False, find_path=True, surround_object=True, discard_simple=False):
    shapenet_dir = Path('data/shapenet_real')
    output_dir = Path('data/scenes/random_scene')
    output_dir.mkdir(exist_ok=True, parents=True)
    extents = (np.random.uniform(2, 7), np.random.uniform(2, 7))
    obj_num_range_dict = {'Armchairs': 4, 'StraightChairs': 4, 'Beds':1, 'Sofas':2, 'L-Sofas':1, 'Desks':2, 'Tables': 2}
    obj_paths = []
    for category in obj_num_range_dict:
        obj_path_list = np.array(list(shapenet_dir.glob('{}/*/*.obj'.format(category))))
        obj_num = np.random.randint(0, obj_num_range_dict[category] + 1)
        obj_ids = np.random.choice(len(obj_path_list), obj_num)
        obj_paths = obj_paths + obj_path_list[obj_ids].tolist()
    if len(obj_paths) == 0:
        print('no object sampled')
        return

    scene_meshes = []
    for obj_path in obj_paths:
        print(obj_path)
        obj_mesh = trimesh.load(obj_path, force='mesh')
        obj_mesh.apply_transform(shapenet_to_zup)
        theta = torch.pi * 2 * torch.FloatTensor(1).uniform_()
        random_rotz = pytorch3d.transforms.euler_angles_to_matrix(torch.FloatTensor([0, 0, theta]).reshape(1, 3),
                                                                  convention="XYZ").detach().cpu().numpy()
        transform = np.eye(4)
        transform[:3, :3] = random_rotz.reshape((3, 3))
        transform[0, 3] = 0.5 * np.random.uniform(-extents[0], extents[0]) - obj_mesh.centroid[0]
        transform[1, 3] = 0.5 * np.random.uniform(-extents[1], extents[1]) - obj_mesh.centroid[1]
        obj_mesh.apply_transform(transform)
        scene_meshes.append(obj_mesh)
    obj_bounds = [obj_mesh.bounds for obj_mesh in scene_meshes]
    scene_bounds = np.copy(trimesh.util.concatenate(scene_meshes).bounds)
    scene_bounds[0, :2] -= 1.2
    scene_bounds[1, :2] += 1.2
    scene_bounds[:, 2] = 0
    extents, transform = trimesh.bounds.to_extents(scene_bounds)
    floor_mesh = trimesh.creation.box(extents=extents,
                                      transform=transform)
    scene_meshes.append(floor_mesh)
    if visualize:
        scene = pyrender.Scene()
        for obj_mesh in scene_meshes:
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh))
        pyrender.Viewer(scene, use_raymond_lighting=True)

    scene_mesh = trimesh.util.concatenate(scene_meshes)
    scene_cfg = '{:.2f}_{:.2f}_{}_{}'.format(extents[0], extents[1], obj_num, time.time())
    scene_mesh.export(output_dir / (scene_cfg + '.ply'))
    scene_mesh.apply_transform(zup_to_shapenet)
    export_path = output_dir / (scene_cfg + '_navmesh_loose.ply')
    navmesh = create_navmesh(scene_mesh, export_path, agent_radius=0.2, visualize=visualize)
    export_path = output_dir / (scene_cfg + '_navmesh_tight.ply')
    navmesh_tight = create_navmesh(scene_mesh, export_path, agent_radius=0.05, visualize=visualize)
    sample_pairs = sample_navmesh(navmesh, obj_bounds, num_samples=100, visualize=visualize, find_path=find_path,
                                  surround_object=surround_object, discard_simple=discard_simple)
    export_path = output_dir / (scene_cfg + '_samples.pkl')
    with open(export_path, 'wb') as f:
        pickle.dump(sample_pairs, f)

def test_random_test(visualize=False, find_path=True):
    shapenet_dir = Path('data/shapenet_real')
    output_dir = Path('data/scenes/random_scene_test')
    output_dir.mkdir(exist_ok=True)
    extents = (np.random.uniform(2, 7), np.random.uniform(2, 7))
    obj_num_range_dict = {'Armchairs': 4, 'StraightChairs': 4, 'Beds':1, 'Sofas':2, 'L-Sofas':1, 'Desks':2, 'Tables': 2}
    obj_paths = []
    for category in obj_num_range_dict:
        obj_path_list = np.array(list(shapenet_dir.glob('{}/*/*.obj'.format(category))))
        obj_num = np.random.randint(0, obj_num_range_dict[category] + 1)
        obj_ids = np.random.choice(len(obj_path_list), obj_num)
        obj_paths = obj_paths + obj_path_list[obj_ids].tolist()
    if len(obj_paths) == 0:
        print('no object sampled')
        return

    scene_meshes = []
    for obj_path in obj_paths:
        print(obj_path)
        obj_mesh = trimesh.load(obj_path, force='mesh')
        obj_mesh.apply_transform(shapenet_to_zup)
        theta = torch.pi * 2 * torch.FloatTensor(1).uniform_()
        random_rotz = pytorch3d.transforms.euler_angles_to_matrix(torch.FloatTensor([0, 0, theta]).reshape(1, 3),
                                                                  convention="XYZ").detach().cpu().numpy()
        transform = np.eye(4)
        transform[:3, :3] = random_rotz.reshape((3, 3))
        transform[0, 3] = 0.5 * np.random.uniform(-extents[0], extents[0]) - obj_mesh.centroid[0]
        transform[1, 3] = 0.5 * np.random.uniform(-extents[1], extents[1]) - obj_mesh.centroid[1]
        obj_mesh.apply_transform(transform)
        scene_meshes.append(obj_mesh)
    obj_bounds = [obj_mesh.bounds for obj_mesh in scene_meshes]
    scene_bounds = np.copy(trimesh.util.concatenate(scene_meshes).bounds)
    scene_bounds[0, :2] -= 1
    scene_bounds[1, :2] += 1
    scene_bounds[:, 2] = 0
    extents, transform = trimesh.bounds.to_extents(scene_bounds)
    floor_mesh = trimesh.creation.box(extents=extents,
                                      transform=transform)
    scene_meshes.append(floor_mesh)
    if visualize:
        scene = pyrender.Scene()
        for obj_mesh in scene_meshes:
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh))
        pyrender.Viewer(scene, use_raymond_lighting=True)

    scene_mesh = trimesh.util.concatenate(scene_meshes)
    scene_cfg = '{:.2f}_{:.2f}_{}_{}'.format(extents[0], extents[1], obj_num, time.time())
    scene_dir = output_dir / scene_cfg
    scene_dir.mkdir(exist_ok=True, parents=True)
    scene_mesh.export(output_dir / scene_cfg / 'mesh.ply')
    scene_mesh.apply_transform(zup_to_shapenet)
    scene_mesh.export(output_dir / scene_cfg / 'mesh.obj')
    export_path = output_dir / scene_cfg / 'navmesh.ply'
    navmesh = create_navmesh(scene_mesh, export_path, agent_radius=0.2, visualize=visualize)
    export_path = output_dir / scene_cfg / 'navmesh_tight.ply'
    navmesh_tight = create_navmesh(scene_mesh, export_path, agent_radius=0.01, visualize=visualize)
    sample_paths = sample_navmesh(navmesh, obj_bounds, num_samples=10, visualize=visualize, find_path=find_path, return_path=True)
    print(sample_paths)
    for path_idx, path in enumerate(sample_paths):
        export_path = output_dir / scene_cfg / 'path{}.pkl'.format(str(path_idx))
        with open(export_path, 'wb') as f:
            pickle.dump(path, f)
        export_path = output_dir / scene_cfg / 'pair{}_start.json'.format(str(path_idx))
        with open(export_path, 'w') as f:
            json.dump({
                'x': -path[0, 0],
                'y': path[0, 2],
                'z': -path[0, 1]}, f)
        export_path = output_dir / scene_cfg / 'pair{}_target.json'.format(str(path_idx))
        with open(export_path, 'w') as f:
            json.dump({
                'x': -path[-1, 0],
                'y': path[-1, 2],
                'z': -path[-1, 1]}, f)

def sample_navmesh(navmesh, obj_bounds, crop_range=2.0, num_samples=8, visualize=False,
                   find_path=False, surround_object=True, opposite=False, return_path=False, discard_simple=False):
    sample_pairs = []
    sample_paths =[]
    for _ in range(num_samples):
        if surround_object:
            obj_id = np.random.choice(len(obj_bounds))
            obj_bound = deepcopy(obj_bounds[obj_id])
            extents, transform = trimesh.bounds.to_extents(obj_bound)
            extents = extents + np.array([crop_range, crop_range, crop_range])
            crop_box = trimesh.creation.box(extents=extents,
                                              transform=transform)
            navmesh_crop = navmesh.slice_plane(crop_box.facets_origin, -crop_box.facets_normal)
            if len(navmesh_crop.vertices) == 0:
                continue
            obj_center = 0.5 * (obj_bound[0] + obj_bound[1])
            theta = np.pi * np.random.uniform() * 2
            random_dir = np.array([np.cos(theta), np.sin(theta), 0])
            crop_half1 = navmesh_crop.slice_plane(obj_center, random_dir)
            crop_half2 = navmesh_crop.slice_plane(obj_center, -random_dir)
            if len(crop_half1.vertices) == 0:
                continue
            if len(crop_half2.vertices) == 0:
                continue
            start = np.array(trimesh.sample.sample_surface_even(crop_half1, 1)[0])
            target = -start if opposite else np.array(trimesh.sample.sample_surface_even(crop_half2, 1)[0])
        else:
            start = np.array(trimesh.sample.sample_surface_even(navmesh, 1)[0])
            target = np.array(trimesh.sample.sample_surface_even(navmesh, 1)[0])
        if visualize:
            scene = pyrender.Scene()
            scene.add(pyrender.Mesh.from_trimesh(navmesh))
            if surround_object:
                crop_half1.visual.vertex_colors = np.array([255, 0, 0, 255])
                crop_half2.visual.vertex_colors = np.array([0, 255, 0, 255])
                scene.add(pyrender.Mesh.from_trimesh(crop_half1))
                scene.add(pyrender.Mesh.from_trimesh(crop_half2))
            for marker in (start, target):
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = marker + np.array([0, 0, 0.2])
                sm = trimesh.creation.uv_sphere(radius=0.10)
                sm.visual.vertex_colors = [1.0, 1.0, 0.0]
                sm.apply_transform(trans_mat)
                scene.add(pyrender.Mesh.from_trimesh(sm))
            pyrender.Viewer(scene, use_raymond_lighting=True)
        if find_path:
            path = path_find(navmesh, start, target, visualize=visualize)
            if discard_simple and len(path) <= 3:
                continue
            for idx in range(len(path) - 1):
                sample_pairs.append((path[idx], path[idx + 1]))
            if len(path) > 1:
                sample_paths.append(path)
        else:
            sample_pairs.append((start, target))

    if return_path:
        return sample_paths
    else:
        return sample_pairs

def test_random_obstacle(visualize=False, find_path=True):
    shapenet_dir = Path('data/shapenet_real')
    output_dir = Path('data/scenes/random_scene_obstacle')
    output_dir.mkdir(exist_ok=True)
    extents = (np.random.uniform(2, 7), np.random.uniform(2, 7))
    obj_paths = list(shapenet_dir.glob('Armchairs/*/*.obj')) + list(shapenet_dir.glob('StraightChairs/*/*.obj'))
    if len(obj_paths) == 0:
        print('no object sampled')
        return

    scene_meshes = []
    obj_path = random.choice(obj_paths)
    print(obj_path)
    obj_mesh = trimesh.load(obj_path, force='mesh')
    obj_mesh.apply_transform(shapenet_to_zup)
    theta = torch.pi * 2 * torch.FloatTensor(1).uniform_()
    random_rotz = pytorch3d.transforms.euler_angles_to_matrix(torch.FloatTensor([0, 0, theta]).reshape(1, 3),
                                                              convention="XYZ").detach().cpu().numpy()
    transform = np.eye(4)
    transform[:3, :3] = random_rotz.reshape((3, 3))
    transform[0, 3] = 0.5 * np.random.uniform(-extents[0], extents[0]) - obj_mesh.centroid[0]
    transform[1, 3] = 0.5 * np.random.uniform(-extents[1], extents[1]) - obj_mesh.centroid[1]
    obj_mesh.apply_transform(transform)
    scene_meshes.append(obj_mesh)
    obj_bounds = [obj_mesh.bounds for obj_mesh in scene_meshes]
    scene_bounds = np.copy(trimesh.util.concatenate(scene_meshes).bounds)
    scene_bounds[0, :2] -= 1
    scene_bounds[1, :2] += 1
    scene_bounds[:, 2] = 0
    extents, transform = trimesh.bounds.to_extents(scene_bounds)
    floor_mesh = trimesh.creation.box(extents=extents,
                                      transform=transform)
    scene_meshes.append(floor_mesh)
    if visualize:
        scene = pyrender.Scene()
        for obj_mesh in scene_meshes:
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh))
        pyrender.Viewer(scene, use_raymond_lighting=True)

    scene_mesh = trimesh.util.concatenate(scene_meshes)
    scene_cfg = '{:.2f}_{:.2f}_{}_{}'.format(extents[0], extents[1], 1, time.time())
    scene_mesh.export(output_dir / (scene_cfg + '.ply'))
    scene_mesh.apply_transform(zup_to_shapenet)
    export_path = output_dir / (scene_cfg + '_navmesh.ply')
    navmesh = create_navmesh(scene_mesh, export_path, agent_radius=0.2, visualize=visualize)
    export_path = output_dir / (scene_cfg + '_navmesh_tight.ply')
    navmesh_tight = create_navmesh(scene_mesh, export_path, agent_radius=0.01, visualize=visualize)
    sample_pairs = sample_navmesh(navmesh, obj_bounds, crop_range=1.0, num_samples=6, visualize=visualize, find_path=False)
    export_path = output_dir / (scene_cfg + '_samples.pkl')
    with open(export_path, 'wb') as f:
        pickle.dump(sample_pairs, f)

def test_box_obstacle(visualize=False):
    output_dir = Path('data/scenes/random_box_obstacle_far')
    output_dir.mkdir(exist_ok=True)
    extents = np.array([np.random.uniform(0.5, 1), np.random.uniform(0.5, 1), 2])

    scene_meshes = []
    transform = np.eye(4)
    transform[2, 3] = 1
    obj_mesh = trimesh.creation.box(
        extents=extents,
        transform=transform
    )
    theta = torch.pi * 2 * torch.FloatTensor(1).uniform_()
    random_rotz = pytorch3d.transforms.euler_angles_to_matrix(torch.FloatTensor([0, 0, theta]).reshape(1, 3),
                                                              convention="XYZ").detach().cpu().numpy()
    transform = np.eye(4)
    transform[:3, :3] = random_rotz.reshape((3, 3))
    obj_mesh.apply_transform(transform)
    scene_meshes.append(obj_mesh)
    obj_bounds = [obj_mesh.bounds for obj_mesh in scene_meshes]
    scene_bounds = np.copy(trimesh.util.concatenate(scene_meshes).bounds)
    scene_bounds[0, :2] -= 5
    scene_bounds[1, :2] += 5
    scene_bounds[:, 2] = 0
    extents, transform = trimesh.bounds.to_extents(scene_bounds)
    floor_mesh = trimesh.creation.box(extents=extents,
                                      transform=transform)
    scene_meshes.append(floor_mesh)
    if visualize:
        scene = pyrender.Scene()
        for obj_mesh in scene_meshes:
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh))
        pyrender.Viewer(scene, use_raymond_lighting=True)

    scene_mesh = trimesh.util.concatenate(scene_meshes)
    scene_cfg = '{:.2f}_{:.2f}_{}_{}'.format(extents[0], extents[1], 1, time.time())
    scene_mesh.export(output_dir / (scene_cfg + '.ply'))
    scene_mesh.apply_transform(zup_to_shapenet)
    export_path = output_dir / (scene_cfg + '_navmesh.ply')
    navmesh = create_navmesh(scene_mesh, export_path, agent_radius=0.4, visualize=visualize)
    export_path = output_dir / (scene_cfg + '_navmesh_tight.ply')
    navmesh_tight = create_navmesh(scene_mesh, export_path, agent_radius=0.01, visualize=visualize)
    sample_pairs = sample_navmesh(navmesh, obj_bounds, crop_range=2, num_samples=10, visualize=visualize,
                                  find_path=False, opposite=True)

    export_path = output_dir / (scene_cfg + '_samples.pkl')
    with open(export_path, 'wb') as f:
        pickle.dump(sample_pairs, f)

def sample_replica():
    navmesh_path = '/mnt/atlas_root/vlg-nfs/kaizhao/datasets/replica/room_0/navmesh_loose.ply'
    navmesh = trimesh.load(navmesh_path, force='mesh')
    sample_pairs = sample_navmesh(navmesh, None, num_samples=100, visualize=False, surround_object=False, find_path=True)
    export_path = '/mnt/atlas_root/vlg-nfs/kaizhao/datasets/replica/room_0/samples.pkl'
    with open(export_path, 'wb') as f:
        pickle.dump(sample_pairs, f)

if __name__ == '__main__':
    for _ in tqdm(range(100)):
        try:
            test_random(visualize=False, find_path=True, surround_object=False, discard_simple=False)
        except Exception as e:
            print(e)

    export_dir = Path('data/scenes/random_scene')
    file_names = [path.name for path in export_dir.iterdir()]
    scene_names = ['.'.join(name.split('.')[:-1]) for name in file_names if not 'navmesh' in name and not 'samples' in name]
    valid_names = []
    for name in scene_names:
        path1 = export_dir / (name + '.ply')
        path2 = export_dir / (name + '_navmesh_loose.ply')
        path3 = export_dir / (name + '_navmesh_tight.ply')
        path4 = export_dir / (name + '_samples.pkl')
        if path1.exists() and path2.exists() and path3.exists() and path4.exists():
            valid_names.append(name)
    print(valid_names)
    with open(Path('data/scenes/random_scene_names.pkl'), 'wb') as f:
        pickle.dump(valid_names, f)