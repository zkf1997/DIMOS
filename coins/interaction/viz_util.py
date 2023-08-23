import sys
sys.path.append('..')

import os.path as osp
import open3d as o3d
import torch
import numpy as np
import pandas as pd
import trimesh
import pyrender
import PIL.Image as pil_img
from interaction import eulerangles

def create_renderer(H=1080, W=1920, intensity=50, fov=None, point_size=1.0):
    if fov is None:
        fov = np.pi / 3.0
    r = pyrender.OffscreenRenderer(viewport_width=W,
                                   viewport_height=H,
                                   point_size=point_size)
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.333)
    light_directional = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
    light_point = pyrender.PointLight(color=np.ones(3), intensity=intensity)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    return r, camera, light_directional, light_point, material

def create_collage(images, mode='grid'):
    n = len(images)
    W, H = images[0].size
    if mode == 'grid':
        img_collage = pil_img.new('RGB', (2 * W, 2 * H))
        for id, img in enumerate(images):
            img_collage.paste(img, (W * (id % 2), H * int(id / 2)))
    elif mode == 'vertical':
        img_collage = pil_img.new('RGB', (W, n * H))
        for id, img in enumerate(images):
            img_collage.paste(img, (0, id * H))
    elif mode == 'horizantal':
        img_collage = pil_img.new('RGB', (n * W, H))
        for id, img in enumerate(images):
            img_collage.paste(img, (id * W, 0))
    return img_collage

def render_interaction_multview(body, static_scene, clothed_body=None, use_clothed_mesh=False, body_center=True, smooth_body=True,
                                collage_mode='grid', body_contrast=None, obj_points_coord=None, num_view=4, **kwargs):
    H, W = int(480 * 1.5), int(640 * 1.5)
    renderer, camera, light_directional, light_point, material = create_renderer(H=H, W=W, intensity=2.0)
    light_point.intensity = 10.0

    # this will make the camera looks in the -x direction
    camera_pose = np.eye(4)
    camera_pose[0, 3] = 2
    camera_pose[2, 3] = 1
    camera_pose[:3, :3] = eulerangles.euler2mat(-np.pi / 6, np.pi / 2, np.pi / 2, 'sxzy')

    if body_center:
        center = (body.vertices.max(axis=0) + body.vertices.min(axis=0)) / 2.0
        # camera_pose[0, 3] = 0.5
        # camera_pose[2, 3] = 2
    else:
        center = (static_scene.vertices.max(axis=0) + static_scene.vertices.min(axis=0)) / 2.0
        camera_pose[0, 3] = 3

    static_scene.vertices -= center
    body.vertices -= center
    if use_clothed_mesh:
        clothed_body.vertices -= center
    if body_contrast is not None:
        body_contrast.vertices -= center


    images = []
    views = list(range(0, 360, 90)) if num_view == 4 else [0, 90]
    # for ang_id, ang in enumerate(range(0, 360, 90)):
    for ang_id, ang in enumerate(views):
        ang = np.pi / 180 * ang
        rot_z = np.eye(4)
        rot_z[:3, :3] = eulerangles.euler2mat(ang, 0, 0, 'szxy')

        # print(1)
        static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)
        if use_clothed_mesh:
            body_mesh = pyrender.Mesh.from_trimesh(clothed_body, material=pyrender.MetallicRoughnessMaterial(alphaMode="BLEND",
                                              baseColorFactor=(1.0, 1.0, 1.0, 0.5),
                                              metallicFactor=0.0,))
        else:
            body_mesh = pyrender.Mesh.from_trimesh(body, smooth=smooth_body)

        # print(2)
        scene = pyrender.Scene()
        scene.add(camera, pose=np.matmul(rot_z, camera_pose))
        scene.add(light_point, pose=np.matmul(rot_z, camera_pose))
        scene.add(light_directional, pose=np.eye(4))
        scene.add(static_scene_mesh, 'mesh')
        # print(3)
        if obj_points_coord is not None:
            scene.add(pyrender.Mesh.from_points(points=obj_points_coord - center, colors=(1.0, 0.1, 0.1)), 'mesh')
        # print(4)
        scene.add(body_mesh, 'mesh')
        if body_contrast is not None:
            scene.add(pyrender.Mesh.from_trimesh(body_contrast, material=material), 'mesh')
        # print(5)
        color, _ = renderer.render(scene, pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL
                                   # | pyrender.constants.RenderFlags.SKIP_CULL_FACES
                                   # | pyrender.constants.RenderFlags.VERTEX_NORMALS
                                   )
        # print(6)
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        images.append(img)
    # print(7)
    # collage_mode = 'grid' if num_view == 4 else 'horizantal'
    images = create_collage(images, collage_mode)
    static_scene.vertices += center
    body.vertices += center
    return images


def render_composite_interaction_multview(body, scene_meshes,  body_center=True, use_material=True, smooth_body=True,
                                collage_mode='grid', body_contrast=None, obj_points_coord=None, **kwargs):
    H, W = 720, 960
    renderer, camera, light_directional, light_point, material = create_renderer(H=H, W=W, intensity=2.0)
    light_point.intensity = 10.0

    # assert(len(scene_meshes) == obj_points_coord.shape[0])

    # this will make the camera looks in the -x direction
    camera_pose = np.eye(4)
    camera_pose[0, 3] = 2
    camera_pose[2, 3] = 1
    camera_pose[:3, :3] = eulerangles.euler2mat(-np.pi / 6, np.pi / 2, np.pi / 2, 'sxzy')

    center = (body.vertices.max(axis=0) + body.vertices.min(axis=0)) / 2.0

    for scene_mesh in scene_meshes:
        scene_mesh.vertices -= center
    body.vertices -= center
    if body_contrast is not None:
        body_contrast.vertices -= center

    images = []
    # for ang_id, ang in enumerate(range(0, 360, 90)):
    for ang_id, ang in enumerate(range(0, 360, 90)):
        ang = np.pi / 180 * ang
        rot_z = np.eye(4)
        rot_z[:3, :3] = eulerangles.euler2mat(ang, 0, 0, 'szxy')



        scene = pyrender.Scene()
        scene.add(camera, pose=np.matmul(rot_z, camera_pose))
        scene.add(light_point, pose=np.matmul(rot_z, camera_pose))
        scene.add(light_directional, pose=np.eye(4))
        for scene_mesh in scene_meshes:
            static_scene_mesh = pyrender.Mesh.from_trimesh(scene_mesh)
            scene.add(static_scene_mesh, 'mesh')
        if obj_points_coord is not None:
            for obj_idx in range(obj_points_coord.shape[0]):
                scene.add(pyrender.Mesh.from_points(points=obj_points_coord[obj_idx, :, :] - center, colors=(1.0, 0.1, 0.1)), 'mesh')
        body_mesh = pyrender.Mesh.from_trimesh(body, material=material, smooth=smooth_body) if use_material else pyrender.Mesh.from_trimesh(body, smooth=smooth_body)
        scene.add(body_mesh, 'mesh')
        if body_contrast is not None:
            scene.add(pyrender.Mesh.from_trimesh(body_contrast, material=material, smooth=smooth_body) if use_material else pyrender.Mesh.from_trimesh(body_contrast, smooth=smooth_body), 'mesh')

        color, _ = renderer.render(scene, pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL
                                   # | pyrender.constants.RenderFlags.SKIP_CULL_FACES
                                   )
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        images.append(img)
    images = create_collage(images, collage_mode)
    for scene_mesh in scene_meshes:
        scene_mesh.vertices += center
    body.vertices += center
    if body_contrast is not None:
        body_contrast.vertices += center
    return images

def render_body_multview(body, body_center=True, num_view=4, use_material=True,
                                collage_mode='grid', body_contrast=None, **kwargs):
    H, W = 480 * 2, 640 * 2
    renderer, camera, light_directional, light_point, material = create_renderer(H=H, W=W, intensity=2.0)
    light_point.intensity = 10.0

    # this will make the camera looks in the -x direction
    camera_pose = np.eye(4)
    camera_pose[0, 3] = 2
    camera_pose[2, 3] = 1
    camera_pose[:3, :3] = eulerangles.euler2mat(-np.pi / 6, np.pi / 2, np.pi / 2, 'sxzy')

    center = (body.vertices.max(axis=0) + body.vertices.min(axis=0)) / 2.0

    body.vertices -= center
    if body_contrast is not None:
        body_contrast.vertices -= center - np.array((0.5, 0.0, 0.0))

    images = []
    views = list(range(0, 360, 90)) if num_view == 4 else [0, 90]
    # for ang_id, ang in enumerate(range(0, 360, 90)):
    for ang_id, ang in enumerate(views):
        ang = np.pi / 180 * ang
        rot_z = np.eye(4)
        rot_z[:3, :3] = eulerangles.euler2mat(0, 0, ang, 'szxy')



        scene = pyrender.Scene()
        scene.add(camera, pose=np.matmul(rot_z, camera_pose))
        scene.add(light_point, pose=np.matmul(rot_z, camera_pose))
        scene.add(light_directional, pose=np.eye(4))
        body_mesh = pyrender.Mesh.from_trimesh(body, material=material, smooth=True) if use_material else pyrender.Mesh.from_trimesh(body, smooth=True)
        scene.add(body_mesh, 'mesh')
        if body_contrast is not None:
            scene.add(pyrender.Mesh.from_trimesh(body_contrast, material=material, smooth=True) if use_material else pyrender.Mesh.from_trimesh(body_contrast, smooth=True), 'mesh')

        color, _ = renderer.render(scene, pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL
                                   # | pyrender.constants.RenderFlags.SKIP_CULL_FACES
                                   )
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        images.append(img)
    images = create_collage(images, collage_mode)
    body.vertices += center
    if body_contrast is not None:
        body_contrast.vertices += center - np.array((0.5, 0.0, 0.0))
    return images

def render_obj_multview(obj_pointcloud, frame,
                        collage_mode='grid', frame_contrast=None, body=None, **kwargs):
    H, W = 480, 640
    renderer, camera, light_directional, light_point, material = create_renderer(H=H, W=W, intensity=2.0, point_size=5.0)
    light_point.intensity = 10.0

    # this will make the camera looks in the -x direction
    camera_pose = np.eye(4)
    camera_pose[0, 3] = 2
    camera_pose[2, 3] = 1
    camera_pose[:3, :3] = eulerangles.euler2mat(-np.pi / 6, np.pi / 2, np.pi / 2, 'sxzy')

    center = (obj_pointcloud.vertices.max(axis=0) + obj_pointcloud.vertices.min(axis=0)) / 2.0

    obj_pointcloud.vertices -= center
    frame.vertices -= center
    if frame_contrast is not None:
        frame_contrast.vertices -= center
    if body is not None:
        body.vertices -= center

    images = []
    # for ang_id, ang in enumerate(range(0, 360, 90)):
    for ang_id, ang in enumerate(range(0, 360, 90)):
        ang = np.pi / 180 * ang
        rot_z = np.eye(4)
        rot_z[:3, :3] = eulerangles.euler2mat(ang, 0, 0, 'szxy')

        scene = pyrender.Scene()
        scene.add(camera, pose=np.matmul(rot_z, camera_pose))
        scene.add(light_point, pose=np.matmul(rot_z, camera_pose))
        scene.add(light_directional, pose=np.eye(4))
        scene.add(pyrender.Mesh.from_points(points=obj_pointcloud.vertices, colors=obj_pointcloud.colors / 255.0), 'mesh')
        scene.add(pyrender.Mesh.from_trimesh(frame, smooth=False), 'mesh')
        if body is not None:
            body_mesh = pyrender.Mesh.from_trimesh(body, material=pyrender.MetallicRoughnessMaterial(alphaMode="BLEND",
                                              baseColorFactor=(1.0, 1.0, 1.0, 0.5),
                                              metallicFactor=0.0,),
                                                   smooth=False)
            # print('transparent', body_mesh.is_transparent)
            scene.add(body_mesh, 'mesh')
        if frame_contrast is not None:
            scene.add(pyrender.Mesh.from_trimesh(frame_contrast, smooth=False), 'mesh')

        color, _ = renderer.render(scene, pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL
                                   | pyrender.constants.RenderFlags.RGBA
                                   # | pyrender.constants.RenderFlags.SKIP_CULL_FACES
                                   )
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        images.append(img)
    images = create_collage(images, collage_mode)
    obj_pointcloud.vertices += center
    frame.vertices += center
    if frame_contrast is not None:
        frame_contrast.vertices += center
    if body is not None:
        body.vertices += center
    return images

# https://www.programcreek.com/python/?code=stemkoski%2Fthree.py%2Fthree.py-master%2Fthree.py%2Fmathutils%2FMatrixFactory.py
def makeLookAt(position, target, up):

    forward = np.subtract(target, position)
    forward = np.divide(forward, np.linalg.norm(forward))

    right = np.cross(forward, up)

    # if forward and up vectors are parallel, right vector is zero;
    #   fix by perturbing up vector a bit
    if np.linalg.norm(right) < 0.001:
        epsilon = np.array([0.001, 0, 0])
        right = np.cross(forward, up + epsilon)

    right = np.divide(right, np.linalg.norm(right))

    up = np.cross(right, forward)
    up = np.divide(up, np.linalg.norm(up))

    return np.array([[right[0], up[0], -forward[0], position[0]],
                     [right[1], up[1], -forward[1], position[1]],
                     [right[2], up[2], -forward[2], position[2]],
                     [0, 0, 0, 1]])

def render_scene_three_view(scene_mesh, body_mesh, collage_mode='grid', center='human', render_points=None, **kwargs):
    H, W = 480, 640
    renderer, camera, light_directional, light_point, material = create_renderer(H=H, W=W, intensity=2.0, point_size=10.0)
    light_point.intensity = 10.0

    center = (body_mesh.vertices.max(axis=0) + body_mesh.vertices.min(axis=0)) / 2.0 if center == 'human' else (scene_mesh.vertices.max(axis=0) + scene_mesh.vertices.min(axis=0)) / 2.0
    scene_mesh.vertices -= center
    body_mesh.vertices -= center
    dist = max(np.absolute(body_mesh.vertices).max() + 1, 1.5) if center == 'human' else max(np.absolute(scene_mesh.vertices).max() + 1, 1.5)

    camera_poses = [
        makeLookAt(np.array([dist, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 1])),
        makeLookAt(np.array([0, dist, 0]), np.array([0, 0, 0]), np.array([0, 0, 1])),
        makeLookAt(np.array([0, 0, dist]), np.array([0, 0, 0]), np.array([-1, 0, 0])),
    ]

    images = []
    # for ang_id, ang in enumerate(range(0, 360, 90)):
    for camera_pose in camera_poses:

        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        scene.add(light_point, pose=camera_pose)
        scene.add(light_directional, pose=np.eye(4))
        for mesh in [scene_mesh, body_mesh]:
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial(alphaMode="BLEND",
                                              baseColorFactor=(1.0, 1.0, 1.0, 1.0),
                                              metallicFactor=0.0,), smooth=False), 'mesh')
        if render_points is not None:
            points_mesh = pyrender.Mesh.from_points(render_points.vertices - center, render_points.colors / 255.0)
            # points_mesh.material = pyrender.MetallicRoughnessMaterial(alphaMode="BLEND",
            #                                   baseColorFactor=(1.0, 1.0, 1.0, 1.0),
            #                                   metallicFactor=0.0,)
            # print(points_mesh.primitives[0].color_0)
            # print(points_mesh.is_transparent)
            scene.add(points_mesh, 'mesh')

        color, _ = renderer.render(scene, pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL
                                   | pyrender.constants.RenderFlags.RGBA
                                   | pyrender.constants.RenderFlags.SKIP_CULL_FACES
                                   )
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        images.append(img)
    images = create_collage(images, collage_mode)
    scene_mesh.vertices += center
    body_mesh.vertices += center
    return images

def render_alignment_three_view(scene_mesh, shapenet_mesh, collage_mode='grid', center='human', render_points=None, **kwargs):
    H, W = 480, 640
    renderer, camera, light_directional, light_point, material = create_renderer(H=H, W=W, intensity=2.0, point_size=10.0)
    light_point.intensity = 2.0

    center = (scene_mesh.vertices.max(axis=0) + scene_mesh.vertices.min(axis=0)) / 2.0
    scene_mesh.vertices -= center
    shapenet_mesh.vertices -= center
    dist = max(np.absolute(scene_mesh.vertices).max() + 1, 1.5)

    camera_poses = [
        makeLookAt(np.array([dist, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 1])),
        makeLookAt(np.array([0, dist, 0]), np.array([0, 0, 0]), np.array([0, 0, 1])),
        makeLookAt(np.array([0, 0, dist]), np.array([0, 0, 0]), np.array([-1, 0, 0])),
    ]

    images = []
    # for ang_id, ang in enumerate(range(0, 360, 90)):
    for camera_pose in camera_poses:

        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        scene.add(light_point, pose=camera_pose)
        scene.add(light_directional, pose=np.eye(4))
        for mesh in [scene_mesh, shapenet_mesh]:
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), 'mesh')

        color, _ = renderer.render(scene, pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL
                                   | pyrender.constants.RenderFlags.RGBA
                                   | pyrender.constants.RenderFlags.SKIP_CULL_FACES
                                   )
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        images.append(img)
    images = create_collage(images, collage_mode)
    scene_mesh.vertices += center
    shapenet_mesh.vertices += center
    return images