"""open primitive.blend, run script render"""
import bpy
import logging
logger = logging.getLogger(__name__)
import mathutils
from mathutils import Vector, Quaternion, Matrix
import pyquaternion as pyquat
import numpy as np
from glob import glob
import pdb
import os
from pathlib import  Path
import sys
import math
import pickle

def aa2quaternion(aa):
    rod = Vector((aa[0], aa[1], aa[2]))
    angle_rad = rod.length
    axis = rod.normalized()
    return Quaternion(axis, angle_rad)

def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

def convert_str_direction_to_vector(direction):
    return {
      "X": np.array([1., 0., 0.], dtype=np.float64),
      "Y": np.array([0., 1., 0.], dtype=np.float64),
      "Z": np.array([0., 0., 1.], dtype=np.float64),
      "-X": np.array([-1., 0., 0.], dtype=np.float64),
      "-Y": np.array([0., -1., 0.], dtype=np.float64),
      "-Z": np.array([0., 0., -1.], dtype=np.float64),
    }[direction.upper()]

def normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x)
    if norm < eps:
        pdb.set_trace()
    return x / norm

def cam_rot_update(position, target, y_rot, up='Y', front='-Z'):
    if isinstance(up, str):
        up = normalize(convert_str_direction_to_vector(up))
    if isinstance(front, str):
        front = normalize(convert_str_direction_to_vector(front))
    right = np.cross(up, front)
    look_at_front = normalize(target - position)
    look_at_right = normalize(np.cross(y_rot, look_at_front))
    look_at_up = normalize(np.cross(look_at_front, look_at_right))
    rotation_matrix1 = np.stack([look_at_right, look_at_up, look_at_front])
    rotation_matrix2 = np.stack([right, up, front])
    rot_euler = Quaternion(pyquat.Quaternion(matrix=(rotation_matrix1.T @ rotation_matrix2))).to_euler()
    return rot_euler

# https://blender.stackexchange.com/a/100442/66171
def update_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=10.0):
    """
    Focus the camera to a focus point and place the camera at a specific distance from that
    focus point. The camera stays in a direct line with the focus point.

    :param camera: the camera object
    :type camera: bpy.types.object
    :param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
    :type focus_point: mathutils.Vector
    :param distance: the distance to keep to the focus point (default=``10.0``)
    :type distance: float
    """
    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    # Use * instead of @ for Blender <2.8
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))

def add_material_target(objname, color):
    mat = bpy.data.materials.new(objname)
    # # Add material slot to parachute mesh object (currently active)
    bpy.ops.object.material_slot_add()
    # Assign the material to the new material slot (currently active)
    obj = bpy.data.objects[objname]
    obj.active_material = mat
    # assign the texture
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value=(color[0],color[1],color[2],1.0)
    bsdf.inputs[5].default_value=0
    #bsdf.inputs[17].default_value=(0.8,0.503,0.009,1.0)
#    bsdf.inputs[17].default_value=(color[0],color[1],color[2],1.0)
    bsdf.inputs[17].default_value=0.8


def render_walk_points(wpath, collection_name=None):
    # color = np.random.rand(3)
    color = np.array([0, 0, 1.0])
    for pt in wpath:
        bpy.ops.mesh.primitive_torus_add()
        torus = bpy.context.object
        torus.scale = Vector((0.2, 0.2, 0.1))
        torus.location = Vector((pt[0], pt[1], pt[2]))
        add_material_target(torus.name, color=color)

        if collection_name is not None:
            collection = bpy.data.collections[collection_name + '_targets']
            collection.objects.link(torus)  # link it with collection
            master_collection = bpy.context.scene.collection
            master_collection.objects.unlink(torus)  # unlink it from master collection

# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####


import bpy
import bmesh
from bpy_extras.io_utils import ImportHelper, \
    ExportHelper  # ImportHelper/ExportHelper is a helper class, defines filename and invoke() function which calls the file selector.
import pdb
from mathutils import Vector, Quaternion, Matrix
from math import radians
import numpy as np
import os
import pickle
import random
import glob
# from scipy.spatial.transform import Rotation as R

from bpy.props import (BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty)
from bpy.types import (PropertyGroup)

SMPLX_MODELFILE = "smplx_model_20210421.blend"
SMPLX_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
    'right_elbow', 'left_wrist', 'right_wrist',
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2', 'left_index3',
    'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3',
    'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2',
    'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1',
    'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3',
    'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3'
]  # same to the definition in https://github.com/vchoutas/smplx/blob/master/smplx/joint_names.py

NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15
ROT_NEGATIVE_X = Matrix(np.array([[1.0000000, 0.0000000, 0.0000000],
                                  [0.0000000, 0.0000000, 1.0000000],
                                  [0.0000000, -1.0000000, 0.0000000]])
                        )
ROT_POSITIVE_Y = Matrix(np.array([[-1.0000000, 0.0000000, 0.0000000],
                                  [0.0000000, 1.0000000, 0.0000000],
                                  [0.0000000, 0.0000000, -1.0000000]])
                        )
'''
note
    - rotation in pelvis is in the original smplx coordinate
    - rotation of the armature is in the blender coordinate
'''

FPS_SOURCE = 0
FPS_TARGET = 30
FPS_DOWNSAMPLE = 1


def rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = armature.pose.bones[bone_name].rotation_quaternion
    (axis, angle) = quat.to_axis_angle()
    rodrigues = axis
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues

# Remove default cube
if 'Cube' in bpy.data.objects:
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()


def animate_smplx_one_primitive(armature, scene, data, frame):
    smplx_params = data['smplx_params'][0]
    pelvis_locs = data['pelvis_loc'][0]
    transf_rotmat = Matrix(data['transf_rotmat'].reshape(3, 3))
    transf_transl = Vector(data['transf_transl'].reshape(3))
    n_frames_per_mp = 10
    if frame == 0:
        ss = 0
    elif data['mp_type'] == '1-frame':
        ss = 1
    elif data['mp_type'] == '2-frame':
        ss = 2
    elif data['mp_type'] == 'start-frame':
        ss = 0
        print('start_frame')
    elif data['mp_type'] == 'target-frame':
        n_frames_per_mp = 1
    elif data['mp_type'] == 'humor':
        n_frames_per_mp = 300
        ss = 0

    for t in range(ss, n_frames_per_mp):
        print('|-- processing frame {}...'.format(frame), end='\r')
        scene.frame_set(frame)
        transl = pelvis_locs[t].reshape(3)
        global_orient = np.array(smplx_params[t][3:6]).reshape(3)
        body_pose = np.array(smplx_params[t][6:69]).reshape(63).reshape(NUM_SMPLX_BODYJOINTS, 3)

        # Update body pose
        for index in range(NUM_SMPLX_BODYJOINTS):
            pose_rodrigues = body_pose[index]
            bone_name = SMPLX_JOINT_NAMES[index + 1]  # body pose starts with left_hip
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

        # set global configurations
        ## set global orientation and translation at local coodinate
        if global_orient is not None:
            armature.rotation_mode = 'QUATERNION'
            global_orient_w = transf_rotmat @ (aa2quaternion(global_orient).to_matrix())
            armature.rotation_quaternion = global_orient_w.to_quaternion()

        if transl is not None:
            transl_w = transf_rotmat @ Vector(transl) + transf_transl
            armature.location = transl_w

        # Activate corrective poseshapes (update pose blend shapes)
        bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

        # set the current status to a keyframe for animation
        armature.keyframe_insert('location', frame=frame)
        armature.keyframe_insert('rotation_quaternion', frame=frame)
        bones = armature.pose.bones
        for bone in bones:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
        frame += 1

    return frame

def animate_smplx(filepath, render_wpath=False, debug=0):
    print()
    print()

    '''create a new collection for the body and the target'''
    collection_name = "motions_{:05d}".format(random.randint(0, 1000))
    collection_motion = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection_motion)
    collection_targets = bpy.data.collections.new(collection_name + '_targets')
    collection_motion.children.link(collection_targets)

    '''read search results'''
    with open(filepath, "rb") as f:
        dataall = pickle.load(f, encoding="latin1")
    print('read files and setup global info...')
    motiondata = dataall['motion']
    wpath = dataall['wpath']
    if render_wpath:
        render_walk_points(wpath, collection_name)

    '''set keyframe range'''
    scene = bpy.data.scenes['Scene']
    scene.render.fps = FPS_TARGET
    scene.frame_end = 9 * len(motiondata)

    '''add a smplx into blender context'''
    gender = str(motiondata[0]['gender'])
    bpy.data.window_managers['WinMan'].smplx_tool.smplx_gender = gender
    bpy.data.window_managers['WinMan'].smplx_tool.smplx_texture = 'smplx_texture_m_alb.png' if gender == 'male' else 'smplx_texture_f_alb.png'
    bpy.ops.scene.smplx_add_gender()

    '''set global variables'''
    obj = bpy.context.object
    if obj.type == 'MESH':
        armature = obj.parent
    else:
        armature = obj
        obj = armature.children[0]
        bpy.context.view_layer.objects.active = obj  # mesh needs to be active object for recalculating joint locations
    bpy.ops.object.smplx_set_texture()  # context needs to be mesh

    print('animate character: {}'.format(obj.name))
    collection_motion.objects.link(armature)  # link it with collection
    collection_motion.objects.link(obj)  # link it with collection
    bpy.context.scene.collection.objects.unlink(armature)  # unlink it from master collection
    bpy.context.scene.collection.objects.unlink(obj)  # unlink it from master collection

    '''update the body shape according to beta'''
    betas = np.array(motiondata[0]["betas"]).reshape(-1).tolist()
    bpy.ops.object.mode_set(mode='OBJECT')
    for index, beta in enumerate(betas):
        key_block_name = f"Shape{index:03}"
        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = beta
        else:
            print(f"ERROR: No key block for: {key_block_name}")
    ## Update joint locations. This is necessary in this add-on when applying body shape.
    bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')
    print('|-- shape updated...')
    bpy.ops.object.smplx_set_texture()  # context needs to be mesh

    '''move the origin to the body pelvis, and rotate around x by -90degree'''
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    deltaT = armature.pose.bones['pelvis'].head.z  # the position at pelvis
    bpy.ops.object.mode_set(mode='POSE')
    armature.pose.bones['pelvis'].location.y -= deltaT
    armature.pose.bones['pelvis'].rotation_quaternion = ROT_NEGATIVE_X.to_quaternion()
    bpy.ops.object.mode_set(mode='OBJECT')

    '''update the body pose'''
    transl = None
    global_orient = None
    body_pose = None
    jaw_pose = None
    left_hand_pose = None
    right_hand_pose = None
    expression = None

    '''main loop to update body pose and insert keyframes'''
    frame = 0
    for data in motiondata:
        frame = animate_smplx_one_primitive(armature, scene, data, frame)

class Blender:
    def __init__(self, samples=128, custom_scene=None, engine='cycles'):
        self.samples = samples
        # self.mesh_name = custom_scene
        # self.mesh_name = os.path.basename(self.mesh_name).split('.')[0]
        self.clear_and_reset_blender_scene(custom_scene)
        if engine == 'cycles':
            self.set_gpu()
        elif engine == 'eevee':
            bpy.context.scene.render.engine = "BLENDER_EEVEE"
        elif engine == 'workbench':
            bpy.context.scene.render.engine = "BLENDER_WORKBENCH"

    def clear_and_reset_blender_scene(self, custom_scene=None, scene_mesh=None):
        # Resets Blender to an entirely empty scene (or a custom one)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        logger.info("Loading scene from '%s'", custom_scene)
        bpy.ops.wm.open_mainfile(filepath=custom_scene)

    def set_gpu(self):
        # by default we use gpu to render. it is a pity if you do not have one.
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = self.samples
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        devices_used = [d.name for d in bpy.context.preferences.addons["cycles"].preferences.devices
                        if d.use]
        logger.info("Using the following GPU Device(s): %s", devices_used)

file_path = 'results_ssm2_67_condi_marker_inter_0.pkl'
animate_smplx(filepath=file_path, render_wpath=False, debug=0)