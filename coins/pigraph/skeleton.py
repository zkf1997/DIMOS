import json

import numpy as np
import open3d as o3d
import smplx
import torch
from scipy.spatial.transform import Rotation

from load_human import load_human
from pigraph_config import *

DEBUG = False
trans = None

# use smplx inputs, joints layout: https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/joint_names.py#L17
# we use the first 22 joints
JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]
# print(len(JOINT_NAMES))
joint_name_to_idx = {}
for idx, joint_name in enumerate(JOINT_NAMES):
    joint_name_to_idx[joint_name] = idx

bones = [
    ['pelvis', 'left_hip'],
    ['pelvis', 'right_hip'],
    ['pelvis', 'neck'],
    ['left_knee', 'left_hip'],
    ['right_knee', 'right_hip'],
    ['left_knee', 'left_ankle'],
    ['right_knee', 'right_ankle'],
    ['left_foot', 'left_ankle'],
    ['right_foot', 'right_ankle'],
    ['neck', 'head'],
    ['neck', 'left_shoulder'],
    ['neck', 'right_shoulder'],
    ['left_elbow', 'left_shoulder'],
    ['right_elbow', 'right_shoulder'],
    ['left_elbow', 'left_wrist'],
    ['right_elbow', 'right_wrist']
]

parent_joint_idx = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
        16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
        35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52,
        53]  # smplx kinematic chain as index of parent joint

# calculate lookat matrix, from three.py
# modified the axis signs
def makeLookAt(position, target, up=np.array([0, 0, 1])):
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

    return np.array([[right[0], -up[0], forward[0], position[0]],
                     [right[1], -up[1], forward[1], position[1]],
                     [right[2], -up[2], forward[2], position[2]],
                     [0, 0, 0, 1]])

# skeleton of joints, use smplx as input, have absolute positions for first 127 joints, and relative orientations for first 55 joints
class Skeleton:
    def __init__(self, num_joints=NUM_JOINTS, positions=None, relative_orientations=None, orientations=None, relative_positions=None, transform=None, gender='male', shape=np.zeros((1, 10))):
        # joints: {positions, orientations}, use axis-angle orientation
        self.num_joints = NUM_JOINTS  # number of joints
        self.positions = positions  # NUM_JOINTS * 3, absolute world frame position coords
        self.relative_orientations = relative_orientations  # NUM_JOINTS * 3, orientation relative to parent joint, for pelvis is the global orientation relative to world coords, local to parent
        self.orientations = orientations  #NUM_JOINTS * 3, orientation relative to world frame
        self.relative_positions = relative_positions  # NUM_JOINTS * 3, position relative to parent joint
        self.gender = gender
        self.shape = shape.reshape((1, 10))
        if self.orientations is None:
            self.orientations = self.calc_orientations()
        if self.relative_positions is None:
            self.relative_positions = self.calc_relative_positions()
        if self.positions is None:
            self.positions = self.calc_positions()
        self.center_of_mass = self.get_com()
        if transform is not None:
            self.transform(transform)

    def calc_orientations(self):
        num_orientations = self.num_joints
        rots = []
        for idx in range(num_orientations):
            rot = Rotation.from_rotvec(self.relative_orientations[idx])
            parent = parent_joint_idx[idx]
            if parent != -1:
                rot = rot * rots[parent]
            rots.append(rot)
        return np.asarray([rot.as_rotvec() for rot in rots])

    def calc_relative_positions(self):
        num_relative_pos = self.num_joints
        relative_pos = np.zeros((num_relative_pos, 3))
        for idx in range(num_relative_pos):
            parent = parent_joint_idx[idx]
            if parent == -1:
                relative_pos[idx, :] = self.positions[idx]
            else:
                rot = Rotation.from_rotvec(self.orientations[parent])
                relative_pos[idx, :] = rot.apply(self.positions[idx] - self.positions[parent])

        return relative_pos

    # calculate absolute positions from relative positions and absolute orientations
    def calc_positions(self):
        positions = np.zeros((self.num_joints, 3))
        for idx in range(self.num_joints):
            parent = parent_joint_idx[idx]
            if parent == -1:
                positions[idx, :] = self.relative_positions[idx]
            else:
                rot = Rotation.from_rotvec(self.orientations[parent]).inv()
                positions[idx, :] = rot.apply(self.relative_positions[idx]) + positions[parent, :]

        return positions

    # calculates center of mass using weights from pigraph
    # // Body part mass ratios from p59 of http://www.dtic.mil/dtic/tr/fulltext/u2/710622.pdf
    # joint layout pigraph used: https://lisajamhoury.medium.com/understanding-kinect-v2-joints-and-coordinate-system-4f4b90b9df16
    def get_com(self):
        mHead = 0.073
        mTrunk = 0.507
        mUpperArm = 0.026
        mForearm = 0.016
        mHand = 0.007
        mThigh = 0.103
        mCalf = 0.043
        mFoot = 0.015

        p = self.positions
        pHead = p[joint_name_to_idx['head']]
        pTrunk = (p[joint_name_to_idx['neck']] + p[joint_name_to_idx['pelvis']]) * 0.5
        pUpperArmL = (p[joint_name_to_idx['left_shoulder']] + p[joint_name_to_idx['left_elbow']]) * 0.5
        pUpperArmR = (p[joint_name_to_idx['right_shoulder']] + p[joint_name_to_idx['right_elbow']]) * 0.5
        pForearmL = (p[joint_name_to_idx['left_elbow']] + p[joint_name_to_idx['left_wrist']]) * 0.5
        pForearmR = (p[joint_name_to_idx['right_elbow']] + p[joint_name_to_idx['right_wrist']]) * 0.5
        pHandL = p[joint_name_to_idx['left_wrist']]
        pHandR = p[joint_name_to_idx['right_wrist']]
        if self.num_joints >= 55:
            pHandL = (p[joint_name_to_idx['left_wrist']] + p[joint_name_to_idx['left_middle3']]) * 0.5
            pHandR = (p[joint_name_to_idx['right_wrist']] + p[joint_name_to_idx['right_middle3']]) * 0.5
        pThighL = (p[joint_name_to_idx['left_hip']] + p[joint_name_to_idx['left_knee']]) * 0.5
        pThighR = (p[joint_name_to_idx['right_hip']] + p[joint_name_to_idx['right_knee']]) * 0.5
        pCalfL = (p[joint_name_to_idx['left_ankle']] + p[joint_name_to_idx['left_knee']]) * 0.5
        pCalfR = (p[joint_name_to_idx['right_ankle']] + p[joint_name_to_idx['right_knee']]) * 0.5
        pFootL = (p[joint_name_to_idx['left_ankle']] + p[joint_name_to_idx['left_foot']]) * 0.5
        pFootR = (p[joint_name_to_idx['right_ankle']] + p[joint_name_to_idx['right_foot']]) * 0.5

        pCOM = mHead * pHead + mTrunk * pTrunk + \
               mUpperArm * (pUpperArmL + pUpperArmR) + \
               mForearm * (pForearmL + pForearmR) + \
               mHand * (pHandL + pHandR) + \
               mThigh * (pThighL + pThighR) + \
               mCalf * (pCalfL + pCalfR) + \
               mFoot * (pFootL + pFootR)

        # return [pHead, pTrunk, pUpperArmL, pUpperArmR, pForearmL, pForearmR, pHandL, pHandR, pThighL, pThighR, pCalfL, pCalfR, pFootL, pFootR, pCOM]
        return pCOM

    # tranform skeleton using given 4*4 transform matrix
    def transform(self, trans):
        points_homogeneous = np.concatenate((self.positions, np.ones((self.positions.shape[0], 1))), axis=1)  # point positions in homogeneous coordinates
        points_tranformed = np.dot(trans, points_homogeneous.T).T
        self.positions = points_tranformed[:, :-1] / points_tranformed[:, -1:]  # convert back to 3d positions
        self.center_of_mass = self.get_com()
        # since we use relative orientations, only need to transform root orientation
        rot1 = Rotation.from_rotvec(self.relative_orientations[0])
        rot2 = Rotation.from_matrix(trans[:3, :3])
        rot = rot2 * rot1
        if DEBUG:
            print(self.relative_orientations[0], rot.as_rotvec())
        self.relative_orientations[0] = rot.as_rotvec()

        # self.orientations = self.calc_orientations()
        # self.relative_positions = self.calc_relative_positions()


    def get_visualize_geometries(self, use_smplx=False):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[-2, -2, -2])
        geometries = [mesh_frame]
        if use_smplx:
            geometries.append(self.to_smplx_mesh())
            # return geometries

        # add joints
        joints_pcl = o3d.geometry.PointCloud()
        joints_pcl.points = o3d.utility.Vector3dVector(
            np.concatenate((self.positions[:NUM_JOINTS], [self.center_of_mass]), axis=0))
        colors = [[0.8, 0.3, 0.3]] * NUM_JOINTS
        colors.append([0.3, 0.8, 0.3])
        joints_pcl.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(joints_pcl)

        # add bones
        lines = []
        for bone in bones:
            joint1, joint2 = bone
            lines.append([joint_name_to_idx[joint1], joint_name_to_idx[joint2]])
        colors = [[1, 0, 0]] * len(bones)
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(self.positions[:NUM_JOINTS]),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

        return geometries


    def visualize(self, trans=None, use_smplx=False):
        # render
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().line_width = 50

        geometries = self.get_visualize_geometries(use_smplx=use_smplx)
        for geometry in geometries:
            vis.add_geometry(geometry)

        if trans is not None:
            ctr = vis.get_view_control()
            cam_param = ctr.convert_to_pinhole_camera_parameters()
            cam_param.extrinsic = np.linalg.inv(trans)
            ctr.convert_from_pinhole_camera_parameters(cam_param)

        vis.poll_events()
        vis.update_renderer()
        vis.run()

    def save(self, filename, use_smplx=False):
        # render
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().line_width = 50

        geometries = self.get_visualize_geometries(use_smplx=use_smplx)
        for geometry in geometries:
            vis.add_geometry(geometry)

        target = self.center_of_mass
        cam = target + np.array([0.5, 0.5, 1.5])
        # calculate lookat matrix
        transform = makeLookAt(position=cam, target=target)
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param.extrinsic = np.linalg.inv(transform)
        ctr.convert_from_pinhole_camera_parameters(cam_param)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(filename, True)
        vis.destroy_window()

    def to_smplx_mesh(self):
        body_model = smplx.create(smplx_model_folder, model_type='smplx',
                                  gender=self.gender, ext='npz', use_pca=False,
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
                                  batch_size=1)
        torch_param = {}
        torch_param['global_orient'] = torch.tensor(self.relative_orientations[0, :].reshape((1, 3)),
                                                    dtype=torch.float32)
        # torch_param['transl'] = torch.tensor(self.positions[0, :].reshape((1, 3)), dtype=torch.float32)
        # print(torch_param['transl'])
        torch_param['body_pose'] = torch.tensor(self.relative_orientations[1:22, :].reshape((1, 63)),
                                                dtype=torch.float32)
        if NUM_JOINTS == 55:
            torch_param['jaw_pose'] = torch.tensor(self.relative_orientations[22:23, :].reshape((1, 3)),
                                                   dtype=torch.float32)
            torch_param['leye_pose'] = torch.tensor(self.relative_orientations[23:24, :].reshape((1, 3)),
                                                    dtype=torch.float32)
            torch_param['reye_pose'] = torch.tensor(self.relative_orientations[24:25, :].reshape((1, 3)),
                                                    dtype=torch.float32)
            torch_param['left_hand_pose'] = torch.tensor(self.relative_orientations[25:40, :].reshape((1, 45)),
                                                         dtype=torch.float32)
            torch_param['right_hand_pose'] = torch.tensor(self.relative_orientations[40:55, :].reshape((1, 45)),
                                                          dtype=torch.float32)

        torch_param['betas'] = torch.tensor(self.shape, dtype=torch.float32).reshape((1, 10))
        torch_param['expression'] = torch.zeros([1, body_model.num_expression_coeffs], dtype=torch.float32)

        smplx_output = body_model(return_verts=True, return_fullpose=True, **torch_param)
        # print(smplx_output.joints.detach().cpu().numpy()[0][0])

        transl = self.positions[0] - smplx_output.joints.detach().cpu().numpy()[0][0]
        body = o3d.geometry.TriangleMesh()
        body.vertices = o3d.utility.Vector3dVector(smplx_output.vertices.detach().cpu().numpy()[0] + transl)
        body.paint_uniform_color([0.8, 0.8, 0.8])
        body.triangles = o3d.utility.Vector3iVector(body_model.faces)
        body.compute_vertex_normals()
        return body


if __name__ == '__main__':
    DEBUG = True

    human_path = "/local/home/proxe/PROX_temporal/PROXD_temp_v2/MPH16_00157_01/results.pkl"
    smplx_output, body_model = load_human(human_path)
    joints = smplx_output.joints.detach().cpu().numpy()
    full_poses = smplx_output.full_pose.detach().cpu().numpy()
    print(joints[0].shape, full_poses[0].shape)
    # print(full_poses[0])
    skeleton = Skeleton(positions=np.asarray(joints[0][:NUM_JOINTS]),
                        relative_orientations=np.asarray(full_poses[0][:NUM_JOINTS * 3]).reshape((-1, 3)))

    # cam2world tranform
    cam2world_path = r"/local/home/proxe/cam2world/MPH16.json"
    with open(cam2world_path, 'r') as f:
        trans = np.array(json.load(f))

    skeleton.transform(trans)
    # skeleton.visualize(trans)

    # skeleton2 = Skeleton(relative_orientations=skeleton.relative_orientations, relative_positions=skeleton.relative_positions)
    skeleton.visualize(trans, use_smplx=True)
    skeleton.save(filename=os.path.join(results_folder, 'skeleton.png'), use_smplx=True)
    # import open3d as o3d
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
    # vis.get_render_option().mesh_show_back_face = True
    #
    # joints_pcl = o3d.geometry.PointCloud()
    # joints_pcl.points = o3d.utility.Vector3dVector(np.concatenate((joints[0][:NUM_JOINTS], [skeleton.center_of_mass]), axis=0))
    # colors = []
    # for idx in range(NUM_JOINTS):
    #     colors.append([idx / NUM_JOINTS, 0, 0])
    # colors.append([0.3, 0.8, 0.3])
    # joints_pcl.colors = o3d.utility.Vector3dVector(colors)
    # joints_pcl.transform(trans)
    #
    # vis.add_geometry(joints_pcl)
    # ctr = vis.get_view_control()
    # cam_param = ctr.convert_to_pinhole_camera_parameters()
    # cam_param.extrinsic = np.linalg.inv(trans)
    # ctr.convert_from_pinhole_camera_parameters(cam_param)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.run()