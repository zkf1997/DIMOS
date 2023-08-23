import json
import time

import numpy as np
import open3d as o3d
import smplx
import torch
from scipy.stats import norm, vonmises
from tqdm import tqdm

from pigraph_config import *
from load_human import load_human
from data.scene import Scene
from skeleton import NUM_JOINTS, Skeleton
from data.shape_distribution import shape_distribution

DEBUG = False
trans = None

# fit joint positions given relative orientations and global tranlation using smpl
def fit_positions_with_smpl(orientations, translations, shape_params, gender):
    num_samples, num_joints = orientations.shape[:2]
    T = num_samples  # number of frames

    body_model = smplx.create(smplx_model_folder, model_type='smplx',
                              gender=gender, ext='npz', use_pca=False,
                              batch_size=T)
    torch_param = {}
    torch_param['global_orient'] = torch.tensor(orientations[:, 0, :],
                                                dtype=torch.float32)
    # torch_param['transl'] = torch.tensor(translations, dtype=torch.float32)
    torch_param['body_pose'] = torch.tensor(orientations[:, 1:22, :].reshape((T, 63)),
                                            dtype=torch.float32)
    if NUM_JOINTS == 55:
        torch_param['jaw_pose'] = torch.tensor(orientations[:, 22:23, :].reshape((T, 3)),
                                               dtype=torch.float32)
        torch_param['leye_pose'] = torch.tensor(orientations[:, 23:24, :].reshape((T, 3)),
                                                dtype=torch.float32)
        torch_param['reye_pose'] = torch.tensor(orientations[:, 24:25, :].reshape((T, 3)),
                                                dtype=torch.float32)
        torch_param['left_hand_pose'] = torch.tensor(orientations[:, 25:40, :].reshape((T, 45)),
                                                     dtype=torch.float32)
        torch_param['right_hand_pose'] = torch.tensor(orientations[:, 40:55, :].reshape((T, 45)),
                                                      dtype=torch.float32)

    torch_param['betas'] = torch.tensor(shape_params, dtype=torch.float32).reshape((T, 10))
    torch_param['expression'] = torch.zeros([T, body_model.num_expression_coeffs], dtype=torch.float32)

    smplx_output = body_model(return_verts=True, return_full_pose=True, **torch_param)
    # vertices = smplx_output.vertices.detach().cpu().numpy()  # [n_frames, 10475, 3]
    joints = smplx_output.joints.detach().cpu().numpy()  # [n_frames, 127, 3]
    translations = translations - joints[:, 0, :]
    return joints[:, :num_joints, :] + translations.reshape((T, 1, 3))

# per joint distribution, input: relative position and relative orientation as (latitude, longitude, rotation angle)
class JointDistribution:
    def __init__(self, positions, orientations):
        self.positions = positions  # S * 3
        self.orientations = orientations

        #  fit positions with 3 gaussian distribution and orientations with 3 von mises distributions
        px = positions[:, 0]
        py = positions[:, 1]
        pz = positions[:, 2]
        latitude = orientations[:, 0]
        longitude = orientations[:, 1]
        angle = orientations[:, 2]
        if DEBUG:
            print(px.shape)

        t1 = time.time()
        self.x_distribution = norm(*norm.fit(px))  # (mean, std)
        self.y_distribution = norm(*norm.fit(py))
        self.z_distribution = norm(*norm.fit(pz))
        # fix scale for vonmises, https://stackoverflow.com/a/39020834/14532053
        self.latitude_distribution = vonmises(*vonmises.fit(latitude, fscale=1) ) # kappa, loc and scale
        self.longitude_distribution = vonmises(*vonmises.fit(longitude, fscale=1))
        self.angle_distribution = vonmises(*vonmises.fit(angle, fscale=1))
        t2 = time.time()
        if DEBUG:
            print((t2 - t1)*1000, 'ms')

    # yield num_sampls position and orientation samples
    def sample(self, num_samples=1):
        positions = np.asarray([self.x_distribution.mean(), self.y_distribution.mean(),
                               self.z_distribution.mean()])[np.newaxis].repeat(num_samples, axis=0)
        latitudes = self.latitude_distribution.mean() * np.ones((num_samples, 1))
        longitudes = self.longitude_distribution.mean() * np.ones((num_samples, 1))
        angles = self.angle_distribution.rvs(size=num_samples)[..., np.newaxis]
        orientations = np.concatenate((latitudes, longitudes, angles), axis=1)
        return positions, orientations, self.log_probability(positions, orientations)  # (n*3), (n*3), (n,)
        # return position, orientation, 0.0

    # https://github.com/msavva/pigraphs/blob/ee794f3acef4eac418ca0f69bb410fef34b99246/libsg/core/SkeletonDistribution.cpp#L60
    def log_probability(self, positions, orientations):
        position = positions[0]
        orientation = orientations[0]
        # mean_log_probability = self.x_distribution.logpdf(position[0]) + self.y_distribution.logpdf(position[1]) + \
        #                   self.z_distribution.logpdf(position[2]) + self.latitude_distribution.logpdf(orientation[0]) + \
        #                   self.longitude_distribution.logpdf(orientation[1])
        angle_log_probability = self.angle_distribution.logpdf(orientations[:, 2])
        # return mean_log_probability + angle_log_probability
        return angle_log_probability

class ComposedJointDistribution:
    def __init__(self, joint_dist1, joint_dist2):
        self.x = (joint_dist1.x_distribution.mean() + joint_dist2.x_distribution.mean()) / 2.0
        self.y = (joint_dist1.y_distribution.mean() + joint_dist2.y_distribution.mean()) / 2.0
        self.z = (joint_dist1.z_distribution.mean() + joint_dist2.z_distribution.mean()) / 2.0

        self.latitudes = [joint_dist1.latitude_distribution.mean(), joint_dist2.latitude_distribution.mean()]
        self.longitudes = [joint_dist1.latitude_distribution.mean(), joint_dist2.latitude_distribution.mean()]
        self.angle_distributions = [joint_dist1.angle_distribution, joint_dist2.angle_distribution]

    # sample from mixture of distributions: https://stackoverflow.com/a/47763145
    def sample(self, num_samples=1):
        positions = np.asarray([self.x, self.y,
                                self.z])[np.newaxis].repeat(num_samples, axis=0)
        submodel_latitudes = [latitude * np.ones(num_samples) for latitude in self.latitudes]
        submodel_longitudes = [longitude * np.ones(num_samples) for longitude in self.longitudes]
        submodel_angles = [submodel.rvs(size=num_samples) for submodel in self.angle_distributions]
        submodel_choices = np.random.randint(len(self.angle_distributions), size=num_samples)
        latitudes = np.choose(submodel_choices, submodel_latitudes)[:, None]
        longitudes = np.choose(submodel_choices, submodel_longitudes)[:, None]
        angles = np.choose(submodel_choices, submodel_angles)[:, None]

        orientations = np.concatenate((latitudes, longitudes, angles), axis=1)
        return positions, orientations, self.log_probability(positions, orientations)  # (n*3), (n*3), (n,)
        # return position, orientation, 0.0

    def log_probability(self, positions, orientations):
        submodel_logpdf = [submodel.logpdf(orientations[:, 2]) for submodel in self.angle_distributions]
        angle_log_probability = np.asarray(submodel_logpdf).mean(axis=0)
        return angle_log_probability

class SkeletonDistribution:
    def __init__(self, skeletons=None, num_joints=NUM_JOINTS, composition=False, joint_distributions=None):
        self.skeletons = skeletons  #  list of skeletons
        self.num_skeletons = len(skeletons)  # S, number of skeletons
        self.num_joints = num_joints  # J, number of joints of each skeleton
        self.joint_distributions = joint_distributions if composition else self.calc_joint_distributions()  # list of NUM_JOINTS J joint distribution

    @classmethod
    def compose(self, skeleton_dist1, skeleton_dist2):
        skeletons = skeleton_dist1.skeletons + skeleton_dist2.skeletons
        num_joints = skeleton_dist1.num_joints
        composed_joint_distributions = []
        for joint_idx in range(num_joints):
            joint_dist1 = skeleton_dist1.joint_distributions[joint_idx]
            joint_dist2 = skeleton_dist2.joint_distributions[joint_idx]
            composed_joint_distributions.append(ComposedJointDistribution(joint_dist1, joint_dist2))
        return SkeletonDistribution(skeletons, num_joints=num_joints,
                                    joint_distributions=composed_joint_distributions,
                                    composition=True)

    def calc_joint_distributions(self):
        # represent each joint as relative position + relative orientation(latitude, longitude, rotation angle)
        positions = np.asarray([skeleton.relative_positions for skeleton in self.skeletons])  # S*J*3
        orientations = np.asarray([skeleton.relative_orientations for skeleton in self.skeletons]).astype(np.float32)  # S*J*3
        print(orientations.shape)
        # tranform axis angle to latitude longitude rotation angle https://en.wikipedia.org/wiki/N-vector
        angles = np.maximum(np.sqrt((orientations**2).sum(axis=2)).astype(np.float32), np.finfo(np.float32).eps)  # S*J
        latitude = np.arcsin(orientations[:, :, 2] / angles)  # S * J
        longitude = np.arctan2(orientations[:, :, 1], orientations[:, :, 0])  # S * J
        orientations = np.asarray([latitude, longitude, angles]).transpose((1, 2, 0))  # S * J * 3
        if DEBUG:
            print(orientations.shape)

        # calculate per joint distributions
        joint_distributions = []
        if DEBUG:
            print("start fit per joint distributions")
        for idx in tqdm(range(self.num_joints)):
            # if DEBUG:
            #     print(idx)
            joint_distributions.append(JointDistribution(positions[:, idx, :], orientations[:, idx, :]))
        return joint_distributions

    # sample num_samples skeletons, return topk skeletons with best log probability
    def sample(self, num_samples=1, topk=1, gender='male'):
        max_log_probability = 0
        skeleton = None
        if DEBUG:
            print("sampling skeletons")

        positions = np.zeros((num_samples, self.num_joints, 3))
        orientations = np.zeros((num_samples, self.num_joints, 3))
        scores = np.zeros((num_samples, self.num_joints))
        sum_log_probability = 0.0
        for idx, joint_distribution in enumerate(self.joint_distributions):
            position, orientation, score = joint_distribution.sample(num_samples)
            positions[:, idx, :] = position
            orientations[:, idx, :] = orientation
            scores[:, idx] = score

        score_sum = scores.sum(axis=1)
        # max_score_idx = score_sum.argmax()
        topk_idx = np.argpartition(score_sum, -topk)[-topk:]  # indices of topk skeletons, the order within the k is undefined

        orientations = orientations[topk_idx, :, :]
        latitudes = np.asarray(orientations[:, :, 0])
        longitudes = np.asarray(orientations[:, :, 1])
        angles = np.asarray(orientations[:, :, [2]])
        axes = np.asarray([np.cos(latitudes) * np.cos(longitudes),
                           np.cos(latitudes) * np.sin(longitudes),
                           np.sin(latitudes)]).transpose((1, 2, 0))  # S * J * 3
        orientations = axes * angles
        positions = positions[topk_idx, :, :]
        shape_params = np.random.multivariate_normal(**shape_distribution, size=topk)
        positions = fit_positions_with_smpl(orientations, positions[:, 0, :], shape_params, gender)

        skeletons = [Skeleton(positions=positions[idx], relative_orientations=orientations[idx],
                              gender=gender, shape=shape_params[idx]) for idx in range(topk)]

        return skeletons


    def log_probability(self, skeleton):
        pass

if __name__ == "__main__":
    DEBUG = True
    scene = Scene.create("MPH16")

    human_path = "/local/home/proxe/PROX_temporal/PROXD_temp_v2/MPH16_00157_01/results.pkl"
    smplx_output, body_model = load_human(human_path)
    joints = smplx_output.joints.detach().cpu().numpy()
    full_poses = smplx_output.full_pose.detach().cpu().numpy()

    # cam2world tranform
    cam2world_path = r"/local/home/proxe/cam2world/MPH16.json"
    with open(cam2world_path, 'r') as f:
        trans = np.array(json.load(f))

    T = joints.shape[0]
    print("building skeletons")
    skeletons = [Skeleton(positions=np.asarray(joints[idx][:NUM_JOINTS]),
                          relative_orientations=np.asarray(full_poses[idx][:NUM_JOINTS * 3]).reshape((-1, 3)),
                          transform=trans) for idx in range(1400, 1650)]
    print("skeletons built")

    dis = SkeletonDistribution(skeletons)
    print("skeleton distribution fit finish")
    skeletons = dis.sample(num_samples=1000, topk=10)
    # skeleton.visualize()

    # igraph = InteractionGraph(scene, skeletons[0])
    # igraph.visualize(trans)

    # visualize with smplx
    geometries = scene.get_visualize_geometries()
    # render
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().line_width = 50
    for geometry in geometries:
        vis.add_geometry(geometry)
    print("geometries added to vis")
    # if trans is not None:
    #     ctr = vis.get_view_control()
    #     cam_param = ctr.convert_to_pinhole_camera_parameters()
    #     cam_param.extrinsic = np.linalg.inv(trans)
    #     ctr.convert_from_pinhole_camera_parameters(cam_param)

    for idx, skeleton in enumerate(skeletons):
        body = skeleton.get_visualize_geometries(use_smplx=True)
        for geometry in body:
            vis.add_geometry(geometry)
        if trans is not None:
            ctr = vis.get_view_control()
            cam_param = ctr.convert_to_pinhole_camera_parameters()
            cam_param.extrinsic = np.linalg.inv(trans)
            ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()
        # time.sleep(1)
        # filename = os.path.join(rendering_folder, "skeleton{idx}.png".format(idx=idx))
        # img = vis.capture_screen_float_buffer(False)
        # Image.fromarray(np.asarray(img)).save(filename)
        # vis.capture_screen_image(filename, True)
        for geometry in body:
            vis.remove_geometry(geometry)