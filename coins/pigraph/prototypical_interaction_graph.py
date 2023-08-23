import copy
import json

import numpy as np
import smplx
import torch
from tqdm import tqdm

from interaction_graph import InteractionGraph
from load_human import load_sequence, smplx_forward
from pigraph_config import *
from data.scene import Scene, scenes
from skeleton import NUM_JOINTS, Skeleton
from skeleton_distribution import SkeletonDistribution

DEBUG = False
PENALTY = -0.05 # penalty for activated joints never seen in pigraph

def compose_histograms(histogram_list1, histogram_list2):
    composed_histograms = []
    assert  len(histogram_list1) == len(histogram_list2)
    for histogram1, histogram2 in zip(histogram_list1, histogram_list2):
        composed_histograms.append([histogram1, histogram2])
    return composed_histograms

# return the probability one vector of feature is sampled from a vector of histograms
# reference is  pigraph:https://github.com/msavva/pigraphs/blob/ee794f3acef4eac418ca0f69bb410fef34b99246/libsg/interaction/similarity.cpp#L321
def feature_similarity(feature_histograms, features):
    num_features = min(len(feature_histograms), len(features))
    if num_features == 0:
        return 0.0
    num_success = 0 # number of features are successfully sampled from hisotograms
    for idx in range(num_features):
        histogram_or_list = feature_histograms[idx]
        if isinstance(histogram_or_list, list):
            success = 1
            for histogram in histogram_or_list:  # for composed pigraph, can be a list of histograms, need to success in all
                feature = features[idx]  # scalar
                density, bin_edges = histogram
                bin_num = np.digitize(feature, bin_edges) - 1
                pdf = density[bin_num] if (0 <= bin_num < len(density)) else 0.0
                if pdf < 1.0 / (histogram[1][-1] - histogram[1][
                    0]):  # if pdf is greater than average, we consider the sample success
                    success = 0
                    break
            num_success += success
        else:
            histogram = histogram_or_list # numpy histogram:(density, bin_edges)
            feature = features[idx]  # scalar
            # pdf = scipy.stats.rv_histogram(histogram).pdf(feature)
            density, bin_edges = histogram
            bin_num = np.digitize(feature, bin_edges) - 1
            pdf = density[bin_num] if (0 <= bin_num < len(density)) else 0.0
            if pdf > 1.0 / (histogram[1][-1] - histogram[1][0]): # if pdf is greater than average, we consider the sample success
                num_success += 1
    return num_success / num_features  # define the probability of sampling as ratio of successfully sampled feature

# def feature_similarity(feature_histograms, features):
#     num_features = min(len(feature_histograms), len(features))
#     if num_features == 0:
#         print("no features")
#         return 0.0
#     similarity = 0.0
#     for idx in range(num_features):
#         histogram = feature_histograms[idx]  # numpy histogram:(density, bin_edges)
#         feature = features[idx]  # scalar
#         pdf = scipy.stats.rv_histogram(histogram).pdf(feature)
#         if DEBUG:
#             print(feature, pdf)
#         similarity += pdf
#     return similarity / num_features

class PrototypicalInteractionGraph:
    def __init__(self, retarget=False, compose=False, **kwargs):
        if retarget or compose:
            self.__dict__.update(kwargs)
            return
        self.interaction = kwargs.get("interaction")  # interaction type
        self.igraphs = kwargs.get("igraphs")  # interaction graphs
        if self.igraphs == None:
            self.igraphs = self.build_igraphs(self.interaction, kwargs.get("interaction_data"))  # build igraphs according to interaction type
        self.joint_count = [0]*NUM_JOINTS  # the number of igraphs where a joint have contact edges
        self.edge_count = {}  # the number of igraphs where edge (joint, obj_category) exists
        self.skeleton_distribution = self.calc_skeleton_distribution()
        self.feature_histograms = self.calc_histograms()  # dict of histograms for category level object node features and contact edge features

    def build_igraphs(self, interaction, interaction_data=None):
        igraphs = []
        # build from prox_temporal (sequence, start_frame, end_frame) tuples
        if interaction_data == None:
            sequence_list = interaction_sequences[interaction]
            for recording_name, start_frame, end_frame in sequence_list:
                print(recording_name , start_frame, end_frame)
                smplx_output, body_model = load_sequence(recording_name, start_frame, end_frame)
                joints = smplx_output.joints.detach().cpu().numpy()
                full_poses = smplx_output.full_pose.detach().cpu().numpy()
                scene_name = recording_name.split('_')[0]
                scene = Scene.create(scene_name=scene_name)
                for idx in range(end_frame - start_frame + 1):
                    skeleton = Skeleton(positions=np.asarray(joints[idx][:NUM_JOINTS]),
                                        relative_orientations=np.asarray(full_poses[idx][:NUM_JOINTS * 3]).reshape((-1, 3)),
                                        transform=scene.cam2world)
                    igraph = InteractionGraph(scene, skeleton)
                    igraphs.append(igraph)
        # build from interaction data dict
        else:
            body_model_dict = {
                'male': smplx.create(smplx_model_folder, model_type='smplx',
                              gender='male', ext='npz',
                              num_pca_comps=12,),
                'female': smplx.create(smplx_model_folder, model_type='smplx',
                              gender='female', ext='npz',
                              num_pca_comps=12,)
            }
            for record in tqdm(interaction_data):
                # scene_name, sequence, frame_idx, smplx_param, interaction_labels, interaction_obj_idx = record
                scene = scenes[record['scene_name']]
                smplx_param = record['smplx_param']

                body_model = body_model_dict[smplx_param['gender']]
                # print(smplx_param)
                for key in smplx_param:
                    # print(key)
                    if not key in ['gender', 'frame_name']:
                        smplx_param[key] = torch.tensor(smplx_param[key])
                smplx_output = body_model(**smplx_param, return_full_pose=True)
                joints = smplx_output.joints.detach().cpu().numpy()
                full_poses = smplx_output.full_pose.detach().cpu().numpy()
                # print('fullpose', full_poses.shape)
                skeleton = Skeleton(positions=np.asarray(joints[0][:NUM_JOINTS]),
                                    relative_orientations=np.asarray(full_poses[0][:NUM_JOINTS * 3]).reshape((-1, 3)),
                                    gender=smplx_param['gender'], shape=smplx_param['betas']
                                    )
                igraph = InteractionGraph(scene, skeleton)
                # igraph.visualize(use_smplx=True)
                igraphs.append(igraph)

        return igraphs

    def calc_skeleton_distribution(self):
        skeletons = [igraph.skeleton for igraph in self.igraphs]
        if DEBUG:
            print("fit skeleton distribution")
        return SkeletonDistribution(skeletons)

    # calc histogram for object node features and contact edge features
    # here we suppose category level object nodes,i.e. one node for all instances of same category
    # for object node: index with object category id, histogram_dict[object_category]
    # for contact edges: index with tuple of (joint_id, object_category_id), histogram_dict[joint_id, object_category]
    def calc_histograms(self):
        object_node_features = {}  # dict of per category object node features, gathered from all igrpahs
        contact_edge_features = {}  # feautres dict of contact edge between joint and per category object node
        if DEBUG:
            print("gather feature values into matrix")
        for igraph in self.igraphs:
            # # gather object node features
            # for object_node in igraph.scene.object_nodes:
            #     if object_node.category not in object_node_features:
            #         object_node_features[object_node.category] = []
            #     object_node_features[object_node.category].append(object_node.features)

            # gather contact edge features and contact activated object node feature
            for human_object_edge in igraph.human_object_edges:
                joint_id = human_object_edge.human_node_id
                object_node = igraph.scene.object_nodes[human_object_edge.object_node_id]
                object_category = object_node.category
                # contact activated object node features
                if object_category not in object_node_features:
                    object_node_features[object_category] = []
                object_node_features[object_category].append(object_node.features)
                # contact edge features
                if (joint_id, object_category) not in contact_edge_features:
                    contact_edge_features[(joint_id, object_category)] = []
                contact_edge_features[(joint_id, object_category)].append(human_object_edge.features)

        # build histograms
        if DEBUG:
            print("fit histograms")
        feature_histograms = {}
        for object_category in object_node_features:
            features = np.asarray(object_node_features[object_category])  # 2d matrix, each line is feature of one node of this category
            num_features = features.shape[1]
            num_nodes = features.shape[0]
            histograms = []
            for idx in range(num_features):
                histograms.append(np.histogram(features[:, idx], density=True))
            feature_histograms[object_category] = histograms

        for (joint_id, object_category) in contact_edge_features:
            # accumulate joint and edge counts
            self.joint_count[joint_id] += np.asarray(contact_edge_features[(joint_id, object_category)]).shape[0]
            if not (joint_id, object_category) in self.edge_count:
                self.edge_count[(joint_id, object_category)] = 0
            self.edge_count[(joint_id, object_category)] += np.asarray(contact_edge_features[(joint_id, object_category)]).shape[0]

            # calculate edge feature histograms
            features = np.asarray(contact_edge_features[(joint_id, object_category)])  # 2d matrix, each line is feature of one contact edge of this category
            num_features = features.shape[1]
            histograms = []
            for idx in range(num_features):
                histograms.append(np.histogram(features[:, idx], density=True))
            feature_histograms[(joint_id, object_category)] = histograms

        if DEBUG:
            print("fit histograms finish")
        return feature_histograms

    # probability score one igraph is sampled from this pigraph
    # reference pigraph:https://github.com/msavva/pigraphs/blob/ee794f3acef4eac418ca0f69bb410fef34b99246/libsg/interaction/similarity.cpp#L166
    def similarity(self, igraph, log=False):
        sum_similarity = 0
        num_edges = len(igraph.human_object_edges)
        if num_edges == 0:
            return 0.0
        for human_object_edge in igraph.human_object_edges:
            joint_id = human_object_edge.human_node_id
            if self.joint_count[joint_id] == 0: # this joint is never activated in this pigraph
                # sum_similarity += PENALTY
                continue
            object_node = igraph.scene.object_nodes[human_object_edge.object_node_id]
            object_category = object_node.category
            obj_similarity = 0
            edge_similarity = 0
            # joint_weight = self.joint_count[joint_id] / len(self.igraphs) # weight for this joint according to acitvation frequency
            joint_weight = self.joint_count[joint_id] / np.asarray(self.joint_count).sum()
            edge_weight = 0 # weight for this contact edge according the ratio of #(joint_id, object_category) / #joint_id
            # activated object similarity, if not find in pigraph, the similarity is equal to 0 so do not need to add
            if (object_category not in self.feature_histograms) or ((joint_id, object_category) not in self.feature_histograms):
                continue
            obj_similarity = feature_similarity(self.feature_histograms[object_category], object_node.features)
            edge_similarity = feature_similarity(self.feature_histograms[(joint_id, object_category)],
                                                 human_object_edge.features)
            edge_weight = self.edge_count[(joint_id, object_category)] / self.joint_count[joint_id]
            sum_similarity += joint_weight * edge_weight * (obj_similarity + edge_similarity)

            if log:
                if (object_category == 3):
                    print("contact with chair!")
                    print(joint_id, object_category)
                    print(joint_weight, edge_weight, obj_similarity, edge_similarity)

        return sum_similarity

    # retarget a pigraph for verb-noun to a new noun
    # https://github.com/msavva/pigraphs/blob/ee794f3acef4eac418ca0f69bb410fef34b99246/libsg/core/synth/InteractionSynth.cpp#L260
    @classmethod
    def retarget(self, interaction):
        verb, noun = interaction.split("-")
        old_verb, old_noun = self.interaction.split("-")
        assert verb == old_verb
        category = int(category_dict.loc[category_dict["mpcat40"] == noun]["mpcat40index"])
        old_category = int(category_dict.loc[category_dict["mpcat40"] == old_noun]["mpcat40index"])
        joint_count = copy.deepcopy(self.joint_count)
        skeleton_distribution = copy.deepcopy(self.skeleton_distribution)
        edge_count = copy.deepcopy(self.edge_count)
        feature_histograms = copy.deepcopy(self.feature_histograms)
        # TODO: if catogory appears in old pigraph, we need to combine histograms instead of replace
        # print(category, old_category)
        feature_histograms[category] = feature_histograms[old_category]
        del feature_histograms[old_category]
        for (joint_id, object_category) in self.edge_count:
            if (object_category == old_category):
                edge_count[(joint_id, category)] = edge_count[(joint_id, old_category)]
                del edge_count[(joint_id, old_category)]
                # TODO: if catogory appears in old pigraph, we need to combine histograms instead of replace
                feature_histograms[(joint_id, category)] = feature_histograms[(joint_id, old_category)]
                del feature_histograms[(joint_id, old_category)]

        return PrototypicalInteractionGraph(retarget=True, igraphs=self.igraphs, interaction=interaction,
                                            joint_count=joint_count, edge_count=edge_count,
                                            skeleton_distribution=skeleton_distribution,
                                            feature_histograms=feature_histograms)

    @classmethod
    def compose(self, interaction, pigraph1, pigraph2):
        igraphs = pigraph1.igraphs + pigraph2.igraphs
        joint_count = pigraph1.joint_count + pigraph2.joint_count
        edge_count = copy.deepcopy(pigraph1.edge_count)
        for edge in pigraph2.edge_count:
            if not edge in edge_count:
                edge_count[edge] = pigraph2.edge_count[edge]
            else:
                edge_count[edge] += pigraph2.edge_count[edge]
        skeleton_distribution = SkeletonDistribution.compose(pigraph1.skeleton_distribution, pigraph2.skeleton_distribution)

        feature_histograms = copy.deepcopy(pigraph1.feature_histograms)
        for edge in pigraph2.feature_histograms:
            if not edge in feature_histograms:
                feature_histograms[edge] = pigraph2.feature_histograms[edge]
            else:
                feature_histograms[edge] = compose_histograms(feature_histograms[edge], pigraph2.feature_histograms[edge])
        return PrototypicalInteractionGraph(compose=True, igraphs=igraphs, interaction=interaction,
                                            joint_count=joint_count, edge_count=edge_count,
                                            skeleton_distribution=skeleton_distribution,
                                            feature_histograms=feature_histograms)
if __name__ == "__main__":
    DEBUG = True

    print("load scene")
    scene = Scene.create("MPH16")
    print("scene loaded")

    from load_human import load_human

    human_path = "/local/home/proxe/PROX_temporal/PROXD_temp_v2/MPH16_00157_01/results.pkl"
    smplx_output, body_model = load_human(human_path)
    joints = smplx_output.joints.detach().cpu().numpy()
    full_poses = smplx_output.full_pose.detach().cpu().numpy()
    # cam2world tranform
    cam2world_path = r"/local/home/proxe/cam2world/MPH16.json"
    with open(cam2world_path, 'r') as f:
        trans = np.array(json.load(f))

    igraphs = []
    for idx in tqdm(range(1400, 1650, 1)):
        # if DEBUG:
        #     print(idx)
        skeleton = Skeleton(positions=np.asarray(joints[idx][:NUM_JOINTS]),
                            relative_orientations=np.asarray(full_poses[idx][:NUM_JOINTS * 3]).reshape((-1, 3)),
                            transform=trans)
        igraph = InteractionGraph(scene, skeleton)
        igraphs.append(igraph)

    pigraph = PrototypicalInteractionGraph(igraphs=igraphs, interaction="sit_bed")
    igraph = InteractionGraph(scene, pigraph.skeleton_distribution.sample())
    print(pigraph.similarity(igraph))
    igraph.log()
    igraph.visualize()