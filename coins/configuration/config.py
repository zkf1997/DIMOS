from pathlib import Path
import json
import pandas as pd
import numpy as np
from PIL import ImageColor
import platform

local_machine = (platform.node() == 'dalcowks')

proxe_base_folder = Path("data/proxs")
scene_folder = Path.joinpath(proxe_base_folder, "scenes_semantics")
sdf_folder = Path.joinpath(proxe_base_folder, "sdf")
cam2world_folder = Path.joinpath(proxe_base_folder, "cam2world")
# human_folder = Path.joinpath(proxe_base_folder, "PROX_temporal/PROXD_temp_v2")
# graph_folder = Path.joinpath(proxe_base_folder, "scene_graph")
scene_cache_folder = Path.joinpath(proxe_base_folder, 'scene_segmentation')
# downloaded files from POSA: https://posa.is.tue.mpg.de/index.html
posa_folder = Path.joinpath(proxe_base_folder, 'POSA_dir')
mesh_ds_folder = Path.joinpath(proxe_base_folder, 'POSA_dir', 'mesh_ds')
# smplx models
smplx_model_folder = Path.joinpath(proxe_base_folder, "models_smplx_v1_1/models")
# project directory
project_folder = Path(__file__).resolve().parents[1]
# mesh upsample and downsample weights
mesh_operation_file = Path.joinpath(project_folder, "data", 'mesh_operation.npz')
# tranformation matrix between PROX and POSA scenes
scene_registration_file = Path.joinpath(project_folder, "data", 'scene_registration.pkl')
# checkpoints
checkpoint_folder = Path.joinpath(project_folder, 'checkpoints')
checkpoint_folder.mkdir(parents=True, exist_ok=True)
# rendering and results
results_folder = Path.joinpath(Path(__file__).resolve().parents[1], "results")
results_folder.mkdir(parents=True, exist_ok=True)
render_folder = Path.joinpath(Path(__file__).resolve().parents[1], "render")
render_folder.mkdir(parents=True, exist_ok=True)

# scene names
scene_names = ["BasementSittingBooth", "MPH11", "MPH112", "MPH16", "MPH1Library", "MPH8",
               "N0SittingBooth", "N0Sofa", "N3Library", "N3Office", "N3OpenArea", "Werkraum"]
test_scenes = ['MPH1Library', 'MPH16', 'N0SittingBooth', 'N3OpenArea']
train_scenes = sorted(list(set(scene_names) - set(test_scenes)))

# manually selected object instances for interactions. For each scene and each objects category combination, we list the selected instance combinations.
candidate_combination_dict = {
    'MPH1Library':{
        'wall': [[0]],
        'floor': [[1]],
        'chair': [[2], [3], [4], [7], [9]],
        'table': [[11]],
        'shelving': [[12]],
        'floor+wall':[[1, 0]],
        'floor+shelving':[[1, 12]],
        'floor+table':[[1, 11]],
        'chair+table':[[2, 11], [3, 11], [4, 11], [7, 11], [9, 11]]
    },
    'MPH16':{
        'wall': [[0]],
        'floor':[[2]],
        'chair':[[3]],
        'cabinet':[[5], [6]],
        'table':[[4]],
        'bed':[[9]],
        'tv_monitor':[[10]],
        'shelving':[[11], [12]],
        'floor+wall':[[2, 0]],
        'floor+table':[[2, 4]],
        'floor+tv_monitor':[[2, 10]],
        'floor+shelving':[[2, 11], [2, 12]],
        'chair+table':[[3, 4]]
    },
    'N0SittingBooth':{
        'wall':[],
        'floor':[[1]],
        'table':[[2], [3]],
        'floor+table':[[1, 2], [1,3]],
        'floor+wall':[],
    },
    'N3OpenArea':{
        'wall':[[0]],
        'floor':[[1]],
        'chair':[[2], [3], [4], [5]],
        'table':[[6]],
        'sofa':[[11]],
        'floor+wall':[[1, 0]],
        'floor+table':[[1, 6]],
        'chair+table':[[3, 6], [4, 6], [5, 6], [2, 6]],
        # 'chair+table':[[2, 6]],
        'sofa+table':[[11, 6]]
    }
}

# sequence names
recordings_temporal = Path.joinpath(Path(__file__).resolve().parent, "recordings_temporal.txt")
sequence_names = [sequence.split('\n')[0] for sequence in recordings_temporal.open().readlines()]

# interaction names
atomic_interaction_names = ['sit on-chair', 'sit on-sofa', 'sit on-bed', 'sit on-cabinet', 'sit on-table',
                            # 'sit on-stool', 'stand on-furniture',
                            'stand on-floor', 'stand on-table', 'stand on-bed',
                            'stand on-chest_of_drawers',
                            'lie on-sofa', 'lie on-bed',
                            'touch-table', 'touch-board_panel', 'touch-tv_monitor', 'touch-shelving', 'touch-wall', 'touch-shelving'
                            # 'touch-lighting', 'touch-objects',
                     ]
atomic_interaction_names_include_motion = ['jump on-sofa', 'step down-table', 'touch-shelving', 'sit down-sofa', 'step up-table', 'side walk-floor', 'turn-floor', 'sit down-chair', 'stand up-bed', 'step up-sofa', 'step down-sofa', 'step down-chair', 'touch-board_panel', 'sit on-seating', 'sit on-chair', 'walk on-floor', 'sit on-bed', 'stand on-table', 'stand up-sofa', 'turnover-floor', 'lie on-sofa', 'lie down-sofa', 'a pose-floor', 'touch-tv_monitor', 'stand up-chair', 'sit up-sofa', 'restfoot-chair', 'stand on-bed', 'step back-floor', 'touch-chair', 'step up-chair', 'move leg-sofa', 'move on-sofa', 'touch-chest_of_drawers', 'touch-sofa', 'stand up-cabinet', 'sit on-stool',
'lie on-bed', 'touch-table', 'lie on-seating', 'touch-wall', 'stand on-floor', 'sit on-sofa', 'move leg-bed', 'sit on-table', 'sit on-cabinet', 'restfoot-stool', 'sit down-cabinet', 'stand on-chest_of_drawers', 'sit down-bed']
atomic_interaction_names_include_motion_train = ['sit on-sofa', 'touch-shelving', 'touch-tv_monitor', 'sit down-sofa', 'jump on-sofa', 'touch-chair', 'step down-chair', 'walk on-floor', 'touch-chest_of_drawers', 'sit down-bed', 'sit on-table', 'move on-sofa', 'stand on-chest_of_drawers', 'turn-floor', 'lie on-sofa', 'stand up-bed', 'lie on-bed', 'step up-sofa', 'side walk-floor', 'sit down-cabinet', 'stand up-chair', 'stand up-cabinet', 'touch-sofa', 'sit on-cabinet', 'a pose-floor', 'move leg-sofa', 'sit on-bed', 'touch-wall', 'sit on-chair', 'step down-table', 'stand up-sofa', 'sit up-sofa', 'touch-table', 'step up-chair', 'stand on-table', 'step down-sofa', 'sit down-chair', 'stand on-floor', 'stand on-bed', 'touch-board_panel', 'lie down-sofa', 'step up-table']
composed_interaction_names = ['sit on-chair+touch-table', 'sit on-sofa+touch-table',
                      # 'stand on-floor+touch-lighting', 'stand on-floor+touch-objects',
                              'stand on-floor+touch-board_panel', 'stand on-floor+touch-table',
                      'stand on-floor+touch-tv_monitor', 'stand on-floor+touch-shelving', 'stand on-floor+touch-wall',
                      ]
test_composed_interaction_names = [
    'sit on-chair+touch-table',
    'stand on-floor+touch-board_panel', 'stand on-floor+touch-table',
]
interaction_names = atomic_interaction_names_include_motion_train + composed_interaction_names

# load category name and visualization color
#mpcat40index	mpcat40	hex	wnsynsetkey	nyu40	skip	labels
mptsv_path = Path.joinpath(Path(__file__).resolve().parent, "mpcat40.tsv")
category_dict = pd.read_csv(mptsv_path, sep='\t')
category_dict['color'] = category_dict.apply(lambda row: np.array(ImageColor.getrgb(row['hex'])), axis=1)
obj_category_num = 42

# human body param
num_pca_comps = 6
smplx_param_names = ['betas', 'global_orient', 'transl', 'body_pose', 'left_hand_pose', 'right_hand_pose']
smplx_param_names += ['jaw_pose', 'leye_pose', 'reye_pose', 'expression']
used_smplx_param_names = ['transl', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'betas']  # these are used in diversity evaluation

# body part segmentation
body_parts = ['back', 'gluteus', 'L_Hand', 'R_Hand', 'L_Leg', 'R_Leg', 'thighs']
body_part_vertices = {}
for body_part in body_parts:
    with open(Path.joinpath(proxe_base_folder, 'body_segments', body_part + '.json'), 'r') as file:
        body_part_vertices[body_part] = json.load(file)['verts_ind']
#https://github.com/Meshcapade/wiki/blob/main/assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json
with open((project_folder / 'configuration' / 'smplx_vert_segmentation.json'), 'r') as file:
    body_part_vertices_full = json.load(file)
upper_body_parts = ["rightHand", "leftArm",
                    "rightArm", "leftHandIndex1", "rightHandIndex1", "leftForeArm",
                    "rightForeArm", "leftHand",
                    ]
lower_body_parts = ["rightUpLeg", "leftLeg", "leftToeBase", "leftFoot", "rightFoot",
                    "rightLeg", "rightToeBase", "leftUpLeg",
                    ]

# map action to corresponding body parts
action_names = ['sit on', 'lie on', 'stand on', 'touch', 'step back', 'restfoot', 'step down', 'turn', 'jump on', 'sit up', 'stand up', 'turnover', 'sit down', 'move on', 'lie down', 'move leg', 'walk on', 'a pose', 'step up', 'side walk']
action_names_train = ['sit on', 'lie on', 'stand on', 'touch', 'jump on', 'turn', 'move leg', 'stand up', 'sit down', 'sit up', 'side walk', 'step down', 'walk on', 'a pose', 'lie down', 'step up', 'move on']
num_verb = len(action_names)
num_noun = 42
maximum_atomics = 2
action_body_part_mapping = {
    'sit on': ['gluteus', 'thighs'],
    'lie on': ['back', 'gluteus', 'thighs'],
    'stand on': ['L_Leg', 'R_Leg'],
    'touch': ['L_Hand', 'R_Hand'],
}
