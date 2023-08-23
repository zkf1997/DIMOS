import sys
import os
sys.path.append('..')
from configuration.config import *
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser

from data.scene import scenes

parser = ArgumentParser()
parser.add_argument("--scene_name", type=str, default='test')
parser.add_argument("--interaction", type=str, default='all')
parser.add_argument("--num_try", type=int, default=1)
args = parser.parse_args()

os.environ["POSA_dir"] = posa_folder.__str__()
# if not local_machine:
#     os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

def eval_command(command):
    os.system(command)

def synthesize():
    used_scene_names = test_scenes if args.scene_name == 'test' else [args.scene_name]
    for scene_name in used_scene_names:
        scene = scenes[scene_name]
        used_interactions = interaction_names if args.interaction == 'all' else [args.interaction]
        for interaction in tqdm(used_interactions):
            print(interaction)
            if scene.support_interaction(interaction):
                atomic_interactions = interaction.split('+')
                verbs, nouns, candidate_combinations = scene.get_interaction_candidate_objects(interaction)
                for object_combination in tqdm(candidate_combinations):
                    pose_source = 'IPoser'
                    input_path = Path.joinpath(results_folder, pose_source, interaction).__str__()

                    command = 'python src/affordance.py --config cfg_files/contact_semantics.yaml --checkpoint_path $POSA_dir/trained_models/contact_semantics.pt ' \
                              '--pkl_file_path \'{input_path}\'  --scene_name {scene_name} --interaction \'{interaction}\' --object_combination {object_combination} --export_dir POSA_{pose_source}_best{num_try} ' \
                              '--save_mesh 0 --render 0 --show_gen_sample 0 --use_semantics 1  --opt_pose 1 --load_feature 1 --semantics_w 0.5 --pen_w 10 --pose_w 100 --max_init_points {num_try}' \
                        .format(scene_name=scene_name, object_combination='+'.join([str(instance.id) for instance in object_combination]), input_path=input_path, interaction=interaction,
                                pose_source=pose_source, num_try=args.num_try)
                    print(command)
                    eval_command(command)

synthesize()
