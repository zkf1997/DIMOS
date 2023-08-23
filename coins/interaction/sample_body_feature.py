import numpy as np
import torch
from tqdm import tqdm

from body_trainer import *

def interaction_name_to_code(interaction):
    verb_code = np.zeros(num_verb, dtype=np.float32)
    noun_code = np.zeros(num_noun, dtype=np.float32)
    interaction_code = np.zeros(num_noun * num_verb, dtype=np.float32)
    for atomic_interaction in interaction.split('+'):
        verb, noun = atomic_interaction.split('-')
        verb_id = action_names.index(verb)
        verb_code += np.eye(num_verb, dtype=np.float32)[verb_id]
        noun_id = category_dict[category_dict['mpcat40'] == noun].index[0]
        noun_code += np.eye(num_noun, dtype=np.float32)[noun_id]
        interaction_code += np.kron(verb_code, noun_code)
    return interaction_code, verb_code, noun_code

device = torch.device('cuda')

parser = ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="/mnt/scratch/scene_graph/results/body_mesh/more_atomic/version_7/checkpoints/last.ckpt")
parser.add_argument("--sample_dir", type=str, default='IPoser')
args = parser.parse_args()

sample_dir = args.sample_dir

model = LitBodyEncoder.load_from_checkpoint(args.checkpoint).to(device)

batch_size = model.hparams.batch_size
print(batch_size)
print(model.args.use_regressor)
sample_size = 32
test_interactions = interaction_names
for interaction in tqdm(test_interactions):
    interaction_code, verb_code, noun_code = interaction_name_to_code(interaction)
    atomics = interaction.split('+')
    verbs = [atomic.split('-')[0] for atomic in atomics]
    nouns = [atomic.split('-')[1] for atomic in atomics]
    # print(set(noun_ids))
    interaction_code = torch.tensor(interaction_code, dtype=torch.float32, device=device).repeat(batch_size, 1)
    # print(interaction_code.shape)
    num_tries = np.ones(sample_size) * batch_size
    for idx in tqdm(range(sample_size)):
        vertices_batch, features_batch, smplx_params_batch = model.generate(interaction_code)
        for batch_idx in range(batch_size):
            vertices, features = vertices_batch[batch_idx].detach().cpu(), features_batch[batch_idx].detach().cpu()
            smplx_params = {}
            for key in smplx_params_batch:
                smplx_params[key] = smplx_params_batch[key][[batch_idx]]
            _, obj_id = torch.max(features[:, 1:], dim=1)
            # print(obj_id.shape)
            features[obj_id == 41, 0] = 0
            contact = features[:, 0] > 0.5
            features[~contact, 1:] = torch.eye(42)[-1]
            break

        seq_data = smplx_params
        # seq_data['feature'] = {'fc': (features[:, 0] > 0.5).type(torch.int32).detach().cpu().numpy(),
        #                        'fs': obj_id.detach().cpu().numpy()}
        seq_data['feature'] = features.detach().cpu().numpy()
        # seq_data['gender'] = 'male' if (idx < (sample_size // 2)) else 'female'
        seq_data['gender'] = 'neutral'
        # transform according POSA canonical coords frame
        from scipy.spatial.transform import Rotation as R
        orient = R.from_rotvec(seq_data['global_orient'].reshape(3).detach().cpu().numpy())
        orient = R.from_euler('y', -180, degrees=True) * R.from_euler('x', -90, degrees=True) * orient
        seq_data['global_orient'] = np.asarray([orient.as_rotvec()], dtype=np.float32)
        pkl_fname = results_folder / sample_dir / interaction / '{}.pkl'.format(interaction + '_' + str(idx))
        pkl_fname.parent.mkdir(exist_ok=True, parents=True)
        with open(pkl_fname, 'wb') as result_file:
            pickle.dump(seq_data, result_file, protocol=2)

        body_sample_smplx = visualize_vertex_obj_dists(vertices.detach().cpu().numpy(),
                                                       features.detach().cpu().numpy(),
                                                       model.mesh.meshes[2].faces,
                                                       (0.8, 0.0, 0.0))
        snapshot_fname = results_folder / sample_dir / interaction / '{}.png'.format(interaction + '_' + str(idx))
        img_grid = render_body_multview(body_sample_smplx)
        img_grid.save(str(snapshot_fname))
    print(interaction, 'need ', num_tries.mean(), ' sample tries')
