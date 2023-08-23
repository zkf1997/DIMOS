from pigraph_config import *
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser

def eval_command(command):
    os.system(command)

parser = ArgumentParser()
parser.add_argument("--num_process", type=int, default=4)
args = parser.parse_args()

p = Pool(args.num_process)
for use_penetration in range(2):
    # for interaction in interaction_names:
    for interaction in ['lie on-sofa']:
        command = "python synthesize.py --gender {gender} --num_results {num_results} --interaction \'{interaction}\' --num_translations {transl}  " \
                  "--num_skeletons {skeletons} --use_penetration {penetration}".format(
            gender='neutral',
            num_results=10,
            interaction=interaction,
            transl=1024,
            skeletons=16,
            penetration=use_penetration,
        )
        if not local_machine:
            command = 'source ../cluster/launch_cluster.sh ' + command
        print(command)
        p.apply_async(eval_command, args=(command,))
        # break
p.close()
p.join()
