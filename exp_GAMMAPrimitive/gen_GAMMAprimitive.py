'''
this script is to evaluate stochastic motion prediction on amass
'''
import numpy as np
import argparse
import os
import sys
import pickle
import csv
import torch

sys.path.append(os.getcwd())
from utils import *
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized
from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP as GenOP
import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None) # specify the model to evaluate
    parser.add_argument('--testdata', default='chair') # which dataset to evaluate? choose only one
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    '''''dont touch these two, used for all exps'''
    N_SEQ = 10 #for each gender
    N_GEN = 3

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfgall = ConfigCreator(args.cfg)
    modelcfg = cfgall.modelconfig
    losscfg = cfgall.lossconfig
    traincfg = cfgall.trainconfig
    predictorcfg = ConfigCreator(modelcfg['predictor_config'])
    regressorcfg = ConfigCreator(modelcfg['regressor_config'])
    t_his = predictorcfg.modelconfig['t_his']
    
    load_pretrained_model=True
    testcfg = {}
    testcfg['gpu_index'] = args.gpu_index
    testcfg['ckpt_dir'] = traincfg['save_dir']
    testcfg['testdata'] = args.testdata
    testcfg['result_dir'] = predictorcfg.cfg_result_dir if load_pretrained_model else cfgall.cfg_result_dir
    testcfg['seed'] = args.seed
    testcfg['log_dir'] = cfgall.cfg_log_dir
    testcfg['training_mode'] = False
    testcfg['batch_size'] = N_GEN
    # testcfg['ckpt'] = args.ckpt


    """data"""
    testing_data = [args.testdata]

    if len(testing_data)>1:
        raise NameError('performing testing per dataset please.')
    from exp_GAMMAPrimitive.utils import config_env
    # amass_path = config_env.get_amass_canonicalized_path()
    amass_path = traincfg['dataset_path']
    batch_gen = BatchGeneratorAMASSCanonicalized(amass_data_path=amass_path,
                                                 amass_subset_name=testing_data,
                                                 sample_rate=1,
                                                 body_repr=predictorcfg.modelconfig['body_repr'],
                                                 read_to_ram=False)
    try:
        rec_list_file = 'exp_GAMMAPrimitive/data/exp-motiongen/motionseed_{}.pkl'.format(args.testdata)
        rec_list = np.load(rec_list_file, allow_pickle=True)
        batch_gen.rec_list = [filepath.replace('/home/yzhang/Videos/AMASS-Canonicalized-MP/data',
                        amass_path) for filepath in rec_list]
        batch_gen.index_rec = 0
        print('[INFO] load rec_list from {}'.format(rec_list_file))
    except:
        batch_gen.get_rec_list(shuffle_seed=args.seed)
        print('[INFO] load rec_list from dataset. This is not recommended for comparison')

    """models"""
    testop = GenOP(predictorcfg, regressorcfg, testcfg)
    testop.build_model(load_pretrained_model=load_pretrained_model)

    """eval"""
    testop.generate_primitive_to_files(batch_gen,
                    n_seqs=N_SEQ, n_gens=testcfg['batch_size'],t_his=t_his)
