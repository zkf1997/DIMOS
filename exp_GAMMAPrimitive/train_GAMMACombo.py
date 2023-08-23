import os
import sys
import math
import pickle
import argparse
import time
import torch
import numpy as np
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/models')
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized

import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--resume_training', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """load the right model"""
    from models.models_GAMMA_primitive import GAMMAPrimitiveComboTrainOP

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

    traincfg['resume_training'] = True if args.resume_training==1 else False
    traincfg['verbose'] = True if args.verbose==1 else False
    traincfg['gpu_index'] = args.gpu_index


    """data"""
    batch_gen = BatchGeneratorAMASSCanonicalized(amass_data_path=traincfg['dataset_path'],
                                                 amass_subset_name=traincfg['subsets'],
                                                 sample_rate=3,
                                                 body_repr=predictorcfg.modelconfig['body_repr'],
                                                 read_to_ram=False)
    batch_gen.get_rec_list()

    """model and trainop"""
    trainop = GAMMAPrimitiveComboTrainOP(predictorcfg, regressorcfg, losscfg, traincfg)

    trainop.train(batch_gen)


