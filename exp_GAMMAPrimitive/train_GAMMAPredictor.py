import os
import sys
import math
import pickle
import argparse
import time
import torch
import numpy as np
sys.path.append(os.getcwd())
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized
from exp_GAMMAPrimitive.utils.config_env import *

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
    from models.models_GAMMA_primitive import *

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfgall = ConfigCreator(args.cfg)
    modelcfg = cfgall.modelconfig
    losscfg = cfgall.lossconfig
    traincfg = cfgall.trainconfig

    traincfg['resume_training'] = True if args.resume_training==1 else False
    traincfg['verbose'] = True if args.verbose==1 else False
    traincfg['gpu_index'] = args.gpu_index

    """data"""
    # if args.resume_training==0:
    #     amass_data_path = get_amass_canonicalized_path()
    # else:
    #     amass_data_path = get_amass_canonicalizedx10_path()
    amass_data_path = traincfg['dataset_path']
    # noise = traincfg.get('noise', None)
    batch_gen = BatchGeneratorAMASSCanonicalized(amass_data_path=amass_data_path,
                                                 amass_subset_name=traincfg['subsets'],
                                                 sample_rate=1,
                                                 body_repr=modelcfg['body_repr'],
                                                 )
    batch_gen.get_rec_list(to_gpu=True)

    """model and trainop"""
    trainop = GAMMAPrimitiveVAETrainOP(modelcfg, losscfg, traincfg)
    trainop.train(batch_gen)


