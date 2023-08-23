"""
Naive shape parameter distribution. Used by PiGraph.
"""

import sys
sys.path.append('..')
from configuration.config import *

import numpy as np
import pickle

with open(Path.joinpath(project_folder, "data", 'train.pkl'), 'rb') as data_file:
    train_data = pickle.load(data_file)
shape_params = np.asarray([record['smplx_param']['betas'] for record in train_data]).reshape((-1, 10))
# print(shape_params.shape)
shape_params = np.unique(shape_params, axis=0)
# print(shape_params.shape)
shape_mean = np.mean(shape_params, axis=0)
shape_cov = np.cov(shape_params, rowvar=0)
shape_distribution = {'mean': shape_mean, 'cov': shape_cov}
# print(np.random.multivariate_normal(**shape_distribution, size=4))

def sample_betas(size=1):
    return np.random.multivariate_normal(**shape_distribution, size=size).astype(np.float32)

