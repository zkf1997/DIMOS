import sys
sys.path.append('..')
from configuration.config import *

import smplx
import torch
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
body_model_dict = {
        'male': smplx.create(smplx_model_folder, model_type='smplx',
                             gender='male', ext='npz',
                             num_pca_comps=num_pca_comps).to(device),
        'female': smplx.create(smplx_model_folder, model_type='smplx',
                               gender='female', ext='npz',
                               num_pca_comps=num_pca_comps).to(device),
        'neutral': smplx.create(smplx_model_folder, model_type='smplx',
                               gender='neutral', ext='npz',
                               num_pca_comps=num_pca_comps).to(device)
    }