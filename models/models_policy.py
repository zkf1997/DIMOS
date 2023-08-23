import os, sys, glob
import time
from typing import NamedTuple
import random
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim

import pickle
import json
import pdb
from tensorboardX import SummaryWriter

from models.baseops import MLP


class MLPBlock(nn.Module):
    def __init__(self, h_dim, out_dim, n_blocks, actfun='relu', residual=True):
        super(MLPBlock, self).__init__()
        self.residual = residual
        self.layers = nn.ModuleList([MLP(h_dim, h_dims=(h_dim, h_dim),
                                        activation=actfun)
                                        for _ in range(n_blocks)]) # two fc layers in each MLP
        self.out_fc = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h = x
        for layer in self.layers:
            r = h if self.residual else 0
            h = layer(h) + r
        y = self.out_fc(h)
        return y

class MAPEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_blocks, actfun='relu', residual=True):
        super(MAPEncoder, self).__init__()
        self.residual = residual
        self.layers = nn.ModuleList([MLP(in_dim, h_dims=(h_dim, h_dim),
                                        activation=actfun)]) # two fc layers in each MLP
        for _ in range(n_blocks - 1):
            self.layers.append(MLP(h_dim, h_dims=(h_dim, h_dim),
                activation=actfun))
        # self.out_fc = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h = x
        for layer_idx, layer in enumerate(self.layers):
            r = h if self.residual and layer_idx > 0 else 0
            h = layer(h) + r
        # y = torch.nn.LeakyReLU()(self.out_fc(h)) + h
        y = h
        return y

class PointNetEncoder(nn.Module):
    def __init__(self, channel=3):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, D, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        x = torch.nn.LeakyReLU()(x)
        return x

class GAMMAPolicy(nn.Module):
    '''
    the network input is the states:
        [vec_to_target_marker, vec_to_walking_path]
    the network output is the distribution of z, i.e. N(mu, logvar)
    '''
    def __init__(self, config):
        super(GAMMAPolicy, self).__init__()
        self.h_dim = config['h_dim']
        self.z_dim = config['z_dim']
        self.n_blocks = config['n_blocks']
        self.n_recur = config['n_recur'] # n_recur=0 means no recursive scheme
        self.actfun = config['actfun']
        self.is_stochastic = config.get('is_stochastic', True)
        self.min_logvar = config.get('min_logvar', -1)
        self.max_logvar = config.get('max_logvar', 3)

        if config['body_repr'] in {'ssm2_67_condi_marker', 'ssm2_67_condi_marker_orient', 'ssm2_67_condi_marker_map','ssm2_67_condi_marker_l2norm', 'ssm2_67_condi_marker_height'}:
            self.in_dim = 67*3*2
        elif config['body_repr'] == 'ssm2_67_condi_wpath':
            self.in_dim = 67*3+2
        elif config['body_repr'] == 'ssm2_67_condi_wpath_height':
            self.in_dim = 67*3+3
        elif config['body_repr'] == 'ssm2_67_condi_marker_ray':
            self.in_dim = 67 * 3 * 2 + config['ray_theta_num'] * config['ray_phi_num']
        elif config['body_repr'] in ['ssm2_67_condi_marker_dist', 'ssm2_67_condi_marker_dist_orient', 'ssm2_67_condi_marker_dist_pointcloud']:
            self.in_dim = 67 * 3 * 2 + 67
        elif config['body_repr'] in ['ssm2_67_condi_marker_inter']:
            self.in_dim = 67 * (3 + 3 + 3 + 1 + 1)
        elif config['body_repr'] == 'ssm2_67_condi_marker_pointcloud':
            self.in_dim = 67 * 3 * 2
        elif config['body_repr'] == 'ssm2_67_marker_orient':
            self.in_dim = 67 * 3
        else:
            raise NotImplementedError('other body_repr is not implemented yet.')
        self.use_obj_encoding = 'pointcloud' in config['body_repr']
        self.obj_dimension = 256 if self.use_obj_encoding else 0
        self.use_orient = 'orient' in config['body_repr']
        self.orient_dimension = 2 if self.use_orient else 0
        self.use_map = 'map' in config['body_repr']
        self.map_embedding_dim = self.h_dim if self.use_map else 0
        if self.use_map:
            self.dump_map = config['dump_map']
            self.map_dim = config['map_dim']
            self.map_encoder = MAPEncoder(in_dim=self.map_dim, h_dim=self.h_dim,
                                          out_dim=self.h_dim, n_blocks=self.n_blocks,
                                          actfun=self.actfun)


        ## first a gru to encode X
        self.x_enc = nn.GRU(self.in_dim, self.h_dim)

        ## about the policy network
        self.pnet = MLPBlock(self.h_dim + self.obj_dimension + self.orient_dimension + self.map_embedding_dim,
                            self.z_dim*2 if self.is_stochastic else self.z_dim,
                            self.n_blocks,
                            actfun=self.actfun)
        ## about the value network
        self.vnet = MLPBlock(self.h_dim + self.obj_dimension + self.orient_dimension + self.map_embedding_dim,
                            1,
                            self.n_blocks,
                            actfun=self.actfun)
        if self.use_obj_encoding:
            self.obj_encoder = PointNetEncoder()

    def forward(self, x_in, obj_points=None, target_ori=None, local_map=None):
        '''
        x_in has
        - vec_to_ori:    [t, batch, dim=201]
        - vec_to_target: [t, batch, dim=201]
        - vec_to_wpath:  [t, batch, dim=2]
        '''
        nt, nb, _ = x_in.shape
        _, hx = self.x_enc(x_in)
        hx = hx[0] #[b, d]
        if self.use_obj_encoding:
            obj_encoding = self.obj_encoder(obj_points)  #[b, 256]
            hx = torch.cat([hx, obj_encoding], dim=-1)  # [b, d+256]
        if self.use_orient:
            hx = torch.cat([hx, target_ori], dim=-1)  # [b, d+2]
        if self.use_map:
            map_embedding = torch.zeros(nb, self.h_dim).to(device=x_in.device) if self.dump_map else self.map_encoder(local_map.reshape(nb, self.map_dim))
            # map_embedding = torch.zeros(nb, self.h_dim).to(device=x_in.device)  # test dump map embedding
            hx = torch.cat([hx, map_embedding], dim=-1)  # [b, d+d]
        z_prob = self.pnet(hx)
        # z_prob[:,:self.z_dim] = torch.tanh(z_prob[:,:self.z_dim])*4
        val = self.vnet(hx)
        if self.is_stochastic:
            mu = z_prob[:,:self.z_dim]
            logvar = z_prob[:, self.z_dim:]
            return mu, logvar, val
        else:
            return z_prob, val











