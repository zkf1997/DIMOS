import sys
# sys.path.append('../human_body_prior/src')
from configuration.config import *

# from human_body_prior.tools.rotation_tools import matrot2aa
from data.utils import matrot2aa

import smplx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""

    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_block(x)


class FCResBlock(nn.Module):
    """Residual block using fully-connected layers."""

    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCResBlock, self).__init__()
        self.fc_block = nn.Sequential(nn.Linear(in_size, out_size),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_size, out_size),
                                      nn.BatchNorm1d(out_size))

    def forward(self, x):
        return F.relu(x + self.fc_block(x))

class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)



class SMPLX_Regressor(nn.Module):
    """Regress SMPLX body params from mesh vertices"""
    def __init__(self, mesh, args):
        super(SMPLX_Regressor, self).__init__()

        num_vertices = mesh.num_vertices[mesh.num_downsampling]
        in_features = num_vertices * 3 * 2

        self.layers = nn.Sequential(FCBlock(in_features, 1024),
                                    FCResBlock(1024, 1024),
                                    FCResBlock(1024, 1024),
                                    nn.Linear(1024, 22 * 6 + 10 + 12))

        self.rot_decoder = ContinousRotReprDecoder()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        pred = self.layers(x)

        rotmat = self.rot_decoder(pred[:, :132].reshape(-1, 3, 2))
        axis_angle = matrot2aa(rotmat.view(-1, 3, 3)).view(batch_size, -1, 3)

        result = {}
        result['global_orient'] = axis_angle[:, 0, :].view(batch_size, 3)
        result['body_pose'] = axis_angle[:, 1:, :].reshape(batch_size, 63)
        result['rotmat'] = rotmat
        # result['transl'] = pred[:, 132:135]
        result['transl'] = torch.zeros_like(result['global_orient'])
        result['left_hand_pose'] = pred[:, 132:138]
        result['right_hand_pose'] = pred[:, 138:144]
        result['betas'] = pred[:, 144:154]
        result['nonrot'] = pred[:, 132:154]

        return result


class SMPLX_Regressor_Joint(nn.Module):
    """Regress SMPLX body params from joints locations"""
    def __init__(self, num_joints=55):
        super(SMPLX_Regressor_Joint, self).__init__()

        in_features = num_joints * 3 * 2

        self.layers = nn.Sequential(FCBlock(in_features, 1024),
                                    FCResBlock(1024, 1024),
                                    FCResBlock(1024, 1024),
                                    nn.Linear(1024, 22 * 6 + 10 + 12))

        self.rot_decoder = ContinousRotReprDecoder()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        pred = self.layers(x)

        rotmat = self.rot_decoder(pred[:, :132].reshape(-1, 3, 2))
        axis_angle = matrot2aa(rotmat.view(-1, 3, 3)).view(batch_size, -1, 3)

        result = {}
        result['global_orient'] = axis_angle[:, 0, :].view(batch_size, 3)
        result['body_pose'] = axis_angle[:, 1:, :].reshape(batch_size, 63)
        result['rotmat'] = rotmat
        # result['transl'] = pred[:, 132:135]
        result['transl'] = torch.zeros_like(result['global_orient'])
        result['left_hand_pose'] = pred[:, 132:138]
        result['right_hand_pose'] = pred[:, 138:144]
        result['betas'] = pred[:, 144:154]
        result['nonrot'] = pred[:, 132:154]

        return result

class SMPLX_Regressor_Joint_Orient(nn.Module):
    """Regress SMPLX body params from joint locations and orientations"""
    def __init__(self, num_joints=55, num_dim=9):
        super(SMPLX_Regressor_Joint_Orient, self).__init__()

        in_features = num_joints * num_dim

        self.layers = nn.Sequential(FCBlock(in_features, 1024),
                                    FCResBlock(1024, 1024),
                                    FCResBlock(1024, 1024),
                                    nn.Linear(1024, 10 + 12))

        self.rot_decoder = ContinousRotReprDecoder()

    def forward(self, x):
        batch_size = x.shape[0]
        rotmat = self.rot_decoder(x[:, :22, 3:9].reshape(-1, 3, 2))
        axis_angle = matrot2aa(rotmat.view(-1, 3, 3)).view(batch_size, 22, 3)
        pred = self.layers(x.reshape(batch_size, -1))

        result = {}
        result['global_orient'] = axis_angle[:, 0, :].view(batch_size, 3)
        result['body_pose'] = axis_angle[:, 1:, :].reshape(batch_size, 63)
        result['rotmat'] = rotmat
        # result['transl'] = pred[:, 132:135]
        result['transl'] = torch.zeros_like(result['global_orient'])
        result['left_hand_pose'] = pred[:, :6]
        result['right_hand_pose'] = pred[:, 6:12]
        result['betas'] = pred[:, 12:]
        result['nonrot'] = pred

        return result