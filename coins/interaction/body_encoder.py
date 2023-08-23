import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration.config import *
from graph_layers import GraphResBlock, GraphLinear, spmm, batch_sparse_dense_matmul

class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        nn.init.kaiming_uniform_(self.mu.weight)
        nn.init.uniform_(self.mu.bias)
        self.logvar = nn.Linear(num_feat_in, latentD)
        nn.init.kaiming_uniform_(self.logvar.weight)
        nn.init.uniform_(self.logvar.bias)

    def forward(self, Xout):
        Xout = Xout.squeeze()
        return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))
        # return torch.distributions.normal.Normal(self.mu(Xout), torch.exp(0.5 * self.logvar(Xout)))

class ds_us_fn(nn.Module):
    def __init__(self, M):
        super(ds_us_fn, self).__init__()
        self.M = M

    def forward(self, x):
        # print(self.M.shape, x.shape)
        # return torch.matmul(self.M, x.transpose(1, 2)).transpose(1, 2)
        if x.ndimension() < 3:
            x = x.transpose()
            x = spmm(self.M, x)
        elif x.ndimension() == 3:
            B, C, V = x.shape
            x = x.permute(2, 0, 1).reshape(V, B*C)  #BxCxV -> VxBxC ->Vx(B*C)
            x = torch.matmul(self.M, x).reshape(-1, B, C).permute(1, 2, 0)  # V'x(B*C) -> BxCxV'
        return x


class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""

    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Conv1d(in_size, out_size, 1)]
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
        self.fc_block = nn.Sequential(nn.Conv1d(in_size, out_size, 1),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(out_size, out_size, 1),
                                      nn.BatchNorm1d(out_size))

    def forward(self, x):
        return F.relu(x + self.fc_block(x))

class BodyEncoder(nn.Module):
    def __init__(self, mesh, args):
        super(BodyEncoder, self).__init__()
        self.mesh = mesh
        # self.ref_vertices = nn.Parameter(mesh.ref_vertices.permute(1, 0), requires_grad=False)  # 3 * 10475
        self.ref_vertices = nn.Parameter(mesh.ref_vertices_by_level(args.final_downsample_level).permute(1, 0),
                                         requires_grad=False)
        self.ref_vertices_init = mesh.ref_vertices_by_level(args.init_downsample_level).permute(1, 0)
        self.num_vertices = self.ref_vertices.shape[-1]
        self.args = args

        # graph CVAE
        # input_level = 1
        init_level = self.args.init_downsample_level
        final_level = self.args.final_downsample_level
        encoder_channels = self.args.encoder_channels
        encoder_layers = nn.ModuleList()
        encoder_layers.append(GraphLinear(3 + num_noun + 1 + num_verb * num_noun, encoder_channels))
        # encoder_layers.append(GraphLinear(64, encoder_channels))
        for mesh_level in range(init_level, final_level):
            for _ in range(args.conv_per_level):
                encoder_layers.append(
                    GraphResBlock(encoder_channels, encoder_channels, self.mesh._A[mesh_level])
                )
            encoder_layers.append(
                ds_us_fn(self.mesh._D[mesh_level])
            )
            # print(self.mesh._D[mesh_level].shape)
        for _ in range(args.conv_per_level):
            encoder_layers.append(
                GraphResBlock(encoder_channels, encoder_channels, self.mesh._A[final_level])
            )
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_linear = nn.Linear(self.mesh.num_vertices[final_level] * encoder_channels, args.latent_dimension)

        self.dist_decoder = NormalDistDecoder(self.mesh.num_vertices[final_level] * encoder_channels, args.latent_dimension)

        decoder_channels = self.args.decoder_channels
        decoder_layers = nn.ModuleList(
            [GraphLinear(3 + args.latent_dimension + num_verb * num_noun, decoder_channels)]  # concatenate
        )
        for mesh_level in reversed(range(init_level + 1, final_level + 1)):
            for _ in range(args.conv_per_level):
                decoder_layers.append(
                    GraphResBlock(decoder_channels, decoder_channels, self.mesh._A[mesh_level])
                )
            decoder_layers.append(
                ds_us_fn(self.mesh._U[mesh_level - 1])
            )
            # print(self.mesh._U[mesh_level - 1].shape)
        for _ in range(args.conv_per_level):
            decoder_layers.append(
                GraphResBlock(decoder_channels, decoder_channels, self.mesh._A[init_level])
            )
        decoder_layers += [
            GraphResBlock(decoder_channels, 64, self.mesh._A[init_level]),
            GraphResBlock(64, 64, self.mesh._A[init_level]),
            nn.GroupNorm(64 // 8, 64),
            nn.ReLU(inplace=True),
            GraphLinear(64, 3 + num_noun + 1),
        ]
        self.decoder = nn.Sequential(*decoder_layers)
        if args.residual:
            self.residual_net = nn.Sequential(FCBlock(3 + num_verb * num_noun, 512),
                                        FCResBlock(512, 512),
                                        FCResBlock(512, 512),
                                        nn.Conv1d(512, num_noun + 1, 1))

    def forward(self, body_vertices, contact_features, interaction_code):
        """Forward pass
        """
        batch_size, num_vertices = body_vertices.shape[:2]
        body_vertices = body_vertices.transpose(1, 2)  #  transpose to (batch, channel, nodes)

        feature = self.encoder(
            torch.cat([body_vertices,
                       contact_features.transpose(1, 2),
                       interaction_code.unsqueeze(2).expand(-1, -1, num_vertices)], dim=1)
        )
        if self.args.model == 'AE':
            z = self.latent_linear(feature.reshape(batch_size, -1))
            z_dist = None
        else:
            z_dist = self.dist_decoder(feature.reshape(batch_size, -1))
            z = z_dist.rsample()
        decoder_input = torch.cat((z.unsqueeze(2).expand(-1, -1, self.num_vertices),
                                   interaction_code.unsqueeze(2).expand(-1, -1, self.num_vertices),
                                   self.ref_vertices.unsqueeze(0).expand(batch_size, -1, -1)), dim=1)

        pred = self.decoder(decoder_input).transpose(1, 2)  # (batch, channel, nodes) -> (batch, nodes, channel)
        x_rec = pred[:, :, :3]
        pred_f = pred[:, :, 3:]
        if self.args.residual:
            residual_f = self.residual_net(
                torch.cat([interaction_code.unsqueeze(2).expand(-1, -1, self.mesh.num_vertices[self.args.init_downsample_level]),
                           self.ref_vertices_init.unsqueeze(0).expand(batch_size, -1, -1)
                           ], dim=1)  # Bx(3+4*42)xV
            ).transpose(1, 2)
            # print(residual_f.shape, pred_f.shape)
            pred_f = pred_f + residual_f
        f = torch.cat((torch.sigmoid(pred_f[:, :, 0]).unsqueeze(-1),
                       pred_f[:, :, 1:]), dim=-1)
        return x_rec, f, z_dist

    def sample(self, batch_size, interaction_code):
        assert self.args.model == 'VAE'
        set_training = False
        if self.training:
            set_training = True
            self.eval()
        z = torch.distributions.normal.Normal(
            loc=torch.zeros((batch_size, self.args.latent_dimension), requires_grad=False, device=self.args.device),
            scale=torch.ones((batch_size, self.args.latent_dimension), requires_grad=False,
                             device=self.args.device)).rsample()
        decoder_input = torch.cat((z.unsqueeze(2).expand(-1, -1, self.num_vertices),
                                   interaction_code.unsqueeze(2).expand(-1, -1, self.num_vertices),
                                   self.ref_vertices.unsqueeze(0).expand(batch_size, -1, -1)), dim=1)

        pred = self.decoder(decoder_input).transpose(1, 2)  # (batch, channel, nodes) -> (batch, nodes, channel)
        x_rec = pred[:, :, :3]
        pred_f = pred[:, :, 3:]
        if self.args.residual:
            residual_f = self.residual_net(
                torch.cat([interaction_code.unsqueeze(2).expand(-1, -1, self.mesh.num_vertices[self.args.init_downsample_level]),
                           self.ref_vertices_init.unsqueeze(0).expand(batch_size, -1, -1)
                           ], dim=1)  # Bx(3+4*42)xV
            ).transpose(1, 2)
            pred_f = pred_f + residual_f
        f = torch.cat((torch.sigmoid(pred_f[:, :, 0]).unsqueeze(-1),
                       pred_f[:, :, 1:]), dim=-1)
        if set_training:
            self.train()
        return x_rec, f
