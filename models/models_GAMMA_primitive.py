import time
import torch
import numpy as np
import os, sys, glob
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import pickle
import json
import random
import pdb
from tensorboardX import SummaryWriter
from typing import Union
from typing import Tuple

from models.baseops import MLP
from models.baseops import VAE
from models.baseops import TrainOP
from models.baseops import TestOP
from models.baseops import get_scheduler
from models.baseops import get_logger
from models.baseops import get_body_model
from models.baseops import CanonicalCoordinateExtractor
from models.baseops import RotConverter
from exp_GAMMAPrimitive.utils.config_env import *

import matplotlib.pyplot as plts


def slerp_blend(a, b):
    times = [0, 1]

    inter_times = [0.5]

class GAMMAPrimitiveVAE(VAE):
    """the marker predictor in GAMMA.

    GAMMA contains two basic modules, the marker predictor and the body regressor.
    This marker predictor takes a motion seed as input, and produces future frames in a motion primitive.
    In addition, it can take a goal as an extra condition, so as to produce goal-driven motions.

    """
    def __init__(self, configs):
        super(GAMMAPrimitiveVAE, self).__init__()
        self.body_repr = body_repr = configs['body_repr']
        if body_repr == 'ssm2_67':
            self.in_dim = in_dim = 67*3 # the marker dim
            self.c_dim = c_dim = in_dim
        elif body_repr in ['ssm2_67_condi_marker2tarloc']: #when close to the target
            self.in_dim = in_dim = 67*3 # the marker dim
            self.c_dim = c_dim = 67*3*2 # the marker dim, the vec_to_target dim

        self.h_dim = h_dim = configs['h_dim'] #256
        self.z_dim = z_dim = configs['z_dim']
        self.use_drnn_mlp = configs['use_drnn_mlp']
        self.hdims_mlp = hdims_mlp = configs['hdims_mlp'] #[512, 256]
        self.residual = configs['residual']

        # encode
        self.x_enc = nn.GRU(c_dim, h_dim)
        self.e_rnn = nn.GRU(in_dim, h_dim)
        self.e_mlp = MLP(2*h_dim, hdims_mlp, activation='tanh')
        self.e_mu = nn.Linear(self.e_mlp.out_dim, z_dim)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, z_dim)

        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(h_dim, hdims_mlp + [h_dim], activation='tanh')
        self.d_rnn = nn.GRUCell(in_dim + z_dim + h_dim, h_dim)
        self.d_mlp = MLP(h_dim, hdims_mlp, activation='tanh')
        self.d_out = nn.Linear(self.d_mlp.out_dim, in_dim)


    def encode(self, x, y):
        _, hx = self.x_enc(x)
        _, hy = self.e_rnn(y)
        h = torch.cat((hx[0], hy[0]), dim=-1)
        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)


    def decode(self, x, z, t_pred):
        _, hx = self.x_enc(x)
        hx = hx[0] #[b, d]
        if self.use_drnn_mlp:
            h_rnn = self.drnn_mlp(hx)
        else:
            h_rnn = hx
        y = []
        for i in range(t_pred):
            y_p = x[-1][:,:self.in_dim] if i==0 else y_i
            rnn_in = torch.cat([hx, z, y_p], dim=-1)
            h_rnn = self.d_rnn(rnn_in, h_rnn)
            hfc = self.d_mlp(h_rnn)
            y_i = self.d_out(hfc)
            if self.residual:
                y_i += y_p
            y.append(y_i)
        y = torch.stack(y)
        return y


    def forward(self, x, y):
        t_pred = y.shape[0]
        mu, logvar = self.encode(x, y)
        z = self._sample(mu, logvar)
        y_pred = self.decode(x, z, t_pred)

        return y_pred, mu, logvar


    def sample_prior(self,
                    x: torch.Tensor,
                    z: Union[torch.Tensor, None] = None,
        ) -> torch.Tensor:
        """sample multiple motion primitives based on the same motion seed x.
            only used for during testing
        Args:
            x: the input motion seed, with shape [t_his,b,d]
            z: the latent variable, either None or [b, d_dim]

        Returns:
            the predicted markers, with shape [t_pred,b,d]

        Raises:
            None
        """
        if z is None:
            z = torch.randn((x.shape[1], self.z_dim), device=x.device)
        t_pred = 10-x.shape[0]
        y_pred = self.decode(x, z, t_pred)
        return y_pred


    def sample_prior_return_z(self,
                            x: torch.Tensor,
        ) -> Tuple[torch.Tensor]:
        """sample multiple motion primitives based on the same motion seed x.
        And the employed latent variables are returned
            only used for during testing
        Args:
            x: the input motion seed, with shape [t_his,b,d]

        Returns:
            a list with
            - the predicted markers, with shape [t_pred,b,d]
            - the employed latent variables with shape [b, d]

        Raises:
            None
        """
        z = torch.randn((x.shape[1], self.z_dim), device=x.device)
        y_pred = self.decode(x, z)

        return y_pred, z



class ResNetBlock(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_blocks, actfun='relu'):
        super(ResNetBlock, self).__init__()

        self.in_fc = nn.Linear(in_dim, h_dim)
        self.layers = nn.ModuleList([MLP(h_dim, h_dims=(h_dim, h_dim),
                                        activation=actfun)
                                        for _ in range(n_blocks)]) # two fc layers in each MLP
        self.out_fc = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h = self.in_fc(x)
        for layer in self.layers:
            h = layer(h)+h
        y = self.out_fc(h)
        return y


class MoshRegressor(nn.Module):
    """the body regressor in GAMMA.

    GAMMA contains two basic modules, the marker predictor and the body regressor.
    This body regressor takes a batch of primitive markers as well as the betas, and produces the corresponding body parameters
    The body parameter vector is in the shape of (b, 93), including the translation, global_orient, body_pose and hand_pose.

    Gender is assumed to be pre-fixed when using the body regressor.

    """
    def __init__(self, config):
        super(MoshRegressor, self).__init__()
        if config['body_repr'] == 'ssm2_67':
            self.in_dim = 67*3 # the marker dim
        self.h_dim = config['h_dim']
        self.n_blocks = config['n_blocks']
        self.n_recur = config['n_recur']
        self.body_shape_dim = 10
        self.actfun = config['actfun']
        self.use_cont = config.get('use_cont', False)
        if self.use_cont:
            self.body_dim = 3 + 1*6 + 21*6 + 24
        else:
            self.body_dim = 3 + 1*3 + 21*3 + 24

        ## about the policy network
        self.pnet = ResNetBlock(self.in_dim+self.body_dim+self.body_shape_dim,
                                self.h_dim, self.body_dim, self.n_blocks,
                                actfun=self.actfun)

    def _cont2aa(self, xb):
        transl = xb[:,:3]
        body_ori_and_pose_cont = xb[:,3:3+22*6].contiguous()
        body_ori_and_pose_aa = RotConverter.cont2aa(body_ori_and_pose_cont.view(transl.shape[0],-1,6)
                                            ).reshape(xb.shape[0],-1)
        global_orient = body_ori_and_pose_aa[:,:3]
        body_pose = body_ori_and_pose_aa[:,3:]
        left_hand_pose = xb[:,3+22*6:3+22*6+12]
        right_hand_pose = xb[:,3+22*6+12:]
        out = torch.cat([transl,global_orient,body_pose,
                            left_hand_pose,right_hand_pose],dim=-1)
        return out


    def _forward(self,
                x_ref: torch.Tensor,
                prev_transl: torch.Tensor,
                prev_glorot: torch.Tensor,
                prev_theta: torch.Tensor,
                prev_lefthandpose: torch.Tensor,
                prev_righthandpose: torch.Tensor,
                betas: torch.Tensor,
        ) -> torch.Tensor:
        """the regressor used inside of this class.
        all inputs are in torch.FloatTensor on cuda

        Args:
            - x_ref: the target markers, [b, nk, 3]
            - prev_transl: [b, 3]
            - prev_glorot: [b, 3 or 6], axis angle or cont rot
            - prev_theta: [b, 63 or 126], 21 joints rotation, axis angle or cont rot
            - prev_left_hand_pose: [b, 12], hand pose in the pca space
            - prev_right_hand_pose: [b, 12], hand pose in the pca space
            - betas: [b,10] body shape

        Returns:
            - the body parameter vector (b, 93)

        Raises:
            None
        """
        xb = torch.cat([prev_transl, prev_glorot,
                        prev_theta,
                        prev_lefthandpose,
                        prev_righthandpose],
                        dim=-1)
        xr = x_ref.reshape(-1, self.in_dim)

        for _ in range(self.n_recur):
            xb = self.pnet(torch.cat([xr, xb, betas],dim=-1)) + xb

        return xb


    def forward(self,
                marker_ref: torch.Tensor,
                betas: torch.Tensor,
        ) -> torch.Tensor:
        """the regressor forward pass
        all inputs are in torch.FloatTensor on cuda
        all inputs are from the same gender.

        Args:
            - marker_ref: the target markers, [b, nk, 3]
            - betas: [b,10] body shape

        Returns:
            - the body parameter vector (b, 93), rotations are in axis-angle (default) or cont6d

        Raises:
            None
        """
        ## initialize variables
        n_meshes = marker_ref.shape[0]
        batch_transl = torch.zeros(n_meshes,3).to(marker_ref.device)
        if self.use_cont:
            batch_glorot = torch.zeros(n_meshes,6).to(marker_ref.device)
            body_pose = torch.zeros(n_meshes,21*6).to(marker_ref.device)
        else:
            batch_glorot = torch.zeros(n_meshes,3).to(marker_ref.device)
            body_pose = torch.zeros(n_meshes,21*3).to(marker_ref.device)
        left_hand_pose = torch.zeros(n_meshes,12).to(marker_ref.device)
        right_hand_pose = torch.zeros(n_meshes,12).to(marker_ref.device)
        ## forward pass
        xb = self._forward(marker_ref,
                            batch_transl, batch_glorot, body_pose,
                            left_hand_pose, right_hand_pose,
                            betas)
        if self.use_cont:
            out = self._cont2aa(xb)
        else:
            out = xb

        return out




import pdb
class GAMMAPrimitiveCombo(VAE):
    """the wrapper of GAMMA, which contains a marker predictor and a body regressor

    This body regressor takes a batch of primitive markers as well as the betas, and produces the corresponding body parameters
    The body parameter vector is in the shape of (b, 93), including the translation, global_orient, body_pose and hand_pose.

    t_his = 1 or 2 is pre-specified. This will be updated
    Gender is assumed to be pre-fixed when using the body regressor.

    """
    def __init__(self, markercfg, bparamscfg):
        super(GAMMAPrimitiveCombo, self).__init__()
        # marker predictor
        self.predictor = GAMMAPrimitiveVAE(markercfg)
        # mosh regressor
        self.regressor = MoshRegressor(bparamscfg)

    def forward(self, X, Y, betas):
        ## forward pass of marker
        [Y_rec, mu, logvar] = self.predictor(X,Y)
        ## forward pass of regressor
        nt, nb = Y_rec.shape[:2]
        Xb_rec = self.regressor(Y_rec.contiguous().view(nt*nb, -1),
                                betas.contiguous().view(nt*nb, -1))
        Xb_rec = Xb_rec.view(nt, nb, -1) # we get aa rotations automatically
        return Y_rec, mu, logvar, Xb_rec

    def sample_prior(self,
                    X: torch.Tensor,
                    betas: torch.Tensor,
                    z: Union[None, torch.Tensor]=None,
        ) -> Tuple[torch.Tensor, torch.Tensor]: #[t, b, d]
        """generate motion primitives based on a motion seed X
        all inputs are in torch.FloatTensor on cuda
        all inputs are from the same gender.

        Args:
            - X: the motion seed, with the shape [t, b, d]
            - betas: [b,10] body shape
            - z: when None, we get random samples. Or we get the motions corresponding to z.

        Returns:
            a list containing following:
            - Y_gen: the predicted markers
            - Yb_gen: the predicted-and-regressed body parameters. Rotations in axis-angle

        Raises:
            None
        """
        Y_gen = self.predictor.sample_prior(X, z)
        nt, nb = Y_gen.shape[:2]
        Yb_gen = self.regressor(Y_gen.view(nt*nb, -1), betas.view(nt*nb, -1))
        Yb_gen = Yb_gen.view(nt, nb, -1)
        return Y_gen, Yb_gen

    def sample_prior_return_z(self,
                            X: torch.Tensor,
                            betas: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]: #[t, b, d]
        """generate motion primitives based on a motion seed X, it also returns the latent variable z
        all inputs are in torch.FloatTensor on cuda
        all inputs are from the same gender.

        Args:
            - X: the motion seed, with the shape [t, b, d]
            - betas: [b,10] body shape

        Returns:
            a list containing following:
            - Y_gen: the predicted markers
            - Yb_gen: the predicted-and-regressed body parameters. Rotations in axis-angle
            - z: the returned random variable z
        Raises:
            None
        """
        Y_gen, z = self.predictor.sample_prior_return_z(X)
        nt, nb = Y_gen.shape[:2]
        Yb_gen = self.regressor(Y_gen.view(nt*nb, -1), betas.view(nt*nb, -1))
        Yb_gen = Yb_gen.view(nt, nb, -1)
        return Y_gen, Yb_gen, z


class GAMMAPrimitiveVAETrainOP(TrainOP):

    def build_model(self):
        self.model = GAMMAPrimitiveVAE(self.modelconfig)
        self.model.train()
        self.model.to(self.device)
        self.coord_extractor = CanonicalCoordinateExtractor(self.device)
        self.max_rollout=self.trainconfig.get('max_rollout', None)
        self.noise = self.trainconfig.get('noise', None)
        self.t_his = self.modelconfig['t_his']

    def _calc_loss_rec(self, Y, Y_rec):
        '''
        - to calculate the reconstruction loss of markers, including the 1-st order derivative
        - Y and Y_rec should be in the format of [time, batch, dim]
        '''
        ## frame-wise reconstruction loss
        loss_rec = F.l1_loss(Y, Y_rec)
        ## temporal smoothness loss
        loss_td = F.l1_loss(Y_rec[1:]-Y_rec[:-1], Y[1:]-Y[:-1])
        ## sum up
        loss_rec = self.lossconfig['weight_rec'] * loss_rec + self.lossconfig['weight_td'] * loss_td
        return loss_rec

    def calc_loss(self, data, epoch):
        t_his = self.t_his
        X = data[:t_his]
        Y = data[t_his:,:,:self.model.in_dim]
        # forward pass of the model
        [Y_rec, mu, logvar] = self.model(x=X, y=Y)
        # loss computation
        ## rec loss
        loss_rec = self._calc_loss_rec(Y, Y_rec)
        ## kl-divergence
        loss_kld = 0.5 * torch.mean(-1 - logvar + mu.pow(2) + logvar.exp())
        if self.lossconfig['robust_kld']:
            loss_kld = torch.sqrt(1 + loss_kld**2)-1
        # kl loss annealing
        weight_kld = self.lossconfig['weight_kld']
        if self.lossconfig['annealing_kld']:
           weight_kld = min( ( float(epoch) / (0.9*self.trainconfig['num_epochs']) ), 1.0) * self.lossconfig['weight_kld']
        # add them together
        loss = loss_rec + weight_kld * loss_kld
        return loss, np.array([loss.item(), loss_rec.item(), loss_kld.item()])


    def calc_loss_rollout(self, data, epoch):
        ref_markers, ref_jts = data
        n_t, n_b = ref_markers.shape[:2]
        ref_jts = ref_jts.contiguous().view(n_t, n_b, -1, 3)
        t = 0
        loss = []
        loss_info = []
        while t < n_t:
            '''set time range'''
            t_lb = t
            t_ub = t+10
            if t_ub >= n_t:
                break
            t_his = self.t_his
            t_pred = 10-t_his

            '''make prediction and calculate loss'''
            ref_markers_ = ref_markers[t_lb:t_ub]
            ref_jts_ = ref_jts[t_lb:t_ub]
            if t == 0: # only use gt marker at the first primitive
                X = ref_markers_[:t_his].detach()
                Y = ref_markers_[t_his:,:,:self.model.in_dim].detach()
                R_prev, T_prev = self.coord_extractor.get_new_coordinate(ref_jts_[0])
            else:
                R_curr, T_curr = self.coord_extractor.get_new_coordinate(ref_jts_[0])
                '''get transformed ground truth marker traj Y'''
                Yg = ref_markers_[t_his:,:,:self.model.in_dim].view(t_pred,n_b, -1, 3)
                Y = torch.einsum('bij,tbpj->tbpi', R_curr.permute(0,2,1), Yg-T_curr.unsqueeze(0) )
                '''get transformed motion seed X'''
                X_prev = Y_rec[-t_his:].view(t_his,n_b, -1, 3)
                Xg = torch.einsum('bij,tbpj->tbpi', R_prev, X_prev)+T_prev.unsqueeze(0) #transform to global coord
                X = torch.einsum('bij,tbpj->tbpi', R_curr.permute(0,2,1), Xg-T_curr.unsqueeze(0)) #transform to current coord
                '''get the target-driven features'''
                if self.model.body_repr == 'ssm2_67_condi_marker2tarloc':
                    gvec = (Y[-1:] - X)/torch.norm(Y[-1:]-X, dim=-1, keepdim=True)
                    X = torch.cat([X, gvec],dim=-1)
                Y = Y.contiguous().view(t_pred,n_b, -1).detach()
                X = X.contiguous().view(t_his, n_b, -1).detach()
                R_prev = R_curr
                T_prev = T_curr

            # forward pass of the model
            [Y_rec, mu, logvar] = self.model(x=X, y=Y)

            ## rec loss
            loss_rec = self._calc_loss_rec(Y, Y_rec)
            ## kl-divergence
            loss_kld = 0.5 * torch.mean(-1 - logvar + mu.pow(2) + logvar.exp())
            if self.lossconfig['robust_kld']:
                loss_kld = torch.sqrt(1 + loss_kld**2)-1
            # kl loss annealing
            weight_kld = self.lossconfig['weight_kld']
            if self.lossconfig['annealing_kld']:
                weight_kld = min( ( float(epoch) / (0.9*self.trainconfig['num_epochs']) ), 1.0) * self.lossconfig['weight_kld']

            loss_ = loss_rec + weight_kld * loss_kld
            loss_info_ = np.array([loss_.item(), loss_rec.item(), loss_kld.item()])

            loss.append(loss_)
            loss_info.append(loss_info_)
            ####
            t += t_pred
            if len(loss) >= self.max_rollout:
                break

        loss = torch.stack(loss).mean()
        loss_info = np.mean(np.stack(loss_info), axis=0)

        return loss, loss_info



    def train(self, batch_gen):
        self.build_model()

        starting_epoch = 0
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.trainconfig['learning_rate'])
        scheduler = get_scheduler(optimizer, policy='lambda',
                                    num_epochs_fix=self.trainconfig['num_epochs_fix'],
                                    num_epochs=self.trainconfig['num_epochs'])

        if self.trainconfig['resume_training']: # can also be used for fine-tuning
            ckp_list = sorted(glob.glob(os.path.join(self.trainconfig['save_dir'],
                                        'epoch-*.ckp')),
                                        key=os.path.getmtime)
            if len(ckp_list)>0:
                checkpoint = torch.load(ckp_list[-1])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if not self.trainconfig.get('fine_tune', False):
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    starting_epoch = checkpoint['epoch']
                    print('[INFO] --resume training from {}'.format(ckp_list[-1]))
                else:
                    print('[INFO] --fine-tune from {}'.format(ckp_list[-1]))
            else:
                raise FileExistsError('the pre-trained checkpoint does not exist.')

        # training main loop
        loss_names = ['ALL', 'REC', 'KLD']
        for epoch in range(starting_epoch, self.trainconfig['num_epochs']):
            epoch_losses = 0
            epoch_nsamples = 0
            stime = time.time()
            ## training subloop for each epoch
            while batch_gen.has_next_rec():
                noise = self.trainconfig['noise'] if 'noise' in self.trainconfig else None
                if self.max_rollout is None:
                    data = batch_gen.next_batch(self.trainconfig['batch_size'], noise=noise).to(self.device)
                    if data is None:
                        continue
                    optimizer.zero_grad()
                    loss, losses_items = self.calc_loss(data, epoch)
                    loss.backward(retain_graph=False)
                    optimizer.step()
                else:
                    data = batch_gen.next_batch_with_jts(self.trainconfig['batch_size'], noise=noise)
                    if data is None:
                        continue
                    optimizer.zero_grad()
                    loss, losses_items = self.calc_loss_rollout(data, epoch)
                    loss.backward(retain_graph=False)
                    optimizer.step()
                epoch_losses += losses_items
                epoch_nsamples += 1
            if self.max_rollout is None:
                batch_gen.reset()
            else:
                batch_gen.reset_with_jts()
            scheduler.step()
            ## logging
            epoch_losses /= epoch_nsamples
            eps_time = time.time()-stime
            lr = optimizer.param_groups[0]['lr']
            info_str = '[epoch {:d}]:'.format(epoch+1)
            for name, val in zip(loss_names, epoch_losses):
                self.writer.add_scalar(name, val, epoch+1)
                info_str += '{}={:f}, '.format(name, val)
            info_str += 'time={:f}, lr={:f}'.format(eps_time, lr)

            self.logger.info(info_str)

            if ((1+epoch) % self.trainconfig['saving_per_X_ep']==0) :
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, self.trainconfig['save_dir'] + "/epoch-" + str(epoch + 1) + ".ckp")

            if self.trainconfig['verbose']:
                print(info_str)


        if self.trainconfig['verbose']:
            print('[INFO]: Training completes!')
            print()



class GAMMARegressorTrainOP(TrainOP):
    def build_model(self):
        self.model = MoshRegressor(self.modelconfig)
        self.model.train()
        self.model.to(self.device)
        self.use_cont = self.modelconfig.get('use_cont', False)
        '''get body moel'''
        bm_path = get_body_model_path()
        self.bm = get_body_model(bm_path,
                                model_type='smplx', gender=self.modelconfig['gender'],
                                batch_size=self.trainconfig['batch_size']*self.modelconfig['seq_len'],
                                device=self.device)
        '''get markers'''
        marker_path = get_body_marker_path()
        if self.modelconfig['body_repr'] == 'ssm2_67':
            with open(os.path.join(marker_path,'SSM2.json')) as f:
                markerdict = json.load(f)['markersets'][0]['indices']
            self.model.markers = self.markers = list(markerdict.values())
        else:
            raise ValueError('other marker placement is not considered yet.')



    def calc_loss(self, x_ref, xb, betas):
        '''
        - xb has axis-angle rotations
        '''
        body_param = {}
        body_param['transl'] = xb[:,:3]
        body_param['global_orient'] = xb[:,3:6]
        body_param['body_pose'] = xb[:,6:69]
        body_param['left_hand_pose'] = xb[:,69:81]
        body_param['right_hand_pose'] = xb[:,81:]
        body_param['betas'] = betas

        x_pred = self.bm(return_verts=True, **body_param).vertices[:,self.model.markers,:]
        loss_marker = F.l1_loss(x_ref, x_pred)
        loss_hpose = torch.mean( (xb[:,69:])**2 )
        loss = loss_marker + self.lossconfig['weight_reg_hpose']*loss_hpose
        return loss, np.array([loss_marker.item(), loss_hpose.item()])


    def train(self, batch_gen):
        self.build_model()
        batch_size = self.trainconfig['batch_size']
        gender = self.modelconfig['gender']

        starting_epoch = 0
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.trainconfig['learning_rate'])
        scheduler = get_scheduler(optimizer, policy='lambda',
                                    num_epochs_fix=self.trainconfig['num_epochs_fix'],
                                    num_epochs=self.trainconfig['num_epochs'])

        if self.trainconfig['resume_training']:
            ckp_list = sorted(glob.glob(os.path.join(self.trainconfig['save_dir'],'epoch-*.ckp')),
                                key=os.path.getmtime)
            if len(ckp_list)>0:
                checkpoint = torch.load(ckp_list[-1])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                starting_epoch = checkpoint['epoch']
                print('[INFO] --resuming training from {}'.format(ckp_list[-1]))

        # training main loop
        loss_names = ['MSE_MARKER', 'MSE_HPOSE']
        for epoch in range(starting_epoch, self.trainconfig['num_epochs']):
            epoch_losses = 0
            epoch_nsamples = 0
            stime = time.time()
            ## training subloop for each epoch
            while batch_gen.has_next_rec():
                data = batch_gen.next_batch_genderselection(batch_size,
                                                            gender)
                if data is None:
                    continue
                batch_betas, marker_ref = data[:2]
                marker_ref = marker_ref.contiguous().view([-1, self.model.in_dim])
                marker_ref = marker_ref.view(marker_ref.shape[0], -1, 3)
                batch_betas = batch_betas.contiguous().view([-1, 10])
                # forward pass
                xb_new = self.model(marker_ref.detach(), batch_betas)

                optimizer.zero_grad()
                loss, losses_items = self.calc_loss(marker_ref, xb_new, batch_betas)
                loss.backward(retain_graph=False)
                optimizer.step()
                epoch_losses += losses_items
                epoch_nsamples += 1
            batch_gen.reset()
            scheduler.step()
            ## logging
            epoch_losses /= epoch_nsamples
            eps_time = time.time()-stime
            lr = optimizer.param_groups[0]['lr']
            info_str = '[epoch {:d}]:'.format(epoch+1)
            for name, val in zip(loss_names, epoch_losses):
                self.writer.add_scalar(name, val, epoch+1)
                info_str += '{}={:f}, '.format(name, val)
            info_str += 'time={:f}, lr={:f}'.format(eps_time, lr)

            self.logger.info(info_str)

            if ((1+epoch) % self.trainconfig['saving_per_X_ep']==0) :
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, self.trainconfig['save_dir'] + "/epoch-" + str(epoch + 1) + ".ckp")

            if self.trainconfig['verbose']:
                print(info_str)

        if self.trainconfig['verbose']:
            print('[INFO]: Training completes!')
            print()



class GAMMAPrimitiveComboTrainOP(TrainOP):
    def __init__(self, predictorcfg, regressorcfg, lossconfig, trainconfig):
        self.dtype = torch.float32
        gpu_index = trainconfig.get('gpu_index',0)
        self.device = torch.device('cuda',
                index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.predictorcfg = predictorcfg
        self.regressorcfg = regressorcfg
        self.lossconfig = lossconfig
        self.trainconfig = trainconfig
        self.logger = get_logger(self.trainconfig['log_dir'])
        self.writer = SummaryWriter(log_dir=self.trainconfig['log_dir'])

        self.max_rollout = trainconfig.get('max_rollout', 1e8)
        self.t_his = self.predictorcfg.modelconfig['t_his']
        self.t_pred = self.predictorcfg.modelconfig['t_pred']
        self.use_cont = self.regressorcfg.modelconfig.get('use_cont', False)
        self.use_scheduled_sampling = self.trainconfig.get('scheduled_sampling', False)
        self.n_meshes = self.trainconfig['batch_size']*self.t_pred
        self.use_gt_transform = self.trainconfig.get('use_gt_transform', False)
        self.ft_regressor = self.trainconfig.get('ft_regressor', False)
        self.use_Yproj_as_seed = self.trainconfig.get('use_Yproj_as_seed', False)

    def build_model(self, load_pretrained=False):
        '''motion model'''
        self.model = GAMMAPrimitiveCombo(self.predictorcfg.modelconfig,
                                        self.regressorcfg.modelconfig)
        self.model.predictor.train()
        self.model.to(self.device)

        '''body model'''
        self.bm = get_body_model(self.trainconfig['body_model_path'],
                                type='smplx', gender=self.regressorcfg.modelconfig['gender'],
                                batch_size=self.n_meshes,
                                device=self.device).eval()

        '''marker setting'''
        marker_path = get_body_marker_path()
        with open(os.path.join(marker_path,'SSM2.json')) as f:
            markerdict = json.load(f)['markersets'][0]['indices']
        self.markers = list(markerdict.values())

        '''other utilities'''
        self.coord_extractor = CanonicalCoordinateExtractor(self.device)

        '''load pre-trained checkpoints'''
        if load_pretrained:
            self.trainconfig['resume_training']=False
            ## for marker
            ckpt_path = os.path.join(self.predictorcfg.trainconfig['save_dir'],'epoch-300.ckp')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.predictor.load_state_dict(checkpoint['model_state_dict'])
            print('[INFO] --load pre-trained predictor: {}'.format(ckpt_path))
            ## for regressor
            ckpt_path = os.path.join(self.regressorcfg.trainconfig['save_dir'],'epoch-100.ckp')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.regressor.load_state_dict(checkpoint['model_state_dict'])
            print('[INFO] --load pre-trained regressor: {}'.format(ckpt_path))


    def _calc_loss_rec(self, Y, Y_rec):
        '''
        - to calculate the reconstruction loss of markers, including the 1-st order derivative
        - Y and Y_rec should be in the format of [time, batch, dim]
        '''
        ## frame-wise reconstruction loss
        loss_rec = F.l1_loss(Y, Y_rec)
        ## temporal smoothness loss
        loss_td = F.l1_loss(Y_rec[1:]-Y_rec[:-1], Y[1:]-Y[:-1])
        ## sum up
        loss_rec = self.lossconfig['weight_rec'] * loss_rec + self.lossconfig['weight_td'] * loss_td
        return loss_rec


    def calc_loss_regressor(self, x_ref, x_ref_pred, xb, betas):
        x_pred, _, hpose = self._smplx_parser(xb, betas)
        ## marker reconstruction/cycle loss
        loss_marker_rec = self._calc_loss_rec(x_ref, x_pred)
        ## hand regularization loss
        loss_hpose = torch.mean( hpose**2 )
        loss = loss_marker_rec + self.lossconfig['weight_reg_hpose']*loss_hpose
        return loss, np.array([loss_marker_rec.item(), loss_hpose.item()])



    def calc_loss_marker(self, Y, Y_rec, mu, logvar, epoch):
        ## reconstruction loss
        loss_rec = self._calc_loss_rec(Y, Y_rec)
        ## kl-divergence
        loss_kld = 0.5 * torch.mean(-1 - logvar + mu.pow(2) + logvar.exp())
        if self.lossconfig['robust_kld']:
            loss_kld = torch.sqrt(1 + loss_kld**2)-1
        # kl loss annealing
        weight_kld = self.lossconfig['weight_kld']
        if self.lossconfig['annealing_kld']:
           weight_kld = min( ( float(epoch) / (0.9*self.trainconfig['num_epochs']) ), 1.0) * self.lossconfig['weight_kld']
        ## add them together
        if self.use_scheduled_sampling:
            loss = loss_rec + weight_kld * loss_kld
        else:
            loss = weight_kld * loss_kld
        return loss, np.array([loss_rec.item(), loss_kld.item()])


    def calc_loss_one(self, data, epoch):
        ## parse variables from data, assuming we use 'ssm2_67_and_smpl_params'
        betas, ref_markers= data[:2]
        X = ref_markers[:self.t_his]
        Y = ref_markers[self.t_his:, :, :67*3]
        betas_Y = betas[self.t_his:]

        # forward pass of the model
        [Y_rec, mu, logvar, Yb_rec] = self.model(X, Y, betas_Y)

        # calculate marker reconstruction loss
        loss_marker, loss_marker_info = self.calc_loss_marker(Y, Y_rec, mu, logvar, epoch)
        # calculate the body param regression loss
        loss_bparams, loss_bparams_info = self.calc_loss_regressor(Y, Y_rec, Yb_rec, betas_Y)
        # merge them
        loss = loss_marker + loss_bparams
        loss_info = np.concatenate([loss_marker_info, loss_bparams_info])

        return loss, loss_info

    def _smplx_parser(self, Yb, betas):
        xb_ = Yb.view(self.n_meshes, -1)
        betas_ = betas.view(self.n_meshes, -1)
        body_param = {}
        body_param['transl'] = xb_[:,:3]
        body_param['global_orient'] = xb_[:,3:6]
        body_param['body_pose'] = xb_[:,6:69]
        body_param['left_hand_pose'] = xb_[:,69:81]
        body_param['right_hand_pose'] = xb_[:,81:]
        body_param['betas'] = betas_
        hpose = torch.cat([body_param['left_hand_pose'],
                            body_param['right_hand_pose']], dim=-1)
        marker_pred = self.bm(return_verts=True, **body_param).vertices[:,self.markers,:]
        marker_pred = marker_pred.view(self.n_meshes, -1).view(self.t_pred,
                                                self.trainconfig['batch_size'], -1)
        jts_pred = self.bm(return_verts=True, **body_param).joints[:,:22] #[n_meshes, J, 3]
        jts_pred = jts_pred.view(self.t_pred, self.trainconfig['batch_size'], -1, 3)

        return marker_pred, jts_pred, hpose


    def calc_loss_rollout_UseGTTransform(self, data, epoch):
        '''
        - parse variables from data, assuming we use 'ssm2_67_and_smpl_params
        - this script is used for GAMMACombo-v4
        '''
        betas, ref_markers, _,_,_, ref_jts = data
        betas = betas.to(self.device)
        ref_markers = ref_markers.to(self.device)
        n_t, n_b = betas.shape[:2]
        ref_jts = ref_jts.view(n_t, n_b, -1, 3).to(self.device)
        t = 0
        loss = []
        loss_info = []
        while t < n_t:
            '''set time range'''
            t_lb = t
            t_ub = t+10
            if t_ub >= n_t:
                break

            '''make prediction and calculate loss'''
            betas_ = betas[t_lb:t_ub]
            ref_markers_ = ref_markers[t_lb:t_ub]
            ref_jts_ = ref_jts[t_lb:t_ub]
            betas_Y = betas_[self.t_his:]
            if t == 0: # only use gt marker at the first primitive
                X = ref_markers_[:self.t_his].detach()
                Y = ref_markers_[self.t_his:].detach()
                # R_prev, T_prev = self.get_new_coordinate(ref_jts_, ref='start')
                R_prev, T_prev = self.coord_extractor.get_new_coordinate(ref_jts_)

            else: # motion seed = average of predicted and reprojected marker in the new transformed coordinate
                '''get transformed motion seed X'''
                # R_curr, T_curr = self.get_new_coordinate(ref_jts_, ref='start')
                R_curr, T_curr = self.coord_extractor.get_new_coordinate(ref_jts_)
                if self.use_Yproj_as_seed:
                    Y_proj, _, _ = self._smplx_parser(Yb_rec, betas_Y)
                    Y_blend = Y_proj
                else:
                    Y_blend = Y_rec # only use the predicted markers
                X_prev = Y_blend.view(self.t_pred,self.trainconfig['batch_size'], -1, 3)[-self.t_his:]
                Xg = torch.einsum('bij,tbpj->tbpi', R_prev, X_prev)+T_prev.unsqueeze(0) #transform to global coord
                X = torch.einsum('bij,tbpj->tbpi', R_curr.permute(0,2,1), Xg-T_curr.unsqueeze(0)) #transform to current coord
                X = X.contiguous().view(self.t_his, self.trainconfig['batch_size'], -1).detach()
                '''get transformed ground truth marker traj Y'''
                Yg = ref_markers_[self.t_his:].view(self.t_pred,self.trainconfig['batch_size'], -1, 3)
                Y = torch.einsum('bij,tbpj->tbpi', R_curr.permute(0,2,1), Yg-T_curr.unsqueeze(0) )
                Y = Y.contiguous().view(self.t_pred,self.trainconfig['batch_size'], -1).detach()
                R_prev = R_curr
                T_prev = T_curr

            # forward pass of the model
            [Y_rec, mu, logvar, Yb_rec] = self.model(X, Y, betas_Y)

            # calculate marker reconstruction loss
            loss_marker, loss_marker_info = self.calc_loss_marker(Y, Y_rec, mu, logvar, epoch)
            # calculate the body param regression loss
            loss_bparams, loss_bparams_info = self.calc_loss_regressor(Y, Y_rec, Yb_rec, betas_Y)
            # merge them
            loss_ = loss_marker + loss_bparams
            loss_info_ = np.concatenate([loss_marker_info, loss_bparams_info])

            loss.append(loss_)
            loss_info.append(loss_info_)
            ####
            t += self.t_pred
            if len(loss) >= self.max_rollout:
                break

        loss = torch.stack(loss).mean()
        loss_info = np.mean(np.stack(loss_info), axis=0)

        return loss, loss_info



    def calc_loss_rollout(self, data, epoch):
        '''
        - parse variables from data, assuming we use 'ssm2_67_and_smpl_params
        - this script is used for GAMMACombo-v5 or more. The local coordinate is located at the generated body, rather than the ground truth (as v4)
        '''
        betas, ref_markers, _,_,_, ref_jts = data
        betas = betas.to(self.device)
        ref_markers = ref_markers.to(self.device)
        n_t, n_b = betas.shape[:2]
        ref_jts = ref_jts.view(n_t, n_b, -1, 3).to(self.device)
        t = 0
        loss = []
        loss_info = []
        while t < n_t:
            '''set time range'''
            t_lb = t
            t_ub = t+10
            if t_ub >= n_t:
                break

            '''make prediction and calculate loss'''
            betas_ = betas[t_lb:t_ub]
            ref_markers_ = ref_markers[t_lb:t_ub]
            ref_jts_ = ref_jts[t_lb:t_ub]
            betas_Y = betas_[self.t_his:]
            with torch.no_grad():
                if t == 0: # only use gt marker at the first primitive
                    X = ref_markers_[:self.t_his].detach()
                    Y = ref_markers_[self.t_his:].detach()
                    R_prev, T_prev = self.coord_extractor.get_new_coordinate(ref_jts_)
                else:
                    '''get transformed motion seed X'''
                    _, pred_jts_, _ = self._smplx_parser(Yb_rec, betas_Y)
                    pred_jts_ = pred_jts_.view(self.t_pred,self.trainconfig['batch_size'], -1, 3)
                    R_, T_ = self.coord_extractor.get_new_coordinate(pred_jts_[-self.t_his:])
                    R_curr = torch.einsum('bij,bjk->bik', R_prev, R_)
                    T_curr = torch.einsum('bij,bpj->bpi', R_prev, T_)+T_prev
                    if self.use_Yproj_as_seed:
                        Y_proj, _, _ = self._smplx_parser(Yb_rec, betas_Y)
                        Y_blend = Y_proj
                    else:
                        Y_blend = Y_rec # only use the predicted markers
                    X_prev = Y_blend.view(self.t_pred,self.trainconfig['batch_size'], -1, 3)[-self.t_his:]
                    X = torch.einsum('bij,tbpj->tbpi', R_.permute(0,2,1), X_prev-T_.unsqueeze(0)) #transform to current coord
                    X = X.contiguous().view(self.t_his, self.trainconfig['batch_size'], -1).detach()
                    '''get transformed ground truth marker traj Y'''
                    Yg = ref_markers_[self.t_his:].view(self.t_pred,self.trainconfig['batch_size'], -1, 3)
                    Y = torch.einsum('bij,tbpj->tbpi', R_curr.permute(0,2,1), Yg-T_curr.unsqueeze(0) )
                    Y = Y.contiguous().view(self.t_pred,self.trainconfig['batch_size'], -1).detach()
                    R_prev = R_curr
                    T_prev = T_curr

            # forward pass of the model
            [Y_rec, mu, logvar, Yb_rec] = self.model(X, Y, betas_Y)

            # calculate marker reconstruction loss
            loss_marker, loss_marker_info = self.calc_loss_marker(Y, Y_rec, mu, logvar, epoch)
            # calculate the body param regression loss
            if not self.use_Yproj_as_seed:
                with torch.no_grad():
                    _, loss_bparams_info = self.calc_loss_regressor(Y, Y_rec, Yb_rec, betas_Y)
            # merge them
                loss_ = loss_marker
            else:
                loss_bparams, loss_bparams_info = self.calc_loss_regressor(Y, Y_rec, Yb_rec, betas_Y)
                # merge them
                loss_ = loss_marker + loss_bparams
            loss_info_ = np.concatenate([loss_marker_info, loss_bparams_info])

            loss.append(loss_)
            loss_info.append(loss_info_)
            ####
            t += self.t_pred
            if len(loss) >= self.max_rollout:
                break

        loss = torch.stack(loss).mean()
        loss_info = np.mean(np.stack(loss_info), axis=0)

        return loss, loss_info




    def train(self, batch_gen):
        if self.trainconfig['resume_training']:
            self.build_model(load_pretrained=False)
        else:
            self.build_model(load_pretrained=True)

        starting_epoch = 0
        optimizer = optim.Adam(self.model.predictor.parameters(),
                               lr=self.trainconfig['learning_rate'])
        scheduler = get_scheduler(optimizer, policy='lambda',
                                    num_epochs_fix=self.trainconfig['num_epochs_fix'],
                                    num_epochs=self.trainconfig['num_epochs'])

        if self.trainconfig['resume_training']:
            ckp_list = sorted(glob.glob(os.path.join(self.trainconfig['save_dir'],'epoch-*.ckp')),
                                key=os.path.getmtime)
            if len(ckp_list)>0:
                checkpoint = torch.load(ckp_list[-1])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                starting_epoch = checkpoint['epoch']
                print('[INFO] --resuming training from {}'.format(ckp_list[-1]))

        # training main loop
        loss_names = ['REC', 'KLD', 'REG', 'HPOSE']
        for epoch in range(starting_epoch, self.trainconfig['num_epochs']):
            epoch_losses = 0
            epoch_nsamples = 0
            stime = time.time()
            ## training subloop for each epoch
            while batch_gen.has_next_rec():
                data = batch_gen.next_batch_genderselection(batch_size=self.trainconfig['batch_size'],
                                            gender=self.regressorcfg.modelconfig['gender'],
                                            batch_first=False) # returns [t,b,d]
                if data is None:
                    continue
                if self.use_scheduled_sampling:
                    if self.use_gt_transform:
                        loss, losses_items = self.calc_loss_rollout_UseGTTransform(data, epoch)
                    else:
                        loss, losses_items = self.calc_loss_rollout(data, epoch)
                else:
                    loss, losses_items = self.calc_loss_one(data, epoch)
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                epoch_losses += losses_items
                epoch_nsamples += 1
            batch_gen.reset()
            scheduler.step()
            ## logging
            epoch_losses /= epoch_nsamples
            eps_time = time.time()-stime
            lr = optimizer.param_groups[0]['lr']
            info_str = '[epoch {:d}]:'.format(epoch+1)
            for name, val in zip(loss_names, epoch_losses):
                self.writer.add_scalar(name, val, epoch+1)
                info_str += '{}={:f}, '.format(name, val)
            info_str += 'time={:f}, lr={:f}'.format(eps_time, lr)

            self.logger.info(info_str)


            if ((1+epoch) % self.trainconfig['saving_per_X_ep']==0) :
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, self.trainconfig['save_dir'] + "/epoch-" + str(epoch + 1) + ".ckp")

            if self.trainconfig['verbose']:
                print(info_str)

        if self.trainconfig['verbose']:
            print('[INFO]: Training completes!')
            print()



from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
class GAMMAPrimitiveComboGenOP(TestOP):
    """the interface to GAMMA when using it to produce motions

    """
    def __init__(self, predictorcfg, regressorcfg, testconfig):
        self.dtype = torch.float32
        gpu_index = testconfig.get('gpu_index',0)
        self.device = torch.device('cuda',
                index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.predictorcfg = predictorcfg
        self.regressorcfg = regressorcfg
        self.testconfig = testconfig
        self.use_cont = self.regressorcfg.modelconfig.get('use_cont', False)
        self.t_his = self.predictorcfg.modelconfig['t_his']
        self.var_seed_len = self.predictorcfg.modelconfig.get('var_seed_len', False)


    def build_model(self, load_pretrained_model=False):
        self.model = GAMMAPrimitiveCombo(self.predictorcfg.modelconfig, self.regressorcfg.modelconfig)
        self.model.predictor.eval()
        self.model.regressor.eval()
        self.model.to(self.device)

        '''load pre-trained checkpoints'''
        if load_pretrained_model:
            ## for marker
            try:
                ckpt_path = os.path.join(self.predictorcfg.trainconfig['save_dir'],'epoch-400.ckp')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.model.predictor.load_state_dict(checkpoint['model_state_dict'])
                print('[INFO] --load pre-trained predictor: {}'.format(ckpt_path))
            except:
                ckpt_path = os.path.join(self.predictorcfg.trainconfig['save_dir'],'epoch-200.ckp')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.model.predictor.load_state_dict(checkpoint['model_state_dict'])
                print('[INFO] --load pre-trained predictor: {}'.format(ckpt_path))
            ## for regressor
            ckpt_path = os.path.join(self.regressorcfg.trainconfig['save_dir'],'epoch-100.ckp')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.regressor.load_state_dict(checkpoint['model_state_dict'])
            print('[INFO] --load pre-trained regressor: {}'.format(ckpt_path))
        else:
            try:
                ckpt_path = os.path.join(self.testconfig['ckpt_dir'],'epoch-120.ckp')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
            except:
                ckpt_path = os.path.join(self.testconfig['ckpt_dir'],'epoch-10.ckp')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print('[INFO] --load combo model: {}'.format(ckpt_path))

    def _blend_params(self, body_params, t_his=None):
        # Yb = torch.cat((Xb, Yb_pred), dim=0)
        if t_his is None:
            t_his = self.t_his
        start_idx = 6
        param_n = body_params[t_his-1, :, start_idx:]
        param_p = body_params[t_his+1, :, start_idx:]
        body_params[t_his, :, start_idx:] = (param_n+param_p)/2.0

        t_his = t_his+1
        param_n = body_params[t_his-1, :, start_idx:]
        param_p = body_params[t_his+1, :, start_idx:]
        body_params[t_his, :, start_idx:] = (param_n+param_p)/2.0
        return body_params


    def generate(self,
                data: Union[torch.Tensor, np.ndarray],
                bparams: Union[torch.Tensor, np.ndarray],
                betas: Union[torch.Tensor, np.ndarray],
                n_gens: int=-1,
                to_numpy: bool=True,
                return_z: bool=False,
                param_blending: bool=True,
                t_his = None,
                z: Union[None, torch.Tensor] = None,
        ):
        """generate motion primitives based on a motion seed X
        torch.Tensor and np.ndarray are supported

        Args:
            - data: the motion seed, with the shape [t,b,d]
            - bparams: the body parameters of the motion seed, with the shape [t,b,d]
            - betas: body shape, (10,) or (1,10)
            - n_gens: how many motions are generated. If -1, it has the same batch size with the input.
            - to_numpy: produce numpy if true
            - return_z: return the latnet variables if set to True
            - param_blending: smooth the body poses, to eliminate the first-frame jump and compensate the regressor inaccuracy
            - t_his: frames in the motion seed. If None, then t_his is randomly determined.

        Returns:
            a list containing following:
            - Y_gen: the predicted markers
            - Yb_gen: the predicted-and-regressed body parameters. Rotations in axis-angle
            - (optional only if return_z=True) z: the returned random variable
        Raises:
            None
        """

        if type(data) is np.ndarray:
            traj = torch.tensor(data, device=self.device,
                    dtype=self.dtype).contiguous()
        elif type(data) is torch.Tensor:
            traj = data.detach()

        if type(bparams) is np.ndarray:
            bparams = torch.tensor(bparams, device=self.device,
                    dtype=self.dtype).contiguous()
        elif type(bparams) is torch.Tensor:
            bparams = bparams.detach()

        t_his_ = t_his
        t_pred_ = 10-t_his_

        X = traj[:t_his_]
        Xb = bparams[:t_his_]
        if n_gens > 0:
            X = X.repeat((1, n_gens, 1))
            Xb = Xb.repeat((1, n_gens, 1))

        if type(betas) is np.ndarray: #betas=(10,) or (1,10)
            betas = torch.cuda.FloatTensor(np.tile(betas, (t_pred_, X.shape[1], 1)),device=self.device)
        elif type(betas) is torch.Tensor: #betas=(1,10) or (10,)
            betas = betas.repeat(t_pred_, X.shape[1], 1)

        if not return_z:
            Y_pred, Yb_pred = self.model.sample_prior(X, betas=betas, z=z)
        else:
            Y_pred, Yb_pred, z = self.model.sample_prior_return_z(X, betas=betas)

        #concatenate motion seed and prediction
        Y = torch.cat((X[:,:,:67*3], Y_pred), dim=0)
        Yb = torch.cat((Xb, Yb_pred), dim=0)
        if param_blending:
            Yb = self._blend_params(Yb)
        # from [t,b,d] to [b,t,d]
        Y = Y.permute(1,0,2)
        Yb = Yb.permute(1,0,2)

        if not return_z:
            if to_numpy:
                Y = Y.detach().cpu().numpy()
                Yb = Yb.detach().cpu().numpy()
            return Y, Yb
        else:
            if to_numpy:
                Y = Y.detach().cpu().numpy()
                Yb = Yb.detach().cpu().numpy()
                z = z.detach().cpu().numpy()
            return Y, Yb, z



    def generate_ppo(self,
                    policy_model,
                    states: Union[torch.Tensor, np.ndarray],
                    bparams: Union[torch.Tensor, np.ndarray],
                    betas: Union[torch.Tensor, np.ndarray],
                    n_gens: int=-1,
                    obj_points=None,
                    target_ori=None,
                    local_map=None,
                    to_numpy: bool=True,
                    param_blending: bool=True,
                    use_policy_mean: bool=False,
                    use_policy: bool=True):
        """generate motion primitives based on a motion seed X, and a policy network
        torch.Tensor and np.ndarray are supported

        Args:
            - states: the cat of the motion seed and the goal-based features, with shape [t,b,d]
            - bparams: the body parameters of the motion seed, with the shape [t,b,d]
            - betas: body shape, (10,) or (1,10)
            - n_gens: how many motions are generated. If -1, it has the same batch size with the input.
            - to_numpy: produce numpy if true
            - param_blending: smooth the body poses, to eliminate the first-frame jump and compensate the regressor inaccuracy
            - use_policy_mean: if true, we only use the mean of the inference posterior of policy, rather than random sampling
            - use_policy: if false, it will sample latent variables from N(0,I)

        Returns:
            a list containing following:
            - Y: the history and predicted markers
            - Yb: the history and predicted body parameters
            - z: the latent variable
            - z_log_prob: the log probability of the latent variable z
            - value: the values from the critic network
        Raises:
            None
        """
        if type(states) is np.ndarray:
            traj = torch.tensor(states, device=self.device,
                    dtype=self.dtype).contiguous()
        elif type(states) is torch.Tensor:
            traj = states.detach()

        if type(bparams) is np.ndarray:
            bparams = torch.tensor(bparams, device=self.device,
                    dtype=self.dtype).contiguous()
        elif type(bparams) is torch.Tensor:
            bparams = bparams.detach()
        t_his = traj.shape[0]
        t_pred = 10-t_his
        X = traj[:t_his] # original setting
        Xb = bparams[:t_his] # original setting
        if n_gens > 0:
            X = X.repeat((1, n_gens, 1))
            Xb = Xb.repeat((1, n_gens, 1))
            if target_ori is not None:
                target_ori = target_ori.repeat((n_gens, 1))
            if local_map is not None:
                local_map = local_map.repeat((n_gens, 1, 1))
        if type(betas) is np.ndarray: #betas=(10,)
            betas = torch.cuda.FloatTensor(np.tile(betas, (t_pred, X.shape[1], 1)),device=self.device)
        elif type(betas) is torch.Tensor: #betas=(1,10)
            betas = betas.repeat(t_pred, X.shape[1], 1)

        """policy gen"""
        if use_policy:
            z_mu, z_logvar, value = policy_model(X, obj_points=obj_points, target_ori=target_ori, local_map=local_map)
            value = value[:, 0]
            z_var = torch.exp(z_logvar.clamp(policy_model.min_logvar, policy_model.max_logvar))
            act_distrib_c = Normal(z_mu, z_var ** 0.5)  # batched distribution
            act_distrib = Independent(act_distrib_c, 1)
            z = act_distrib.sample()  # size=(b,128)
            z_log_prob = act_distrib.log_prob(z)  # size=(b=1, )
            if use_policy_mean:
                z_in = z_mu
            else:
                z_in =  z
            z = z_in
        else:
            z_in = torch.randn((Xb.shape[1], self.model.predictor.z_dim), device=Xb.device)
            z = z_in
            z_log_prob = None
            value = None

        """motion gen"""
        Y_gen, Yb_gen = self.model.sample_prior(X[:,:,:67*3], betas=betas, z=z_in)
        Y = torch.cat((X[:,:,:67*3], Y_gen), dim=0)
        Yb = torch.cat((Xb, Yb_gen), dim=0)
        if param_blending:
            Yb = self._blend_params(Yb,t_his = t_his)

        if to_numpy:
            Y = Y.detach().cpu().numpy()
            Yb = Yb.detach().cpu().numpy()
            if use_policy:
                z = z.detach().cpu().numpy()
                z_log_prob = z_log_prob.detach().cpu().numpy()
                value = value.detach().cpu().numpy()

        return Y, Yb, z, z_log_prob, value


    def _compose_body_params_(self, data):
        transl = data['transl']
        glorot = data['glorot']
        body_pose = data['poses'][:,:63]
        hand_pose = np.zeros((transl.shape[0], 24))
        xb = np.concatenate([transl, glorot, body_pose, hand_pose],axis=-1) #[t,d]
        return xb


    def generate_primitive_to_files(self, batch_gen, n_seqs, n_gens, t_his=None):
        '''
        n_seqs: how many sequences to generate
        n_gens: for each input sequence, how many different sequences to predict
        '''
        # self.build_model()
        ### generate data and save them to files. They will need inverse kinematics to get body mesh.
        ### generate data
        gen_results = {}
        gen_results['gt'] = []
        gen_results['betas'] = []
        gen_results['gender'] = []
        gen_results['transf_rotmat'] = []
        gen_results['transf_transl'] = []
        gen_results['markers'] = []
        gen_results['smplx_params'] = []
        gen_results['transl'] = []

        idx = 0
        while idx < n_seqs:
            print('[INFO] generating with motion seed {}'.format(idx))
            data = batch_gen.next_sequence()
            if str(data['gender']) != self.regressorcfg.modelconfig['gender']:
                continue
            motion_np = data['body_feature']
            motion = torch.cuda.FloatTensor(motion_np).unsqueeze(1) #[t,b,d]
            bparams_np = self._compose_body_params_(data)
            bparams = torch.cuda.FloatTensor(bparams_np).unsqueeze(1) #[t,b,d]
            gen_results['gt'].append(motion_np[:,:67*3].reshape((1,motion_np.shape[0],-1,3)))
            gen_results['betas'].append(data['betas'])
            gen_results['gender'].append(str(data['gender']))
            gen_results['transf_rotmat'].append(data['transf_rotmat'])
            gen_results['transf_transl'].append(data['transf_transl'])
            gen_results['transl'].append(data['transl'][None,...]) #[b,t,d]
            # generate
            pred_markers, pred_body_params = self.generate(motion, bparams, betas=data['betas'],
                                                        n_gens=n_gens, t_his=t_his)
            pred_markers = np.reshape(pred_markers, (pred_markers.shape[0], pred_markers.shape[1],-1,3))
            gen_results['markers'].append(pred_markers)
            gen_results['smplx_params'].append(pred_body_params)
            idx+=1
        gen_results['gt'] = np.stack(gen_results['gt'])
        gen_results['markers'] = np.stack(gen_results['markers']) #[#seq, #genseq_per_pastmotion, t, #joints, 3]
        gen_results['smplx_params'] = np.stack(gen_results['smplx_params'])
        gen_results['transl'] = np.stack(gen_results['transl'])

        ### save to file
        outfilename = os.path.join(
                            self.testconfig['result_dir'],
                            'mp_gen_seed{}'.format(self.testconfig['seed']),
                            batch_gen.amass_subset_name[0]
                        )
        if not os.path.exists(outfilename):
            os.makedirs(outfilename)
        outfilename = os.path.join(outfilename,
                        'results_{}_{}.pkl'.format(self.predictorcfg.modelconfig['body_repr'],
                                                self.regressorcfg.modelconfig['gender']
                                                )
                        )
        print(outfilename)
        with open(outfilename, 'wb') as f:
            pickle.dump(gen_results, f)

















