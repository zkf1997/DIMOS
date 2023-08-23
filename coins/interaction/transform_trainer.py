import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import sys

import numpy as np

sys.path.append('..')


import smplx
import trimesh
from datetime import datetime
import pickle
import shutil
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchgeometry as tgm
import pytorch3d
from pytorch3d.structures import Pointclouds, Meshes
import pytorch3d.loss
import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
import open3d as o3d
from datetime import datetime
from copy import deepcopy
from argparse import ArgumentParser, Namespace

from configuration.joints import *
# from data.frame_dataset import FrameDataset
from interaction.dataset import PelvisFrameDataset, PelvisFrameCanonicalDataset
from interaction.viz_util import render_obj_multview
from data.scene import scenes, to_trimesh, to_open3d
from data.utils import *
from interaction.smplx_regressor import SMPLX_Regressor
from interaction.loss import *
# from interaction.transform_net import *
from interaction.interaction_model import InteractionVAE
# from interaction.interaction_loss import *
from interaction.utils import *

def rot6d_to_mat(module_input):
    reshaped_input = module_input.view(-1, 3, 2)

    b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack([b1, b2, b3], dim=-1)

class geodesic_loss_R(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(geodesic_loss_R, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(np.ones(batch)))
        cos = torch.max(cos, m1.new(np.ones(batch)) * -1)

        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            breakpoint()
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))

        else:
            return theta

def create_frame(x, origin_color=(0.8, 0.8, 0.8)):
    pelvis = x[6:].detach().cpu().numpy()
    rotmat = rot6d_to_mat(x[:6]).detach().cpu().numpy().reshape((3, 3))
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotmat
    transform[:3, 3] = pelvis
    return trimesh.creation.axis(transform=transform, origin_color=origin_color)

class LitTransformNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        if not hasattr(args, 'dim_point_feature'):
            args.dim_point_feature = 6
        self.args = args
        self.save_hyperparameters(args)
        self.start_time = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
        # self.save_hyperparameters('args')
        args.device = device
        # self.body_model = smplx.create(smplx_model_folder, model_type='smplx',
        #          gender='neutral', ext='npz',
        #          num_pca_comps=num_pca_comps, batch_size=1)

        if args.model == 'InteractionVAE':
            self.model = InteractionVAE(args)
        else:
            print('not implemented')
            return

    # x: 6d global orientation and 3d location of pelvis
    def forward(self, x, batch):
        return self.model(x, batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=self.args.l2_norm)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.9, verbose=True)
        return ({'optimizer': optimizer,
                 })

    def calc_loss(self, x, x_hat, q_z, batch=None):
        obj_pointclouds, verb_ids = batch['object_pointclouds'], batch['verb_ids']
        batch_size = x.shape[0]
        rotmat = rot6d_to_mat(x[:, :6])
        pelvis = x[:, 6:]
        rotmat_hat = rot6d_to_mat(x_hat[:, :6])
        pelvis_hat = x_hat[:, 6:]
        loss_orient = geodesic_loss_R(reduction='mean')(
            rotmat_hat,
            rotmat
        )
        loss_pelvis = F.l1_loss(pelvis_hat, pelvis)

        location = obj_pointclouds[:, :, :, :3]
        vectors = location - pelvis[:, None, None, :]
        min_dist, _ = torch.min(torch.sum(vectors ** 2, dim=-1), dim=-1)  # BxI
        vectors_hat = location - pelvis_hat[:, None, None, :]
        min_dist_hat, _ = torch.min(torch.sum(vectors_hat ** 2, dim=-1), dim=-1)
        loss_dist = F.l1_loss(min_dist_hat, min_dist)

        # loss of reconstruction of points coord in pelvis frame
        local_coords = torch.matmul(vectors, rotmat.transpose(1, 2)[:, None, :, :])
        local_coords_hat = torch.matmul(vectors_hat, rotmat_hat.transpose(1, 2)[:, None, :, :])
        loss_coord = F.l1_loss(local_coords, local_coords_hat)

        # pelvis-object penetration loss
        dist_hat = torch.sqrt(torch.sum(vectors_hat ** 2, dim=-1))  # BxIxP
        thresh = self.args.thresh_penetration
        # positive value means very close to pelvis, possible penetration
        penetration = thresh - dist_hat
        penetration_mask = (verb_ids == 3).unsqueeze(2)  # whether atomic is touch, BxIx1
        penetration = penetration * penetration_mask.float()
        penetration = penetration[penetration > 0]
        loss_penetration = penetration.mean() if len(penetration) > 0 else torch.tensor(0.0, device=penetration.device)

        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((x.shape[0], self.args.latent_dim), requires_grad=False, device=device),
            scale=torch.ones((x.shape[0], self.args.latent_dim), requires_grad=False, device=device))
        loss_kl = torch.mean(torch.mean(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))
        if self.args.robust_kl:
            loss_kl = torch.sqrt(loss_kl * loss_kl + 1) - 1.0

        loss_dict = dict(orient=loss_orient,
                         pelvis=loss_pelvis,
                         dist=loss_dist,
                         kl=loss_kl,
                         penetration=loss_penetration,
                         coord=loss_coord,
                         )

        annealing_factor = min(1.0, max(float(self.current_epoch) / (self.args.second_stage), 0)) if self.args.use_annealing else 1
        weighted_loss_dict = {
            'orient': loss_dict['orient'] * self.args.weight_orient,
            'pelvis': loss_dict['pelvis'] * self.args.weight_pelvis,
            'dist': loss_dict['dist'] * self.args.weight_dist,
            'coord': loss_dict['coord'] * self.args.weight_coord,
            'penetration': loss_dict['penetration'] * self.args.weight_penetration,
            'kl':
                max(annealing_factor ** 2, 0) *
                self.args.weight_kl * loss_dict['kl'],
        }

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict

    def _common_step(self, batch, batch_idx, mode):
        pelvis, rotmat = batch['pelvis'], batch['pelvis_orient']
        rot6d = rotmat[:, :3, :2].reshape(-1, 6)
        x = torch.cat([rot6d, pelvis], dim=1)
        x_hat, q_z = self(x, batch)
        x_hat = x_hat.squeeze(1)
        loss, loss_dict, weighted_loss_dict = self.calc_loss(x, x_hat, q_z, batch=batch)

        # render reconstructed and sampled interactions
        render_interval = 64 if mode == 'valid' else 256
        if (batch_idx % render_interval == 0) and (self.current_epoch > self.args.render_epoch or self.args.debug):
            x_sample, _ = self.model.sample(batch)
            x_sample = x_sample.squeeze(1)
            obj_points = batch['object_pointclouds']
            centroid, rotation = batch['centroid'], batch['rotation']
            B, I, P, C = obj_points.shape
            obj_points = obj_points.reshape(B, I*P, C)
            batch_size = x.shape[0]
            render_num = 4
            for idx in range(min(batch_size, render_num)):
                base_name = mode + '_E{:03d}_It{:04d}_orient_{:.4f}_pelvis_{:.5f}_id{:d}_{}.png'.format(
                                                self.current_epoch, batch_idx, loss_dict['orient'].item(),
                                                loss_dict['pelvis'].item(), idx, batch['action'][idx])

                colors = np.array([0, 0, 200, 200])
                body=None
                inverse_rotation = rotation[idx].transpose(0, 1).detach().cpu().numpy()
                inverse_transl = np.matmul(inverse_rotation, centroid[idx, :].reshape(3, 1).detach().cpu().numpy())
                inverse_transform = np.eye(4)
                inverse_transform[:3, :3] = inverse_rotation
                inverse_transform[:3, 3] = inverse_transl.squeeze()
                obj_pointcloud = trimesh.PointCloud(
                    vertices=obj_points[idx, :, :3].cpu().numpy(),
                    colors=colors,
                )
                obj_pointcloud.apply_transform(inverse_transform)
                frame_ori = create_frame(x[idx], origin_color=(1.0, 0.0, 0.0))
                frame_ori.apply_transform(inverse_transform)
                frame_rec = create_frame(x_hat[idx], origin_color=(0.0, 1.0, 0.0))
                frame_rec.apply_transform(inverse_transform)
                frame_sample = create_frame(x_sample[idx], origin_color=(0.0, 0.0, 1.0))
                frame_sample.apply_transform(inverse_transform)


                export_file = Path.joinpath(save_dir, 'render', 'contrast_' + base_name)
                export_file.parent.mkdir(exist_ok=True, parents=True)
                img_collage = render_obj_multview(obj_pointcloud, frame_rec, frame_contrast=frame_ori, body=body)
                img_collage.save(str(export_file))

                export_file = Path.joinpath(save_dir, 'render', 'sample_' + base_name)
                img_collage = render_obj_multview(obj_pointcloud, frame_sample)
                img_collage.save(str(export_file))


        return loss, loss_dict, weighted_loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'train')

        self.log('train_loss', loss, prog_bar=False)
        for key in loss_dict:
            self.log(key, loss_dict[key], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self._common_step(batch, batch_idx, 'valid')

        for key in loss_dict:
            self.log('val_' + key, loss_dict[key], prog_bar=False)
        self.log('val_loss', loss)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='InteractionVAE')
    parser.add_argument("--num_verb", type=int, default=4)
    parser.add_argument("--use_pointnet2", type=int, default=1)
    parser.add_argument("--normalize_obj", type=int, default=0)
    parser.add_argument("--num_obj_points", type=int, default=8192)
    parser.add_argument("--dim_point_feature", type=int, default=0)
    parser.add_argument("--num_obj_keypoints", type=int, default=256)
    parser.add_argument("--num_body_points", type=int, default=1)
    parser.add_argument("--dim_body_points", type=int, default=9)
    parser.add_argument("--point_level", type=int, default=3)
    parser.add_argument("--latent_dimension", type=int, default=128)
    parser.add_argument("--use_contact_feature", type=int, default=0)

    # transformer
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=512)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--interaction_bias", type=int, default=0)
    parser.add_argument("--latent_usage", type=str, default='memory')
    parser.add_argument("--template_type", type=str, default='zero')
    parser.add_argument("--return_attention", type=int, default=1)
    parser.add_argument("--mask_body", type=int, default=0)
    parser.add_argument("--mask_prob", type=float, default=0.05)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)
    parser.add_argument("--robust_kl", type=int, default=1)
    parser.add_argument("--weight_pelvis", type=float, default=3)
    parser.add_argument("--weight_orient", type=float, default=1)
    parser.add_argument("--weight_kl", type=float, default=1)
    parser.add_argument("--weight_dist", type=float, default=1)
    parser.add_argument("--weight_coord", type=float, default=1)
    parser.add_argument("--weight_penetration", type=float, default=3)
    parser.add_argument("--thresh_penetration", type=float, default=0.25)
    parser.add_argument("--use_annealing", type=int, default=0)

    parser.add_argument("--use_regressor", type=int, default=0)
    parser.add_argument("--raw_points", type=int, default=0)
    parser.add_argument("--dummy_obj", type=int, default=0)
    parser.add_argument("--use_contact", type=int, default=0)

    parser.add_argument("--learned_prior", type=int, default=0)
    parser.add_argument("--use_kronecker", type=int, default=0)
    parser.add_argument("--freeze", type=int, default=0)

    parser.add_argument("--used_interaction", type=str, default='all')
    parser.add_argument("--skip_composite", type=str, default=None)
    parser.add_argument("--used_instance", type=str, default=None)
    parser.add_argument("--scale_obj", type=int, default=0)
    parser.add_argument("--center_type", type=str, default='object')
    parser.add_argument("--rotation", type=str, default='object')
    parser.add_argument("--point_sample", type=str, default='random')
    parser.add_argument("--use_augment", type=int, default=1)
    parser.add_argument("--data_overwrite", type=int, default=0)
    parser.add_argument("--use_prox_single", type=int, default=0)
    parser.add_argument("--use_annotate", type=int, default=1)
    parser.add_argument("--include_motion", type=int, default=0)
    parser.add_argument("--use_floor_height", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--second_stage", type=int, default=20,
                        help="annealing some loss weights in early epochs before this num")
    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%H:%M:%S.%f"))
    parser.add_argument("--render_thresh", type=float, default=5e-2)
    parser.add_argument("--render_epoch", type=int, default=2333)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    # make demterministic
    pl.seed_everything(233, workers=True)
    torch.autograd.set_detect_anomaly(True)
    # args.deterministic = True

    # data
    train_dataset = PelvisFrameDataset(split='train', num_points=args.num_obj_points,
                                                normalize_obj=args.normalize_obj)
    valid_dataset = PelvisFrameDataset(split='valid', num_points=args.num_obj_points,
                                                normalize_obj=args.normalize_obj)
    # train_dataset = PelvisFrameCanonicalDataset(split='train', num_points=args.num_obj_points,
    #                                    normalize_obj=args.normalize_obj)
    # valid_dataset = PelvisFrameCanonicalDataset(split='valid', num_points=args.num_obj_points,
    #                                    normalize_obj=args.normalize_obj)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True, pin_memory=False)  #pin_memory cause warning in pytorch 1.9.0
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                            drop_last=True, pin_memory=False)
    print('dataset loaded')

    if args.resume_checkpoint is not None:
        print('resume training')
        model = LitTransformNet.load_from_checkpoint(args.resume_checkpoint, args=args)
    else:
        print('start training from scratch')
        model = LitTransformNet(args)

    # callback
    tb_logger = pl_loggers.TensorBoardLogger(str(results_folder / 'transform'), name=args.expr_name)
    save_dir = Path(tb_logger.log_dir)  # for this version
    print(save_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                       monitor="val_loss",
                                                       save_weights_only=True, save_last=True)
    print(checkpoint_callback.dirpath)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=10000, verbose=False,
                                                     mode="min")
    profiler = SimpleProfiler() if args.profiler == 'simple' else AdvancedProfiler(output_filename='profiling.txt')

    # trainer
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=tb_logger,
                                            profiler=profiler,
                                            callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, train_loader, val_loader)



