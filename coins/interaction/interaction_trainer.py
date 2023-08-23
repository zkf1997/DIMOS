import sys
sys.path.append('..')
import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import smplx
import trimesh
import pyrender
from datetime import datetime
import pickle
import shutil
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
import open3d as o3d
from datetime import datetime
from copy import deepcopy
from argparse import ArgumentParser, Namespace

from configuration.config import *
from configuration.joints import *
from data.scene import scenes, to_trimesh
from interaction.mesh import Mesh
from interaction.dataset import InteractionDataset
from interaction.chamfer_distance import chamfer_contact_loss, chamfer_dists
from interaction.viz_util import render_composite_interaction_multview, render_body_multview
from interaction.smplx_regressor import SMPLX_Regressor, SMPLX_Regressor_Joint, SMPLX_Regressor_Joint_Orient
from interaction.loss import *
from interaction.interaction_model import *
from data.utils import *
# from interaction.interaction_loss import *
from interaction.utils import *

def joints_to_bones(joints, num_body_points=22):
    B, J, _ = joints.shape
    joint_idx = np.arange(1, num_body_points)
    parent_idx = np.asarray(parent_joint_idx[joint_idx])
    bone = joints[:, joint_idx, :] - joints[:, parent_idx, :]
    bone_length = (bone ** 2).sum(dim=2)
    return bone_length

class LitInteraction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        if not hasattr(args, 'use_orient'):
            args.use_orient = 0
        if not hasattr(args, 'dim_point_feature'):
            args.dim_point_feature = 6
        args.contact_dim = 2
        args.orient_dim = 6
        self.args = args
        self.save_hyperparameters(args)
        self.start_time = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")

        if args.body_type == 'mesh':
            mesh = Mesh(num_downsampling=args.init_downsample_level)
            self.args.num_body_points = mesh.ref_vertices.shape[0]
            print('body representation of mesh with ',  self.args.num_body_points, ' points')
            args.template_body = mesh.ref_vertices
            upper_body = []
            for part in upper_body_parts:
                upper_body += mesh.body_part_vertices_full[part]
            lower_body = []
            for part in lower_body_parts:
                lower_body += mesh.body_part_vertices_full[part]
            args.body_segment = (upper_body, lower_body)
            self.mesh = mesh
            if args.use_regressor:
                self.smplx_regressor = SMPLX_Regressor(mesh, args)
                self.body_model = smplx.create(smplx_model_folder, model_type='smplx',
                                               gender='neutral', ext='npz',
                                               num_pca_comps=num_pca_comps, batch_size=args.batch_size).to(device)

            self.coord_criterion = torch.nn.L1Loss(reduction='mean')
            self.normal_criterion = NormalVectorLoss(mesh.faces)
            self.edge_length_criterion = EdgeLengthLoss(mesh.faces, bool(args.relative_length))
            self.laplacian_criterion = LaplacianLoss(torch.tensor(mesh.faces, device=device))
            self.normal_consistency_criterion = NormalConsistencyLoss(torch.tensor(mesh.faces, device=device))
        elif args.body_type == 'joint':
            self.contact_vertices_list = [
                [1, 2],  # sit
                [1, 2, 3, 6, 9],  # lie
                [7, 8, 10, 11],  # stand
                [18, 19, 20, 21],  # touch
            ]
            if args.use_regressor:
                self.smplx_regressor = SMPLX_Regressor_Joint(num_joints=args.num_body_points) if not args.use_orient else SMPLX_Regressor_Joint_Orient(num_joints=args.num_body_points)
            self.body_model = smplx.create(smplx_model_folder, model_type='smplx',
                                           gender='neutral', ext='npz',
                                           num_pca_comps=num_pca_comps, batch_size=args.batch_size).to(device)
            ref_joints = self.body_model().joints[0, :args.num_body_points, :].detach().to(device)
            center = 0.5 * (ref_joints.max(dim=0)[0] + ref_joints.min(dim=0)[0])[None]
            ref_joints -= center
            ref_joints /= ref_joints.abs().max().item()
            args.template_body = args.ref_joints = ref_joints  # Pbx3
            if args.use_orient:
                args.dim_body_points += args.orient_dim
                args.template_body = torch.cat((args.template_body,
                                                torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32,
                                                             device=device).expand(args.template_body.shape[0], 6)
                                                ), dim=-1)
        if args.use_contact_feature:
            args.dim_body_points += args.contact_dim
            args.template_body = torch.cat((args.template_body, torch.zeros_like(args.template_body[:, :2])), dim=-1)

        if args.model =='InteractionVAE':
            self.model = InteractionVAE(args)
        else:
            print('not implemented')
            return

    # def on_train_start(self) -> None:
    #     #     backup trainer.py and model
    #     shutil.copy('./interaction_trainer.py', str(save_dir / 'interaction_trainer.py'))
    #     shutil.copy('./interaction_model.py', str(save_dir / 'interaction_model.py'))
    #     return

    def forward(self, x, batch):
        return self.model(x, batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=list(self.model.parameters())+list(self.smplx_regressor.parameters()) if self.args.use_regressor else self.model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=self.args.l2_norm)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.9, verbose=True)
        return ({'optimizer': optimizer,
                 # 'lr_scheduler': {
                 #    'scheduler': lr_scheduler,
                 #    'reduce_on_plateau': True,
                 #    # val_checkpoint_on is val_loss passed in as checkpoint_on
                 #    'monitor': 'joint'
                 #    }
                 })

    def calc_loss(self, x, x_hat, q_z, batch):
        batch_size = x.shape[0]
        # if all enabled, postion: [0:3), orient: [3:9), contact: [9:11)
        x, orient, contact = x[:, :, :3], x[:, :, 3:-self.args.contact_dim], x[:, :, -self.args.contact_dim:]
        x_hat, orient_hat, contact_hat = x_hat[:, :, :3], x_hat[:, :, 3:-self.args.contact_dim], x_hat[:, :, -self.args.contact_dim:]

        # contact loss
        loss_contact = torch.tensor(0.0).to(x.device)

        # penetration loss
        loss_penetration = torch.tensor(0.0).to(x.device)

        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((x.shape[0], self.args.latent_dim), requires_grad=False, device=device),
            scale=torch.ones((x.shape[0], self.args.latent_dim), requires_grad=False, device=device))
        loss_kl = torch.mean(torch.mean(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))
        if self.args.robust_kl:
            loss_kl = torch.sqrt(loss_kl * loss_kl + 1) - 1.0

        loss_dict = dict(penetration=loss_penetration,
                         kl=loss_kl,
                         contact=loss_contact,
                         )
        annealing_factor = min(1.0, max(float(self.current_epoch) / (self.args.second_stage), 0)) if self.args.use_annealing else 1
        weighted_loss_dict = {
            'contact': max(annealing_factor ** 2, 0) * loss_contact * self.args.weight_contact,
            'penetration': max(annealing_factor ** 2, 0) * loss_dict['penetration'] * self.args.weight_penetration,
            'kl':
                max(annealing_factor ** 2, 0) *
                self.args.weight_kl * loss_dict['kl'],
        }

        # contact feature loss
        if self.args.use_contact_feature:
            loss_contact_rec = F.binary_cross_entropy(contact_hat, contact)
            dist_hat = chamfer_dists(x_hat, batch['object_pointclouds'][:, :, :, :3].reshape(x.shape[0], -1, 3))
            loss_contact_dist = (dist_hat * contact_hat[:, :, 0]).mean()
            loss_dict.update({
                'contact_rec': loss_contact_rec,
                'contact_dist': loss_contact_dist,
            })
            weighted_loss_dict.update({
                'contact_rec': loss_contact_rec * self.args.weight_contact_rec,
                'contact_dist': loss_contact_dist * self.args.weight_contact_dist,
            })

        if self.args.body_type == 'joint':
            # loss_pelvis = F.l1_loss(x_hat[:, 0, :], x[:, 0, :])
            loss_joints = F.l1_loss(x_hat, x)
            loss_orient = F.l1_loss(orient_hat, orient) if self.args.use_orient else torch.tensor(0.0).to(x.device)
            loss_bones = F.l1_loss(joints_to_bones(x_hat, num_body_points=self.args.num_body_points),
                                   joints_to_bones(x, num_body_points=self.args.num_body_points),
                                   )
            loss_template = F.l1_loss(joints_to_bones(x_hat, num_body_points=self.args.num_body_points),
                                   joints_to_bones(batch['template_body'][:, :, :3], num_body_points=self.args.num_body_points),
                                   ) if self.args.template_type == 'personal' else torch.tensor(0.0).to(x.device)
            loss_dict.update({
                'joint': loss_joints,
                'orient': loss_orient,
                'template': loss_template,
                'bone': loss_bones,
            })
            weighted_loss_dict.update({
                'joint': loss_joints * self.args.weight_joint,
                'orient': loss_orient * self.args.weight_orient,
                'template': loss_template * self.args.weight_template,
                'bone': loss_bones * self.args.weight_bone,
            })
        elif self.args.body_type == 'mesh':

            loss_coord = self.coord_criterion(x_hat, x)
            loss_normal = self.normal_criterion(x_hat, x)
            loss_edge_length = self.edge_length_criterion(x_hat, x)
            loss_template = self.edge_length_criterion(x_hat, batch['template_body'][:, :, :3]) if self.args.template_type == 'personal' else torch.tensor(0.0).to(x.device)
            loss_laplacian = self.laplacian_criterion(x_hat, x)
            loss_normal_consistency = self.normal_consistency_criterion(x_hat)
            loss_dict.update({
                'coord': loss_coord,
                'normal': loss_normal,
                'edge_length': loss_edge_length,
                'template': loss_template,
                'laplacian': loss_laplacian,
                'normal_consistency': loss_normal_consistency,
            })
            weighted_loss_dict.update({
                'coord': loss_coord * self.args.weight_coord,
                'normal': loss_normal * self.args.weight_normal,
                'edge_length': loss_edge_length * self.args.weight_edge_length,
                'template': loss_template * self.args.weight_template,
                'laplacian': loss_laplacian * self.args.weight_laplacian,
                'normal_consistency': loss_normal_consistency * self.args.weight_normal_consistency,
            })

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict

    def calc_smplx_loss(self, smplx_param, body_vertices, pred_smplx_param, pred_body_vertices):
        loss_rec = torch.nn.L1Loss(reduction='mean')(body_vertices, pred_body_vertices)
        loss_rot = torch.nn.MSELoss(reduction='mean')(smplx_dict_to_rotmat(smplx_param), pred_smplx_param['rotmat'])
        loss_nonrot = torch.nn.MSELoss(reduction='mean')(smplx_dict_to_nonrot(smplx_param, include_transl=False),
                                                         pred_smplx_param['nonrot'])
        loss_dict = dict(smplx_rec=loss_rec,
                         smplx_param_rot=loss_rot,
                         smplx_param_nonrot=loss_nonrot
                         )

        loss = loss_rec * self.args.weight_smplx_rec \
               + loss_rot * self.args.weight_smplx_rot * (0.8 ** self.current_epoch)\
               + loss_nonrot * self.args.weight_smplx_nonrot * (0.8 ** self.current_epoch)
        return loss, loss_dict

    def regress_smplx(self, x_hat):
        batch_size = x_hat.shape[0]
        if not self.args.use_orient:
            input = torch.cat([x_hat[:, :, :3].detach(),
                           self.args.template_body[None, :, :3].expand(batch_size, -1, -1)], dim=-1)
        else:
            input = x_hat[:, :, :9].detach()
        pred_smplx_param = self.smplx_regressor(input)
        smplx_output = self.body_model(**pred_smplx_param)
        # smplx regressor do not predict translation, suppose center is pelvis
        pred_body_vertices = smplx_output.vertices - smplx_output.joints[:, 0, :].unsqueeze(1)
        # assign transl
        pred_smplx_param['transl'] = -smplx_output.joints[:, 0, :]

        return pred_smplx_param, pred_body_vertices

    def _common_step(self, batch, batch_idx, mode):
        x = None
        if self.args.body_type == 'joint':
            x = batch['joints'][:, :self.args.num_body_points, :]  # BxJx3
            if self.args.use_orient:
                full_pose = batch['full_pose'].reshape(-1, self.args.num_body_points, 3)
                full_pose = pytorch3d.transforms.axis_angle_to_matrix(full_pose)[:, :, :3, :2].reshape(-1, self.args.num_body_points, 6)
                x = torch.cat((x, full_pose), dim=-1)
            # contact feature
            if self.args.use_contact_feature:
                # dists = chamfer_dists(x, batch['object_pointclouds'][:, :, :, :3].reshape(x.shape[0], -1, 3))
                dists = batch['joint_contact_dist'][:, :self.args.num_body_points]
                sdf = batch['joint_sdf'][:, :self.args.num_body_points]
                contact_semantic = (dists < self.args.contact_semantic_thresh).float()
                contact_scene = (sdf < self.args.contact_scene_thresh).float()
                x = torch.cat((x, contact_semantic.unsqueeze(2), contact_scene.unsqueeze(2)), dim=-1)  # BxPx5
            if self.args.template_type == 'personal':
                smplx_output = self.body_model(betas=batch['smplx_param']['betas'])
                # smplx regressor do not predict translation, suppose center is pelvis
                batch['template_body'] = smplx_output.joints[:, :55, :] - smplx_output.joints[:, 0, :].unsqueeze(1)
                if self.args.use_orient:
                    batch['template_body'] = torch.cat((batch['template_body'],
                                                        torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32,
                                                             device=device).expand(batch['template_body'].shape[0], batch['template_body'].shape[1], 6)
                                                        ), dim=-1)
                if self.args.use_contact_feature:
                    batch['template_body'] = torch.cat((batch['template_body'], torch.zeros_like(batch['template_body'][:, :, :2])), dim=-1)
        elif self.args.body_type == 'mesh':
            x = batch['body_vertices']
            x = self.mesh.downsample(x)
            # contact feature
            if self.args.use_contact_feature:
                # dists = chamfer_dists(x, batch['object_pointclouds'][:, :, :, :3].reshape(x.shape[0], -1, 3))
                dists = batch['contact_dist']
                sdf = batch['sdf']
                contact_semantic = (dists < self.args.contact_semantic_thresh).float()
                contact_scene = (sdf < self.args.contact_scene_thresh).float()
                x = torch.cat((x, contact_semantic.unsqueeze(2), contact_scene.unsqueeze(2)), dim=-1)  # BxPx5
            if self.args.template_type == 'personal':
                smplx_output = self.body_model(betas=batch['smplx_param']['betas'])
                # smplx regressor do not predict translation, suppose center is pelvis
                template_body = smplx_output.vertices - smplx_output.joints[:, 0, :].unsqueeze(1)
                batch['template_body'] = self.mesh.downsample(template_body)
                if self.args.use_contact_feature:
                    batch['template_body'] = torch.cat((batch['template_body'], torch.zeros_like(batch['template_body'][:, :, :2])), dim=-1)
        x_hat, q_z = self(x, batch)
        loss, loss_dict, weighted_loss_dict = self.calc_loss(x, x_hat, q_z, batch=batch)

        # smplx_regressor
        if self.args.use_regressor:
            pred_smplx_param, pred_body_vertices = self.regress_smplx(x_hat)
            smplx_loss, smplx_loss_dict = self.calc_smplx_loss(batch['smplx_param'], batch['body_vertices'] - batch['pelvis'].unsqueeze(1), pred_smplx_param,
                                                               pred_body_vertices)
            loss = loss + smplx_loss
            loss_dict.update(smplx_loss_dict)

        # render reconstructed and sampled interactions
        render_interval = 256 if mode == 'valid' else 512
        if (batch_idx % render_interval == 0) and (self.current_epoch >= self.args.render_epoch or self.args.debug):
            x_sample = self.model.sample(batch)[0]
            x, contact = x[:, :, :3],  x[:, :, -self.args.contact_dim]
            x_hat, contact_hat = x_hat[:, :, :3], x_hat[:, :, -self.args.contact_dim]
            x_sample, contact_sample = x_sample[:, :, :3], x_sample[:, :, -self.args.contact_dim]
            centroid, rotation = batch['centroid'], batch['rotation']
            x = transform_back(x, centroid, rotation)
            x_hat = transform_back(x_hat, centroid, rotation)
            x_sample = transform_back(x_sample, centroid, rotation)
            batch_size = x.shape[0]
            render_num = 4
            obj_points_coord = transform_back(batch['object_pointclouds'][:, :, :, :3].reshape(batch_size, -1, 3), centroid, rotation).reshape(batch_size, maximum_atomics, -1, 3).cpu().numpy()
            for idx in range(min(batch_size, render_num)):
                # coords frame
                transform = np.eye(4, dtype=np.float32)
                transform[:3, :3] = rotation[idx].T.cpu().numpy()
                transform[:3, 3] = centroid[idx].cpu().numpy()
                coord_frame = trimesh.creation.axis(transform=transform, origin_color=(0.8, 0.8, 0.8))
                colors = np.array([[0.8, 0.1, 0.1]] * x.shape[1])
                if self.args.use_contact_feature:
                    colors[contact[idx].detach().cpu().numpy() > 0.5] = (1., 1., 0.)
                body = skeleton_to_mesh(x[idx].detach().cpu().numpy(), colors) \
                    if self.args.body_type == 'joint' else trimesh.Trimesh(
                    vertices=x[idx].detach().cpu().numpy(),
                    faces=self.mesh.faces,
                    vertex_colors=colors)
                colors = np.array([[0.1, 0.8, 0.1]] * x.shape[1])
                if self.args.use_contact_feature:
                    colors[contact_hat[idx].detach().cpu().numpy() > 0.5] = (1., 1., 0.)
                body_hat = skeleton_to_mesh(x_hat[idx].detach().cpu().numpy(), colors)                     \
                    if self.args.body_type == 'joint' else trimesh.Trimesh(
                    vertices=x_hat[idx].detach().cpu().numpy(),
                    faces=self.mesh.faces,
                    vertex_colors=colors)
                colors = np.array([[0.1, 0.1, 0.8]] * x.shape[1])
                if self.args.use_contact_feature:
                    colors[contact_sample[idx].detach().cpu().numpy() > 0.5] = (1., 1., 0.)
                body_sample = skeleton_to_mesh(x_sample[idx].detach().cpu().numpy(), colors) \
                    if self.args.body_type == 'joint' else trimesh.Trimesh(
                    vertices=x_sample[idx].detach().cpu().numpy(),
                    faces=self.mesh.faces,
                    vertex_colors=colors)
                obj_meshes = []
                for obj_idx in batch['interaction_obj_ids'][idx]:
                    if obj_idx != -1:
                        obj_mesh = scenes[batch['scene_name'][idx]].get_mesh_with_accessory(int(obj_idx))
                        obj_meshes.append(obj_mesh)

                base_name = mode + '_E{:03d}_It{:04d}_id{:d}_{}.png'.format(
                                                self.current_epoch, batch_idx, idx, batch['interaction'][idx])
                export_file = Path.joinpath(save_dir, 'render', 'rec_' + base_name[:-4] + '_body.png')
                export_file.parent.mkdir(exist_ok=True, parents=True)
                img_collage = render_body_multview(body=body_hat, use_material=False)
                img_collage.save(str(export_file))
                export_file = Path.joinpath(save_dir, 'render', 'rec_' + base_name)
                export_file.parent.mkdir(exist_ok=True, parents=True)
                img_collage = render_composite_interaction_multview(body_hat + coord_frame, obj_meshes,
                                                                    collage_mode='grid',
                                                                    use_material=False, smooth_body=False,
                                                                    obj_points_coord=obj_points_coord[idx, :batch['num_atomics'][idx], :, :])
                img_collage.save(str(export_file))

                export_file = Path.joinpath(save_dir, 'render', 'contrast_' + base_name[:-4] + '_body.png')
                export_file.parent.mkdir(exist_ok=True, parents=True)
                img_collage = render_body_multview(body=body_hat, body_contrast=body, use_material=False)
                img_collage.save(str(export_file))
                export_file = Path.joinpath(save_dir, 'render', 'contrast_' + base_name)
                img_collage = render_composite_interaction_multview(body_hat + coord_frame, obj_meshes,
                                                          body_contrast=body,
                                                          collage_mode='grid',
                                                                    use_material=False, smooth_body=False,
                                                          obj_points_coord=obj_points_coord[idx, :batch['num_atomics'][idx], :, :])
                img_collage.save(str(export_file))

                export_file = Path.joinpath(save_dir, 'render', 'sample_' + base_name[:-4] + '_body.png')
                export_file.parent.mkdir(exist_ok=True, parents=True)
                img_collage = render_body_multview(body=body_sample, use_material=False)
                img_collage.save(str(export_file))
                export_file = Path.joinpath(save_dir, 'render', 'sample_' + base_name)
                img_collage = render_composite_interaction_multview(body_sample + coord_frame, obj_meshes,
                                                          collage_mode='grid',
                                                                    use_material=False, smooth_body=False,
                                                          obj_points_coord=obj_points_coord[idx, :batch['num_atomics'][idx], :, :])
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
    parser.add_argument("--body_type", type=str, default='joint')
    parser.add_argument("--use_orient", type=int, default=0)
    parser.add_argument("--num_verb", type=int, default=4)
    parser.add_argument("--num_body_points", type=int, default=22)
    parser.add_argument("--use_pointnet2", type=int, default=0)
    parser.add_argument("--num_obj_points", type=int, default=4096)
    parser.add_argument("--num_obj_keypoints", type=int, default=16)
    parser.add_argument("--dim_body_points", type=int, default=3)
    parser.add_argument("--point_level", type=int, default=3)
    parser.add_argument("--use_contact_feature", type=int, default=0)
    parser.add_argument("--contact_dim", type=int, default=2)
    parser.add_argument("--orient_dim", type=int, default=6)
    parser.add_argument("--contact_semantic_thresh", type=float, default=0.05)
    parser.add_argument("--contact_scene_thresh", type=float, default=0.01)

    # transformer
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=512)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--interaction_bias", type=int, default=0)
    parser.add_argument("--mask_body", type=int, default=0)
    parser.add_argument("--mask_prob", type=float, default=0.05)
    parser.add_argument("--latent_usage", type=str, default='memory')
    parser.add_argument("--template_type", type=str, default='zero')
    parser.add_argument("--nerf_embed", type=int, default=0)
    parser.add_argument("--multires", type=int, default=10)

    parser.add_argument("--obj_geometry_dim", type=int, default=1024)
    parser.add_argument("--obj_category_dim", type=int, default=0)
    parser.add_argument("--verb_dim", type=int, default=4)
    parser.add_argument("--transl_latent_dimension", type=int, default=128)
    parser.add_argument("--latent_dimension", type=int, default=128)
    parser.add_argument("--init_downsample_level", type=int, default=2)
    parser.add_argument("--final_downsample_level", type=int, default=4)
    parser.add_argument("--num_channels", type=int, default=256)
    parser.add_argument("--encoder_channels", type=int, default=256)
    parser.add_argument("--decoder_channels", type=int, default=256)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)
    parser.add_argument("--laplacian_method", type=str, default='cot')
    parser.add_argument("--relative_length", type=int, default=1)
    parser.add_argument("--robust_kl", type=int, default=1)
    parser.add_argument("--weight_joint", type=float, default=1)
    parser.add_argument("--weight_orient", type=float, default=0)
    parser.add_argument("--weight_bone", type=float, default=1)
    parser.add_argument("--weight_coord", type=float, default=1)
    parser.add_argument("--weight_normal", type=float, default=0.5)
    parser.add_argument("--weight_edge_length", type=float, default=0.2)
    parser.add_argument("--weight_template", type=float, default=0.1)
    parser.add_argument("--weight_kl", type=float, default=1e-2)
    parser.add_argument("--weight_laplacian", type=float, default=0)
    parser.add_argument("--weight_normal_consistency", type=float, default=0)
    parser.add_argument("--weight_pelvis", type=float, default=0)
    parser.add_argument("--weight_contact", type=float, default=0)
    parser.add_argument("--weight_contact_rec", type=float, default=0)
    parser.add_argument("--weight_contact_dist", type=float, default=0)
    parser.add_argument("--weight_dist", type=float, default=1)
    parser.add_argument("--weight_penetration", type=float, default=0)  #10
    parser.add_argument("--weight_smplx_rec", type=float, default=1)
    parser.add_argument("--weight_smplx_rot", type=float, default=1)
    parser.add_argument("--weight_smplx_nonrot", type=float, default=0.1)

    parser.add_argument("--use_regressor", type=int, default=0)
    parser.add_argument("--raw_points", type=int, default=0)
    parser.add_argument("--dummy_obj", type=int, default=0)
    parser.add_argument("--use_contact", type=int, default=0)
    parser.add_argument("--use_annealing", type=int, default=0)
    parser.add_argument("--learned_prior", type=int, default=0)
    parser.add_argument("--use_kronecker", type=int, default=0)
    parser.add_argument("--freeze", type=int, default=0)

    # dataset
    parser.add_argument("--used_interaction", type=str, default='all')
    parser.add_argument("--used_instance", type=str, default=None)
    parser.add_argument("--scale_obj", type=int, default=0)
    parser.add_argument("--center_type", type=str, default='human')
    parser.add_argument("--rotation", type=str, default='human')
    parser.add_argument("--point_sample", type=str, default='random')
    parser.add_argument("--use_augment", type=str, default='')
    parser.add_argument("--data_overwrite", type=int, default=0)
    parser.add_argument("--skip_composite", type=str, default='')
    parser.add_argument("--include_motion", type=int, default=0)

    # train
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profiler", type=str, default='simple', help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--second_stage", type=int, default=20,
                        help="annealing some loss weights in early epochs before this num")
    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%H:%M:%S.%f"))
    parser.add_argument("--render_thresh", type=float, default=5e-3)
    parser.add_argument("--render_epoch", type=int, default=2333)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    # make demterministic
    pl.seed_everything(233, workers=True)
    # torch.autograd.set_detect_anomaly(True)
    # args.deterministic = True

    # data
    with open(Path.joinpath(project_folder, "data", 'train.pkl'), 'rb') as data_file:
        train_data = pickle.load(data_file)
    with open(Path.joinpath(project_folder, "data", 'test.pkl'), 'rb') as data_file:
        test_data = pickle.load(data_file)
    test_dataset = InteractionDataset(test_data if args.used_instance is None else train_data,
                                      num_points=args.num_obj_points, use_augment=False,
                                      include_motion=args.include_motion,
                                      used_interaction=args.used_interaction, split='test',
                                      center_type=args.center_type, scale_obj=args.scale_obj,
                                      used_instance=args.used_instance, rotation=args.rotation,
                                      data_overwrite=args.data_overwrite, point_sample=args.point_sample,
                                      )
    train_dataset = InteractionDataset(train_data, num_points=args.num_obj_points, use_augment=args.use_augment,
                                        used_interaction=args.used_interaction, split='train',
                                        center_type=args.center_type, scale_obj=args.scale_obj,
                                        used_instance=args.used_instance, rotation=args.rotation,
                                        skip_prox_composite=args.skip_composite,
                                        include_motion=args.include_motion,
                                        data_overwrite=args.data_overwrite, point_sample=args.point_sample,
                                        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True, pin_memory=False)  #pin_memory cause warning in pytorch 1.9.0
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                            drop_last=True, pin_memory=False)
    print('dataset loaded')

    if args.resume_checkpoint is not None:
        print('resume training')
        model = LitInteraction.load_from_checkpoint(args.resume_checkpoint, args=args)
    else:
        print('start training from scratch')
        model = LitInteraction(args)

    # callback
    tb_logger = pl_loggers.TensorBoardLogger(str(results_folder / 'interaction'), name=args.expr_name)
    save_dir = Path(tb_logger.log_dir)  # for this version
    print(save_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
                                                       monitor="val_loss",
                                                       save_weights_only=True, save_last=True)
    print(checkpoint_callback.dirpath)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000, verbose=False,
                                                     mode="min")
    profiler = SimpleProfiler() if args.profiler == 'simple' else AdvancedProfiler(output_filename='profiling.txt')

    # trainer
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=tb_logger,
                                            profiler=profiler,
                                            progress_bar_refresh_rate=1,
                                            callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, train_loader, val_loader)



