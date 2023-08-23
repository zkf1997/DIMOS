import os
import sys
sys.path.append('..')
sys.path.append('../POSA')

from configuration.config import *
if not local_machine:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import smplx
import trimesh
from datetime import datetime
import pickle
import shutil
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

from interaction.body_encoder import BodyEncoder
from interaction.mesh import Mesh
from interaction.dataset import InteractionFeatureDataset
from interaction.chamfer_distance import chamfer_contact_loss
from interaction.viz_util import render_interaction_multview, render_body_multview
from data.scene import scenes, to_trimesh
from data.utils import *
from interaction.smplx_regressor import SMPLX_Regressor
from interaction.loss import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def code_to_name(verb_code):
    verb_id = torch.nonzero(verb_code.cpu()).numpy()[0]
    return action_names[int(verb_id)]

def dists_to_feature(dists, thresh=0.05):
    min_dists, min_idx = torch.min(dists, dim=-1)
    contact_feature_threshold = thresh
    # contact_feature_threshold = 0.1
    min_idx[min_dists > contact_feature_threshold] = obj_category_num - 1  # for vertices without close enough contacts, set semantic to 41 unlabeled
    fc = (min_dists <= contact_feature_threshold).type(torch.float32)
    fs = F.one_hot(min_idx, num_classes=obj_category_num).type(torch.float32)
    return torch.cat([fc.unsqueeze(2), fs], dim=2)

# visualize body-scene contacts with distance less than 5 cm
def visualize_vertex_obj_dists(x, f, faces, base_color):
    num_vertex = x.shape[0]
    vertices = x
    fc = f[:, 0]
    fs = f[:, 1:]
    contact_obj = np.argmax(fs, axis=1)
    obj_colors = np.array(category_dict.loc[contact_obj]['color'].to_numpy().tolist())  # Vx3
    # print(obj_colors.shape)
    obj_colors = obj_colors / 255.0
    have_contact = (fc > 0.5)
    colors = np.tile(base_color, (num_vertex, 1))
    colors[have_contact, :] = obj_colors[have_contact, :]

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors,
    )

def get_template_features(mesh):
    mesh_level = 2
    V = mesh.num_vertices[mesh_level]
    body_part_vertices = mesh.body_part_vertices
    template_features = np.zeros((4 * 42, V, 43), dtype=np.float32)
    template_features[:, :, 42] = 1.0
    verb_to_vertices = []
    for verb_id in range(4):
        body_parts = action_body_part_mapping[action_names[verb_id]]
        vertices = []
        for body_part in body_parts:
            vertices = vertices + body_part_vertices[body_part]
        verb_to_vertices.append(vertices)

    for interaction_id in range(4 * 42):
        verb_id = interaction_id // 42
        noun_id = interaction_id % 42
        vertices = verb_to_vertices[verb_id]
        template_features[interaction_id, vertices, 0] = 1.0
        template_features[interaction_id, vertices, 1:] = np.eye(42)[noun_id]

    return template_features

class LitBodyEncoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.start_time = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
        if isinstance(args, dict):
            args = Namespace(**args)
        self.save_hyperparameters(args)
        # print(args)

        mesh = Mesh(num_downsampling=args.init_downsample_level)
        args.device = device
        print(args.device)

        self.model = BodyEncoder(mesh, args)
        self.mesh = mesh
        # self.template_features = torch.from_numpy(get_template_features(mesh)).to(args.device)
        if args.use_regressor:
            self.smplx_regressor = SMPLX_Regressor(mesh, args)
            self.body_model = smplx.create(smplx_model_folder, model_type='smplx',
                     gender='neutral', ext='npz',
                     num_pca_comps=num_pca_comps, batch_size=args.batch_size).to(device)

        self.args = args

        self.coord_criterion = torch.nn.L1Loss(reduction='mean')
        self.dist_criterion = torch.nn.L1Loss(reduction='mean')
        self.laplacian_criterion = LaplacianLoss(
            torch.tensor(mesh.faces, device=device)
        )
        self.normal_criterion = NormalVectorLoss(mesh.faces)
        self.edge_length_criterion = EdgeLengthLoss(mesh.faces, bool(args.relative_length))
        self.normal_consistency_criterion = NormalConsistencyLoss(torch.tensor(mesh.faces, device=device))


    def forward(self, body_vertices, features, interaction_code):
        return self.model(body_vertices, features, interaction_code)

    def configure_optimizers(self):
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)
        param_list = list(self.model.parameters())
        if self.args.use_regressor:
            param_list += list(self.smplx_regressor.parameters())
        optimizer = torch.optim.Adam(params=param_list,
                                     lr=self.args.lr,
                                     weight_decay=self.args.l2_norm)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.9, verbose=True)
        return ({'optimizer': optimizer,
                 })

    def calc_loss(self, x, x_hat, f, f_hat, q_z, interaction, interaction_code, batch=None):
        batch_size = x.shape[0]
        loss_coord = self.coord_criterion(x_hat, x)
        # reconstruction of contact semantic features
        loss_fc = F.binary_cross_entropy(f_hat[:, :, 0], f[:, :, 0])
        targets = f[:, :, 1:].argmax(dim=-1).type(torch.long).reshape(batch_size, -1)
        loss_fs = F.cross_entropy(f_hat[:, :, 1:].permute(0, 2, 1), targets, ignore_index=self.args.ignore_index)
        # loss_laplacian = self.laplacian_criterion(x_hat, x)
        loss_laplacian = torch.tensor(0.0, device=x.device)
        loss_normal = self.normal_criterion(x_hat, x)
        loss_edge_length = self.edge_length_criterion(x_hat, x)
        loss_normal_consistency = self.normal_consistency_criterion(x_hat)

        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((x.shape[0], self.args.latent_dimension), requires_grad=False, device=device),
            scale=torch.ones((x.shape[0], self.args.latent_dimension), requires_grad=False, device=device))
        loss_kl = torch.mean(torch.mean(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1])) \
            if self.args.model == 'VAE' else torch.tensor(0.0, dtype=torch.float32,
                                     device=device)

        loss_dict = dict(coord=loss_coord,
                         fc=loss_fc,
                         fs=loss_fs,
                         # fc_t=loss_fc_t,
                         # fs_t=loss_fs_t,
                         laplacian=loss_laplacian,
                         normal=loss_normal,
                         edge_length=loss_edge_length,
                         kl=loss_kl,
                         normal_consistency=loss_normal_consistency,
                         # pelvis=loss_pelvis,
                         )

        annealing_factor = min(1.0, max(float(self.current_epoch) / (self.args.second_stage), 0)) if self.args.use_annealing else 1
        weighted_loss_dict = {
            'coord': loss_dict['coord'] * self.args.weight_coord,
            'fc': loss_dict['fc'] * self.args.weight_fc,
            'fs': loss_dict['fs'] * self.args.weight_fs,
            # 'fc_t': loss_dict['fc_t'] * self.args.weight_fc_t * max(annealing_factor ** 2, 0),
            # 'fs_t': loss_dict['fs_t'] * self.args.weight_fs_t * max(annealing_factor ** 2, 0),
            'laplacian': loss_dict['laplacian'] * self.args.weight_laplacian * annealing_factor,
            'normal': loss_dict['normal'] * self.args.weight_normal,
            'normal_consistency': loss_dict['normal_consistency'] * self.args.weight_normal_consistency,
            'edge_length': loss_dict['edge_length'] * self.args.weight_edge_length,
            'kl':
                max(annealing_factor ** 2, 0) *
                self.args.weight_kl * loss_dict['kl'],
        }

        loss = torch.stack(list(weighted_loss_dict.values())).sum()

        return loss, loss_dict, weighted_loss_dict

    def common_step(self, split, batch, batch_idx):
        smplx_param, pelvis, joints, body_vertices, \
        vertex_obj_dists, interaction, verb_code, noun_code, interaction_code, scene_name, node_idx = batch

        x_downsample = self.mesh.downsample(body_vertices)
        f = dists_to_feature(vertex_obj_dists, thresh=self.args.contact_thresh)
        x_hat, f_hat, q_z = self(x_downsample, f, interaction_code)
        # x_hat = self.mesh.upsample(x_hat)
        loss, loss_dict, weighted_loss_dict = self.calc_loss(x_downsample, x_hat, f, f_hat, q_z, interaction, interaction_code)

        # smplx_regressor
        if self.args.use_regressor:
            pred_smplx_param, pred_body_vertices = self.regress_smplx(x_hat)
            smplx_loss, smplx_loss_dict = self.calc_smplx_loss(smplx_param, body_vertices, pred_smplx_param,
                                                               pred_body_vertices)
            loss = loss + smplx_loss
            loss_dict.update(smplx_loss_dict)

        # visualization
        vis_step = 1024 if split == 'train' else 512
        if batch_idx % vis_step == 0:
            batch_size = x_hat.shape[0]
            x_sample, f_sample = self.model.sample(batch_size, interaction_code)
            if self.args.use_regressor:
                x_hat_smplx = pred_body_vertices
                f_hat_smplx = self.mesh.upsample(f_hat)
                _, x_sample_smplx = self.regress_smplx(x_sample)
                f_sample_smplx = self.mesh.upsample(f_sample)
                output = self.body_model(**smplx_param)
                x_smplx = output.vertices - output.joints[:, 0, :].unsqueeze(1)
                f_smplx = self.mesh.upsample(f)
            for idx in range(min(batch_size, 4)):
                body = visualize_vertex_obj_dists(x_downsample[idx].detach().cpu().numpy(), f[idx].detach().cpu().numpy(), self.mesh.faces, (0.8, 0.8, 0.8))
                body_rec = visualize_vertex_obj_dists(x_hat[idx].detach().cpu().numpy(), f_hat[idx].detach().cpu().numpy(), self.mesh.faces, (0.8, 0.0, 0.0))
                base_name = '_E{:03d}_It{:04d}_loss_{:.4f}_id{:d}_{}.png'.format(
                    self.current_epoch, batch_idx, loss_dict['coord'].item(), idx, interaction[idx])
                export_file = Path.joinpath(save_dir, 'render', 'gt' + base_name)
                export_file.parent.mkdir(exist_ok=True)
                # img_grid = render_body_multview(body)
                # img_grid.save(str(export_file))

                export_file = Path.joinpath(save_dir, 'render', 'rec' + base_name)
                img_grid = render_body_multview(body_rec)
                img_grid.save(str(export_file))

                export_file = Path.joinpath(save_dir, 'render', 'contrast' + base_name)
                img_grid = render_body_multview(body_rec, body_contrast=body)
                img_grid.save(str(export_file))

                body_sample = visualize_vertex_obj_dists(x_sample[idx].detach().cpu().numpy(), f_sample[idx].detach().cpu().numpy(), self.mesh.faces, (0.8, 0.0, 0.0))
                export_file = Path.joinpath(save_dir, 'render', 'sample' + base_name)
                img_grid = render_body_multview(body_sample)
                img_grid.save(str(export_file))

                if self.args.use_regressor:
                    body_smplx = visualize_vertex_obj_dists(x_smplx[idx].detach().cpu().numpy(),
                                                                f_smplx[idx].detach().cpu().numpy(),
                                                                self.mesh.meshes[0].faces,
                                                                (0.8, 0.8, 0.8))
                    body_rec_smplx = visualize_vertex_obj_dists(x_hat_smplx[idx].detach().cpu().numpy(),
                                                          f_hat_smplx[idx].detach().cpu().numpy(), self.mesh.meshes[0].faces,
                                                          (0.8, 0.0, 0.0))
                    body_sample_smplx = visualize_vertex_obj_dists(x_sample_smplx[idx].detach().cpu().numpy(),
                                                          f_sample_smplx[idx].detach().cpu().numpy(), self.mesh.meshes[0].faces,
                                                          (0.8, 0.0, 0.0))
                    export_file = Path.joinpath(save_dir, 'render', 'smplx_contrast' + base_name)
                    img_grid = render_body_multview(body_rec_smplx, body_contrast=body_smplx)
                    img_grid.save(str(export_file))

                    export_file = Path.joinpath(save_dir, 'render', 'smplx_sample' + base_name)
                    img_grid = render_body_multview(body_sample_smplx)
                    img_grid.save(str(export_file))
                # break

        return loss, loss_dict, weighted_loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self.common_step('train', batch, batch_idx)

        self.log('train_loss', loss, prog_bar=False)
        for key in loss_dict:
            self.log(key, loss_dict[key], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, weighted_loss_dict = self.common_step('valid', batch, batch_idx)

        for key in loss_dict:
            self.log('val_' + key, loss_dict[key], prog_bar=True)
        self.log('val_loss', loss)

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
        input = torch.cat([x_hat.detach(),
                           self.mesh.ref_vertices[None, :, :].expand(batch_size, -1, -1)], dim=-1)
        pred_smplx_param = self.smplx_regressor(input)
        smplx_output = self.body_model(**pred_smplx_param)
        pred_body_vertices = smplx_output.vertices - smplx_output.joints[:, 0, :].unsqueeze(1)

        return pred_smplx_param, pred_body_vertices

    def generate(self, interaction_code):
        batch_size = interaction_code.shape[0]
        self.eval()
        x_sample, f_sample = self.model.sample(batch_size, interaction_code)
        if self.args.use_regressor:
            sample_smplx_param, x_sample_smplx = self.regress_smplx(x_sample)
            # f_sample_smplx = self.mesh.upsample(f_sample)
            f_sample_smplx = f_sample
            return self.mesh.downsample(x_sample_smplx), f_sample, sample_smplx_param
        else:
            return x_sample, f_sample

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    # args
    parser = ArgumentParser()
    parser.add_argument("--contact_thresh", type=float, default=0.05)

    parser.add_argument("--model", type=str, default='VAE')
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--obj_geometry_dim", type=int, default=1024)
    parser.add_argument("--obj_category_dim", type=int, default=0)
    parser.add_argument("--verb_dim", type=int, default=4)
    parser.add_argument("--transl_latent_dimension", type=int, default=128)
    parser.add_argument("--latent_dimension", type=int, default=128)
    parser.add_argument("--init_downsample_level", type=int, default=2)
    parser.add_argument("--final_downsample_level", type=int, default=4)
    parser.add_argument("--encoder_channels", type=int, default=128)
    parser.add_argument("--decoder_channels", type=int, default=512)
    parser.add_argument("--conv_per_level", type=int, default=1)
    parser.add_argument("--residual", type=int, default=0)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2_norm", type=float, default=0)
    parser.add_argument("--laplacian_method", type=str, default='cot')
    parser.add_argument("--relative_length", type=int, default=0)
    parser.add_argument("--use_annealing", type=int, default=0)
    parser.add_argument("--weight_coord", type=float, default=1)
    parser.add_argument("--weight_fc", type=float, default=1)
    parser.add_argument("--weight_fs", type=float, default=1)
    parser.add_argument("--weight_fc_t", type=float, default=0)
    parser.add_argument("--weight_fs_t", type=float, default=0)
    parser.add_argument("--weight_normal", type=float, default=0.5)
    parser.add_argument("--weight_normal_consistency", type=float, default=0)
    parser.add_argument("--weight_edge_length", type=float, default=10)
    parser.add_argument("--weight_kl", type=float, default=1e-2)
    parser.add_argument("--weight_laplacian", type=float, default=0)
    parser.add_argument("--weight_pelvis", type=float, default=1)
    parser.add_argument("--weight_contact", type=float, default=1)
    parser.add_argument("--weight_penetration", type=float, default=0)  #10
    parser.add_argument("--weight_smplx_rec", type=float, default=1)
    parser.add_argument("--weight_smplx_rot", type=float, default=1)
    parser.add_argument("--weight_smplx_nonrot", type=float, default=0.1)
    parser.add_argument("--ignore_index", type=int, default=-100)

    parser.add_argument("--seq_length", type=int, default=9)
    parser.add_argument("--dilation", type=int, default=1)

    parser.add_argument("--use_regressor", type=int, default=0)
    parser.add_argument("--learned_prior", type=int, default=0)
    parser.add_argument("--use_kronecker", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profiler", type=str, default=None, help='simple or advanced')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--second_stage", type=int, default=20,
                        help="annealing some loss weights in early epochs before this num")
    parser.add_argument("--used_interaction", type=str, default='all')
    parser.add_argument("--skip_composite", type=str, default='no')
    parser.add_argument("--expr_name", type=str, default=datetime.now().strftime("%H:%M:%S.%f"))
    parser.add_argument("--render_thresh", type=float, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--data_overwrite", type=int, default=0)

    args = parser.parse_args()

    # make demterministic
    pl.seed_everything(233, workers=True)
    # args.deterministic = True
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    # data
    with open(Path.joinpath(project_folder, "data", 'train.pkl'), 'rb') as data_file:
        train_data = pickle.load(data_file)
    with open(Path.joinpath(project_folder, "data", 'test.pkl'), 'rb') as data_file:
        test_data = pickle.load(data_file)
    train_dataset = InteractionFeatureDataset(train_data, split='train', num_points=args.num_points, use_augment=True,
                                              used_interaction=args.used_interaction,
                                              data_overwrite=args.data_overwrite,
                                              skip_composite=args.skip_composite,
                                              )
    test_dataset = InteractionFeatureDataset(test_data, split='test', num_points=args.num_points, use_augment=False,
                                       used_interaction=args.used_interaction, data_overwrite=args.data_overwrite,
                                             skip_composite=args.skip_composite,
                                       )
    # test_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True, pin_memory=False)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                            drop_last=True, pin_memory=False)
    print('dataset loaded')

    #callback
    tb_logger = pl_loggers.TensorBoardLogger(str(results_folder / 'body_mesh'), name=args.expr_name)
    save_dir = Path(tb_logger.log_dir)  #for this version
    print(save_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=str(save_dir / 'checkpoints'),
        monitor="val_loss", save_weights_only=True, save_last=True)
    print(checkpoint_callback.dirpath)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1000, verbose=False, mode="min")
    # args.callbacks=[checkpoint_callback, early_stop_callback]   #cannot pass callbacks in args, otherwise saving hyperparams will lead to serialization error and memory leakage
    # trainer
    if args.profiler == 'simple':
        profiler = SimpleProfiler()
    elif args.profiler == 'advanced':
        profiler = AdvancedProfiler(output_filename='profiling.txt')
    else:
        profiler = None
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=tb_logger,
                                            profiler=profiler,
                                            progress_bar_refresh_rate=1 if local_machine else 256,
                                            callbacks=[checkpoint_callback, early_stop_callback])
    model = LitBodyEncoder.load_from_checkpoint(args.resume_checkpoint, args=args) if args.resume_checkpoint is not None else LitBodyEncoder(args)
    trainer.fit(model, train_loader, val_loader)


