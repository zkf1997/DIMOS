# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import configargparse


def parse_config():
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'Interaction capture'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='Interaction capture')
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--base_dir', type=str, default='$POSA_dir')
    parser.add_argument('--data_dir', type=str, default='$POSA_dir/data')
    parser.add_argument('--PROX_dir', type=str, default='$POSA_dir/PROXD')
    parser.add_argument('--output_dir', type=str, default='$POSA_dir/training')
    parser.add_argument('--ds_us_dir', type=str, default='$POSA_dir/mesh_ds')
    parser.add_argument('--rand_samples_dir', type=str, default='$POSA_dir/random_samples',
                        help='')
    parser.add_argument('--affordance_dir', type=str, default='$POSA_dir/affordance', help='')
    parser.add_argument('--pkl_file_path', type=str, default='$POSA_dir/sample_poses')
    parser.add_argument('--rp_base_dir', default='$POSA_dir/rp_clothed_meshes', type=str, help='')
    parser.add_argument('--model_folder', type=str, default='$POSA_dir/smplx_models',
                        help='')

    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--save_checkpoints', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--load_checkpoint', type=int, default=0)

    parser.add_argument('--train_recordings_list', type=str, nargs='*', )
    parser.add_argument('--test_recordings_list', type=str, nargs='*', )
    parser.add_argument('--recordings_list', type=str, nargs='*', )
    parser.add_argument('--recording_name', type=str, default='')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--affordance_lr', type=float, default=1.0)
    parser.add_argument('--reduce_lr', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--drop_out', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--input_dim', type=int, default=2619)
    parser.add_argument('--nv', type=int, default=655)

    parser.add_argument('--x_dim', type=int, default=655)
    parser.add_argument('--cond_dim', type=int, default=0)
    parser.add_argument('--f_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=1)
    parser.add_argument('--num_hidden_layers', type=int, default=1)

    parser.add_argument('--contact_w', type=float, default=1)
    parser.add_argument('--semantics_w', type=float, default=1)
    parser.add_argument('--kl_w', type=float, default=1e-3)
    parser.add_argument('--pen_w', type=float, default=10.0)
    parser.add_argument('--pose_w', type=float, default=100.0)

    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--reduction', default='mean', type=str,
                        choices=['mean', 'sum'],
                        help='')
    parser.add_argument('--loss_type', default='mse', type=str,
                        choices=['l1', 'mse', 'bce'],
                        help='')
    parser.add_argument('--float_dtype', type=str, default='float32', help='The types of floats used')

    parser.add_argument('--train_data', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--num_pca_comps', type=int, default=6)
    parser.add_argument('--viz', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--show_scene', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--test', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--down_sample', type=int, default=1)
    parser.add_argument('--up_sample', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--use_semantics', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--gender', type=str, default='male', choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL' +
                             'model')
    parser.add_argument('--num_init_points', type=int, default=3)
    parser.add_argument('--num_rand_samples', type=int, default=3)
    parser.add_argument('--contact_threshold', type=float, default=0.05)
    parser.add_argument('--trunc_val', type=float, default=0.5)
    parser.add_argument('--show_contact', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--show_semantics', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--normalization_mode', default=None, type=str, help='')
    parser.add_argument('--num_groups', type=int, default=8)
    parser.add_argument('--affordance_step', type=float, default=0.2)
    parser.add_argument('--num_rendered_samples', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--show_init_pos', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--use_clothed_mesh', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--render', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--save_meshes', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--show_gen_sample', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--make_collage', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--no_obj_classes', type=int, default=42)
    parser.add_argument('--seq_length', type=int, default=10)
    parser.add_argument('--max_init_points', default=10, type=int, help='')
    parser.add_argument('--block_size', type=int, default=1)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--opt_pose', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--tensorboard', default=False, type=lambda x: x.lower() in ['true', '1'])

    parser.add_argument('--scene_name', type=str, default='')
    parser.add_argument('--object_combination', type=str, default='')
    parser.add_argument('--obj_name', type=str, default='scene')
    parser.add_argument('--interaction', type=str, default='default')
    parser.add_argument('--load_feature', type=int, default=0)
    parser.add_argument('--export_dir', type=str, default='POSA_IPoser')
    parser.add_argument('--vposer_w', type=float, default=0)

    args = parser.parse_args()
    args_dict = vars(args)

    return args, args_dict
