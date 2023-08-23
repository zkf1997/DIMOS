"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""


import os, sys, glob
from typing import Tuple
from typing import Union
import pdb
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import logging
import datetime
import smplx
import json
from scipy.spatial.transform import Rotation as R
from exp_GAMMAPrimitive.utils.config_env import get_body_marker_path, get_body_model_path


"""
===============================================================================
basic network modules
===============================================================================
"""

def get_logger(log_dir, mode='train'):
    logger = logging.getLogger(log_dir)
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(log_dir, '{}_{}.log'.format(mode, ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def get_scheduler(optimizer, policy, num_epochs_fix=None, num_epochs=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - num_epochs_fix) / float(num_epochs - num_epochs_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    else:
        return NotImplementedError('scheduler with {} is not implemented'.format(policy))
    return scheduler



def get_body_model(body_model_path, model_type, gender, batch_size, device='cpu'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    # body_model_path = '/home/yzhang/body_models/VPoser'
    body_model = smplx.create(body_model_path, model_type=model_type,
                                    gender=gender, ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=batch_size
                                    ).to(device)
    return body_model



def calc_vec2path(x, a, b):
    n_ab = (b-a)/np.linalg.norm(b-a)
    if x.ndim==1:
        ax = x-a
        ax_proj = np.dot(ax, n_ab)*n_ab
        vec = ax_proj - ax
        return vec
    elif x.ndim==2: #[batch, dim]
        ax = x-np.tile(a, [x.shape[0],1])
        ax_proj = np.einsum('i,pi,j->pj', n_ab, ax, n_ab)
        vec = ax_proj - ax
        return vec


def soft_clip(x, minVal, maxVal):
    return minVal+(maxVal-minVal)*torch.sigmoid(x)



class RotConverter(nn.Module):
    '''
    - this class is modified from smplx/vposer
    - all functions only support data_in with [N, num_joints, D].
        -- N can be n_batch, or n_batch*n_time
    '''
    def __init__(self):
        super(RotConverter, self).__init__()

    @staticmethod
    def cont2rotmat(data_in):
        '''
        :data_in Nxnum_jointsx6
        :return: pose_matrot: Nxnum_jointsx3x3
        '''
        reshaped_input = data_in.contiguous().view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)#[b,3,3]

    @staticmethod
    def aa2cont(data_in):
        '''
        :data_in Nxnum_jointsx3
        :return: pose_matrot: Nxnum_jointsx6
        '''
        batch_size = data_in.shape[0]
        pose_body_6d = tgm.angle_axis_to_rotation_matrix(data_in.reshape(-1, 3))[:, :3, :2].contiguous().view(batch_size, -1, 6)
        return pose_body_6d


    @staticmethod
    def cont2aa(data_in):
        '''
        :data_in Nxnum_jointsx6
        :return: pose_matrot: Nxnum_jointsx3
        '''
        batch_size = data_in.shape[0]
        x_matrot_9d = RotConverter.cont2rotmat(data_in).view(batch_size,-1,9)
        x_aa = RotConverter.rotmat2aa(x_matrot_9d).contiguous().view(batch_size, -1, 3)
        return x_aa

    @staticmethod
    def rotmat2aa(data_in):
        '''
        :data_in data_in: Nxnum_jointsx9
        :return: Nxnum_jointsx3
        '''
        homogen_matrot = F.pad(data_in.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2rotmat(data_in):
        '''
        :data_in Nxnum_jointsx3
        :return: pose_matrot: Nxnum_jointsx9
        '''
        batch_size = data_in.shape[0]
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(data_in.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, -1, 9)

        return pose_body_matrot


    @staticmethod
    def vposer2rotmat6d(vposer, data_in):
        '''
        :data_in Bx32
        :return: pose_matrot: Nxnum_jointsx9
        '''
        batch_size = data_in.shape[0]
        x_pred_pose_9d = vposer.decode(data_in, output_type='matrot').view(-1,1,21,3,3)#[-1, 1, n_joints, 3,3]
        x_pred_pose_6d = x_pred_pose_9d[:,:,:,:,:2].reshape([-1, data_in.shape[-1], 21*6]).permute([0,2,1])

        return x_pred_pose_6d

    @staticmethod
    def vposer2rotmat(vposer, x):
        x_pred_pose_9d = vposer.decode(x.permute([0,2,1]).reshape([-1, 32]),
                                            output_type='matrot')#[-1, 1, n_joints, 9]
        x_pred_pose_9d = x_pred_pose_9d.reshape([-1, x.shape[-1], 21*9]).permute([0,2,1])

        return x_pred_pose_9d



class CanonicalCoordinateExtractor:
    """Summary of class here.

    When the model runs recursively, we need to reset the coordinate and perform canonicalization on the fly.
    This class provides such functionality.
    When specifying the joint locations of the motion primitive, it produces a new canonical coordinate, according to
    the reference frame.
    Both numpy and torch are supported.

    Attributes:
        device: torch.device, to specify the device when the input is torch.tensor
    """

    def __init__(self, device=torch.device('cuda:0')):
        self.device = device

    def get_new_coordinate_torch(self,
                            jts: torch.Tensor
        ) -> Tuple[torch.Tensor]:
        x_axis = jts[:,2,:] - jts[:,1,:] #[b,3]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis,dim=-1, keepdim=True)
        z_axis = torch.FloatTensor([[0,0,1]]).to(self.device).repeat(x_axis.shape[0], 1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = y_axis/torch.norm(y_axis,dim=-1, keepdim=True)
        new_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=-1) #[b,3,3]
        new_transl = jts[:, :1] #[b,1,3]
        return new_rotmat, new_transl


    def get_new_coordinate_numpy(self,
                            jts: np.ndarray
        ) -> Tuple[np.ndarray]: #jts=[b,J,3]

        x_axis = jts[:,2,:] - jts[:,1,:] #[b,3]
        x_axis[:, -1] = 0
        x_axis = x_axis / np.linalg.norm(x_axis,axis=-1, keepdims=True)
        z_axis = np.tile(np.array([[0,0,1]]), (x_axis.shape[0], 1))
        y_axis = np.cross(z_axis, x_axis, axis=-1)
        y_axis = y_axis/np.linalg.norm(y_axis,axis=-1, keepdims=True)
        new_rotmat = np.stack([x_axis, y_axis, z_axis], axis=-1) #[b,3,3]
        new_transl = jts[:, :1] #[b,1,3]

        return new_rotmat, new_transl


    def get_new_coordinate(self,
                        jts: Union[np.ndarray, torch.Tensor],
                        in_numpy: bool=False,
        ) -> Tuple[Union[np.ndarray, torch.Tensor]] :
        """get a new canonical coordinate located at a specific frame

        Args:
            jts: the input joint locations, jts.shape=[b,J,3]
            in_numpy: if True, everything is calculated with numpy, else calculated with pytorch.

        Returns:
            A list of tensors, [new_rotmat, new_transl]
            new_rotmat: rotmat according to the old coordinate. in the shape [b,3,3]
            new_transl: translation according to the old coordinate. in the shape [b,1,3]

        Raises:
            None
        """
        if in_numpy:
            new_rotmat, new_transl = self.get_new_coordinate_numpy(jts)
        else:
            new_rotmat, new_transl = self.get_new_coordinate_torch(jts)

        return new_rotmat, new_transl



class SMPLXParser:
    """operations about the smplx model

    We frequently use the smplx model to extract the markers/joints, and perform relevant transform.
    So we make this smplxparser to provide an unified interface.

    Attributes:
        an example input:
        pconfig_mp = {
            'n_batch':10,
            'device': device,
            'marker_placement': 'ssm2_67'
            }
    """
    def __init__(self, config):
        for key, val in config.items():
            setattr(self, key, val)

        '''set body models'''
        body_model_path = get_body_model_path()
        self.bm_male = smplx.create(body_model_path, model_type='smplx',
                                    gender='male', ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=self.n_batch
                                    )
        self.bm_female = smplx.create(body_model_path, model_type='smplx',
                                    gender='female', ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=self.n_batch
                                    )
        self.bm_male.to(self.device)
        self.bm_female.to(self.device)
        self.bm_male.eval()
        self.bm_female.eval()

        self.coord_extractor = CanonicalCoordinateExtractor(self.device)

        '''set marker'''
        marker_path = get_body_marker_path()
        if config['marker_placement'] == 'cmu_41':
            with open(marker_path+'/CMU.json') as f:
                self.marker = list(json.load(f)['markersets'][0]['indices'].values())
        elif config['marker_placement'] == 'ssm2_67':
            with open(marker_path+'/SSM2.json') as f:
                self.marker = list(json.load(f)['markersets'][0]['indices'].values())



    def forward_smplx(self,
                    betas: Union[torch.Tensor, np.ndarray],
                    gender: str,
                    xb: Union[torch.Tensor, np.ndarray],
                    to_numpy: bool = True,
                    output_type: str = 'markers'
        ) -> Union[torch.Tensor, np.ndarray]:
        """forward kinematics for smplx.

        Args:
            betas: the body shape, can be (10,) or (1, 10)
            gender: str, 'male' or 'female'
            xb: the compact vector of body params, (b, 93)
            to_numpy: if True, input and output are in numpy; else everything is in pytorch
            output_types: string in ['markers', 'joints'].

        Returns:
            output: np.array or torch.tensor, either joints [b, 22, 3] or markers [b, n_markers, 3]

        Raises:
            NotImplementedError rises if output_type is neither 'markers' nor 'joints'
        """
        bm = None
        if gender == 'male':
            bm = self.bm_male
        elif gender == 'female':
            bm = self.bm_female
        bparam = {}
        bparam['transl'] = xb[:,:3]
        bparam['global_orient'] = xb[:,3:6]
        if to_numpy:
            bparam['betas'] = np.tile(betas, (xb.shape[0],1))
        else:
            bparam['betas'] = betas.repeat(xb.shape[0],1)
        bparam['body_pose'] = xb[:,6:69]
        bparam['left_hand_pose'] = xb[:,69:81]
        bparam['right_hand_pose'] = xb[:,81:]
        for key in bparam:
            if type(bparam[key]) == np.ndarray:
                bparam[key] = torch.cuda.FloatTensor(bparam[key])
            elif type(bparam[key]) == torch.Tensor and bparam[key].device!=self.device:
                bparam[key] = bparam[key].to(self.device)
            else:
                bparam[key] = bparam[key]
        smplx_out = bm(return_verts=True, **bparam)
        if output_type == 'markers':
            output = smplx_out.vertices[:,self.marker,:] #[b,n_markers, 3]
        elif output_type == 'joints':
            output = smplx_out.joints[:,:22] #[b,jts, 3]
        elif output_type == 'vertices':
            output = smplx_out.vertices[:, :, :] #[b,jts, 3]
        else:
            raise NotImplementedError('other output types are not supported')
        if to_numpy:
            output = output.detach().cpu().numpy()

        return output


    def get_jts(self,
                betas: Union[torch.Tensor, np.ndarray],
                gender: str,
                xb: Union[torch.Tensor, np.ndarray],
                to_numpy: bool = True,
        ) -> Union[torch.Tensor, np.ndarray]:
        """get a batch of joint locations from the smplx_params vector xb

        Args:
            betas: the body shape, can be (10,) or (1, 10)
            gender: str, 'male' or 'female'
            xb: the compact vector of body params, (b, 93)
            to_numpy: if True, input and output are in numpy; else everything is in pytorch

        Returns:
            batch_jts: np.array or torch.tensor, in the shape of [b, 22, 3]

        Raises:
            None
        """
        func_args = locals()
        func_args.pop('self')
        func_args['output_type']='joints'
        batch_jts = self.forward_smplx(**func_args)
        return batch_jts

    def get_markers(self,
                betas: Union[torch.Tensor, np.ndarray],
                gender: str,
                xb: Union[torch.Tensor, np.ndarray],
                to_numpy: bool = True,
        ) -> Union[torch.Tensor, np.ndarray]:
        """get a batch of joint locations from the smplx_params vector xb

        Args:
            betas: the body shape, can be (10,) or (1, 10)
            gender: str, 'male' or 'female'
            xb: the compact vector of body params, (b, 93)
            to_numpy: if True, input and output are in numpy; else everything is in pytorch

        Returns:
            batch_markers: np.array or torch.tensor, in the shape of [b, n_markers, 3]

        Raises:
            None
        """
        func_args = locals()
        func_args.pop('self')
        func_args['output_type']='markers'
        batch_markers = self.forward_smplx(**func_args)
        return batch_markers

    def get_new_coordinate(self,
                betas: Union[torch.Tensor, np.ndarray],
                gender: str,
                xb: Union[torch.Tensor, np.ndarray],
                to_numpy: bool = True,
        ) -> Tuple[Union[torch.Tensor, np.ndarray]]:
        """get a batch canonical coordinates for recursive use.
        Note that this function extends self.coord_extractor.get_new_coordinate, and assumes the first frame is the reference frame.
        Therefore, it is required to select the motion seed before using this function.

        Args:
            betas: the body shape, can be (10,) or (1, 10)
            gender: str, 'male' or 'female'
            xb: the compact vector of body params, (b, 93)
            to_numpy: if True, input and output are in numpy; else everything is in pytorch

        Returns:
            [new_rotmat, new_transl]: np.array or torch.tensor, in the shape of [b, 3, 3] and [b, 1, 3]

        Raises:
            None
        """
        joints = self.get_jts(betas, gender, xb, to_numpy) #[b,J,3]
        new_rotmat, new_transl = self.coord_extractor.get_new_coordinate(joints,
                                                    in_numpy=to_numpy)

        return new_rotmat, new_transl



    def calc_calibrate_offset(self,
                            bm,
                            betas: Union[torch.Tensor, np.ndarray],
                            body_pose: Union[torch.Tensor, np.ndarray],
                            to_numpy: str=True,
        ) -> Union[torch.Tensor, np.ndarray]:
        """compensate the offset when transforming the smplx body and the body parameterss
        When performing the global transformation, first getting the body and then transforming will lead to a different result
        from first transforming the transl/global_orient and then get the body model. The reason is the global rotation is about
        the pelvis, whereas the global translation is about the kinematic tree root, which is not the body pelvis.

        Args:
            bm: the smplx body model, which is pre-defined
            betas: np.array or torch.tensor, in shape (10, ) or (1, 10)
            body_pose: the body pose in axis-angle, in shape (b, 63)
            to_numpy: if True, input and output are in numpy; else everything is in pytorch

        Returns:
            delta_T: the compensation offset between root and body pelvis, in shape (b, 3)

        Raises:
            None
        """
        n_batches = body_pose.shape[0]
        bodyconfig = {}
        if to_numpy:
            bodyconfig['body_pose'] = torch.FloatTensor(body_pose).to(self.device)
            bodyconfig['betas'] = torch.FloatTensor(betas).repeat(n_batches,1).to(self.device)
            bodyconfig['transl'] = torch.zeros([n_batches,3], dtype=torch.float32).to(self.device)
            bodyconfig['global_orient'] = torch.zeros([n_batches,3], dtype=torch.float32).to(self.device)
        else:
            bodyconfig['body_pose'] = body_pose
            bodyconfig['betas'] = betas.repeat(n_batches,1)
            bodyconfig['transl'] = torch.zeros([n_batches,3], dtype=torch.float32,device=self.device)
            bodyconfig['global_orient'] = torch.zeros([n_batches,3], dtype=torch.float32, device=self.device)
        smplx_out = bm(return_verts=True, **bodyconfig)
        delta_T = smplx_out.joints[:,0,:] # we output all pelvis locations
        if to_numpy:
            delta_T = delta_T.detach().cpu().numpy() #[b, 3]

        return delta_T


    def update_transl_glorot(self,
                            transf_rotmat: Union[torch.Tensor, np.ndarray],
                            transf_transl: Union[torch.Tensor, np.ndarray],
                            betas: Union[torch.Tensor, np.ndarray],
                            gender: str,
                            xb: Union[torch.Tensor, np.ndarray],
                            to_numpy: bool=True,
                            inplace=True):
        """update the (global) body parameters when performing global transform
        When performing the global transformation, first getting the body and then transforming will lead to a different result
        from first transforming the transl/global_orient and then get the body model. The reason is the global rotation is about
        the pelvis, whereas the global translation is about the kinematic tree root, which is not the body pelvis.

        Args:
            transf_rotmat: torch.tensor or np.array in shape (b, 3, 3)
            transf_transl: torch.tensor or np.array in shape (b, 1, 3)
            betas: np.array or torch.tensor, in shape (10, ) or (1, 10)
            gender: str, in ['male', 'female']
            xb: torch.tensor or np.array in shape (b, 93), the smplx body parameters
            to_numpy: if True, input and output are in numpy; else everything is in pytorch

        Returns:
            xb: the body parameters containing the updated transl and global_orient, (b, 93)

        Raises:
            None
        """
        bm = None
        if gender == 'male':
            bm = self.bm_male
        elif gender == 'female':
            bm = self.bm_female

        delta_T = self.calc_calibrate_offset(bm, betas,
                                        xb[:,6:69], to_numpy)
        transl = xb[:,:3]
        glorot = xb[:,3:6]
        if to_numpy:
            ### get new global_orient
            global_ori = R.from_rotvec(glorot).as_matrix() # to [t,3,3] rotation mat
            global_ori_new = np.einsum('bij,bjk->bik', transf_rotmat.transpose(0,2,1), global_ori)
            glorot = R.from_matrix(global_ori_new).as_rotvec()
            ### get new transl
            transl = np.einsum('bij,bj->bi', transf_rotmat.transpose(0,2,1), transl+delta_T-transf_transl[:,0])-delta_T
            if inplace:
                xb[:,:3] = transl
                xb[:,3:6] = glorot
            else:
                xb = np.concatenate([transl, glorot, xb[:, 6:]], axis=1)
        else:
            global_ori = tgm.angle_axis_to_rotation_matrix(glorot)[:,:3,:3]
            global_ori_new = torch.einsum('bij,bjk->bik', transf_rotmat.permute(0,2,1), global_ori)
            glorot = F.pad(global_ori_new, [0,1])
            glorot = tgm.rotation_matrix_to_angle_axis(glorot).view(-1, 3).contiguous()
            transl = torch.einsum('bij,bj->bi', transf_rotmat.permute(0,2,1), transl+delta_T-transf_transl[:,0])-delta_T
            if inplace:
                xb[:,:3] = transl
                xb[:,3:6] = glorot
            else:
                xb = torch.cat([transl, glorot, xb[:, 6:]], dim=1)

        return xb








"""
===============================================================================
basic network modules
===============================================================================
"""



class MLP(nn.Module):
    def __init__(self, in_dim,
                h_dims=[128,128], activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU()
        self.out_dim = h_dims[-1]
        self.layers = nn.ModuleList()
        in_dim_ = in_dim
        for h_dim in h_dims:
            self.layers.append(nn.Linear(in_dim_, h_dim))
            in_dim_ = h_dim

    def forward(self, x):
        for fc in self.layers:
            x = self.activation(fc(x))
        return x



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()


    def _sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, *args, **kwargs):
        pass

    def decode(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass

    def sample_prior(self, *args, **kwargs): #[t, b, d]
        pass





class TrainOP:
    def __init__(self, modelconfig, lossconfig, trainconfig):
        self.dtype = torch.float32
        gpu_index = trainconfig.get('gpu_index',0)
        self.device = torch.device('cuda',
                index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.modelconfig = modelconfig
        self.lossconfig = lossconfig
        self.trainconfig = trainconfig
        self.logger = get_logger(self.trainconfig['log_dir'])
        self.writer = SummaryWriter(log_dir=self.trainconfig['log_dir'])

    def build_model(self):
        pass

    def calc_loss(self):
        pass

    def train(self):
        pass


class TestOP:
    def __init__(self, modelconfig, testconfig, *args):
        self.dtype = torch.float32
        gpu_index = testconfig['gpu_index']
        if gpu_index >= 0:
            self.device = torch.device('cuda',
                    index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        self.modelconfig = modelconfig
        self.testconfig = testconfig

        if not os.path.exists(self.testconfig['ckpt_dir']):
            print('[ERROR]: no model was trained. Program terminates.')
            sys.exit(-1)

    def build_model(self):
        pass

    def visualize(self):
        pass

    def evaluation(self):
        pass

