import numpy as np
import heapq
import copy
import pdb
from scipy.spatial.transform import Rotation as R



class MinHeap(object):
    def __init__(self):
        self.data = []

    def push(self, node):
        heapq.heappush(self.data, node)

    def pop(self):
        try:
            node = heapq.heappop(self.data)
        except IndexError as e:
            node=None
        return node

    def clear(self):
        self.data.clear()

    def is_empty(self):
        return True if len(self.data)==0 else False

    def deepcopy(self):
        return copy.deepcopy(self)

    def len(self):
        return len(self.data)

import torch
import torch.nn.functional as F
def calc_sdf(vertices, sdf_dict):
    # sdf_dim = sdf_dict['dim']
    sdf_grids = torch.from_numpy(sdf_dict['grid'])
    sdf_grids = sdf_grids.squeeze().unsqueeze(0).unsqueeze(0).to(device=vertices.device, dtype=torch.float32) # 1x1xDxDxD
    sdf_extent = torch.tensor(sdf_dict['extent']).to(device=vertices.device, dtype=torch.float32)
    sdf_centroid = torch.tensor(sdf_dict['centroid']).reshape(1, 1, 3).to(device=vertices.device, dtype=torch.float32)

    batch_size, num_vertices, _ = vertices.shape
    vertices = vertices.reshape(1, -1, 3)  # [1, B*V, 3]
    vertices = (vertices - sdf_centroid) / sdf_extent * 2  # convert to [-1, 1]
    sdf_values = F.grid_sample(sdf_grids,
                                   vertices[:, :, [2, 1, 0]].view(1, batch_size * num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                   padding_mode='border',
                                   align_corners=True
                                   # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                   )
    '''
    # illustration of grid_sample dimension order, assume first dimension to be innermost
    # import torch
    # import torch.nn.functional as F
    # import numpy as np
    # sz = 5
    # input_arr = torch.from_numpy(np.arange(sz * sz).reshape(1, 1, sz, sz)).float()
    # indices = torch.from_numpy(np.array([-1, -1, -0.5, -0.5, 0, 0, 0.5, 0.5, 1, 1, -1, 0.5, 0.5, -1]).reshape(1, 1, 7, 2)).float()
    # out = F.grid_sample(input_arr, indices, align_corners=True)
    # print(input_arr)
    # print(out)
    '''
    return sdf_values.reshape(batch_size, num_vertices)


'''
in this script, we implement a tree of motion primitives.
MotionPrimitiveTree = MPT
From starting pose, we first generate different motion primitives as the roots, e.g. MP1, MP2, or MP3...

|-- MP1
    |-- MP1,1
    |-- MP1,2
    ...
|-- MP2
    |-- MP2,1
    |-- MP2,2
    ...
|-- MP3
    |-- MP3,1
    |-- MP3,2
    ...
'''

class MPTNode(object):
    def __init__(self, gender, betas, transf_rotmat, transf_transl, pelvis_loc, joints,
                markers, markers_proj, smplx_params, mp_latent=None, mp_type='2-frame', timestamp=-1,
                curr_target_wpath=None):
        '''
        A MPT node contains (data, parent, children list, quality)
        '''

        self.data = {}
        self.data['gender'] = gender
        self.data['betas'] = betas
        self.data['transf_rotmat'] = transf_rotmat #amass2world
        self.data['transf_transl'] = transf_transl #amass2world
        self.data['mp_type'] = mp_type
        self.data['markers'] = markers
        self.data['markers_proj'] = markers_proj
        self.data['pelvis_loc'] = pelvis_loc
        self.data['joints'] = joints
        self.data['smplx_params'] = smplx_params
        self.data['mp_latent'] = mp_latent
        self.data['timestamp'] = timestamp
        self.data['curr_target_wpath']= curr_target_wpath #(pt_idx, pt_loc)

        self.parent = None
        self.children = []
        self.quality = 0
        self.to_target_dist = 1e6
        self.motion_naturalness = 1e6
        self.q_me = 0

    def __lt__(self, other):
        '''
        note that this definition is to flip the order in the python heapq (a min heap)
        '''
        return self.quality < other.quality


    def add_child(self, child):
        '''
        child - MPTNode
        '''
        if child.quality != 0:
            child.parent = self
            self.children.append(child)
        # else:
        #     # print('[INFO searchop] cannot add low-quality children. Do nothing.')
        #     pass

    def set_parent(self, parent):
        '''
        parent - MPTNode
        '''
        if self.quality != 0:
            self.parent = parent
            return True
        else:
            return False




    def evaluate_quality_soft_contact_wpath(self,
                        terrian_rotmat=np.eye(3),
                        terrian_transl=np.zeros(3),
                        use_orient=True,
                        last_orient=None,  # [1, 2]
                        weight_target=0.1,
                        weight_ori=0.1,
                        wpath=None):
        '''
        - The evaluation is based on body ground contact, in the local coordinate of terrian
        - rotmat and transl of terrian is from its local to world
        - target_node is used for planning
        - start_node is also for planning, but mainly used in A* to calculate the path cost from start to current. Set to None
        '''
        # terrian_transl[-1] = wpath[0][-1]
        #----transform markers to the world coordinate
        Y_l = self.data['markers_proj'].reshape((-1, 67, 3)) #[t,p,3]
        Y_w = np.einsum('ij,tpj->tpi', self.data['transf_rotmat'][0], Y_l)+self.data['transf_transl']
        #----transform markers to the local terrian coordinate
        Y_wr = np.einsum('ij, tpj->tpi', terrian_rotmat.T, Y_w-terrian_transl[None,None,...])

        #----select motion index of proper contact with the ground
        Y_wz = Y_wr[:,:,-1] #[t, P]
        Y_w_speed = np.linalg.norm(Y_w[1:]-Y_w[:-1], axis=-1)*40 #[t=9,P=67]

        '''evaluate contact soft'''
        self.dist2g = dist2gp = max(np.abs(Y_wz.min())-0.0, 0)
        self.dist2skat = dist2skat = max(np.abs(Y_w_speed.min())-0.075,0)
        q_contact = np.exp(-dist2gp) * np.exp(-dist2skat )

        '''evaluate the distance to the final target'''
        R0 = self.data['transf_rotmat']
        T0 = self.data['transf_transl']
        target_wpath = wpath[-1]
        target_wpath_l = np.einsum('ij,j->i', R0[0].T, target_wpath-T0[0,0])[:3]
        self.dist2target=dist2target=np.linalg.norm(target_wpath_l-self.data['pelvis_loc'][-1,:3])
        q_2target = np.exp(-dist2target)

        '''evaluate facing orientation'''
        if use_orient:
            joints = self.data['joints']
            joints_end = joints[-1] #[p,3]
            x_axis = joints_end[2,:] - joints_end[1,:]
            x_axis[-1] = 0
            x_axis = x_axis / np.linalg.norm(x_axis,axis=-1,keepdims=True)
            z_axis = np.array([0,0,1])
            y_axis = np.cross(z_axis, x_axis)
            b_ori = y_axis[:2]

            if self.data['curr_target_wpath'][0] == len(wpath) - 1:  # last waypoint, should inverse the direction
                t_ori = last_orient.squeeze()
                # t_ori = t_ori / np.linalg.norm(t_ori, axis=-1, keepdims=True)
                dist2ori = 1 - np.einsum('i,i->', t_ori, b_ori)
            else:
                # t_ori = target_wpath_l[:2] - joints_end[0, :2]
                # t_ori = t_ori / np.linalg.norm(t_ori, axis=-1, keepdims=True)
                # dist2ori = 1 - np.einsum('i,i->', t_ori, b_ori)
                dist2ori = 0
        else:
            dist2ori = 0
        self.dist2ori = dist2ori

        curr_target_wpath = self.data['curr_target_wpath'][1]
        curr_target_wpath_l = np.einsum('ij,j->i', R0[0].T, curr_target_wpath-T0[0,0])[:3]
        self.dist2target_curr=dist2target_curr=np.linalg.norm(curr_target_wpath_l-self.data['pelvis_loc'][-1,:3])
        # self.quality = dist2gp+dist2skat + 0.1*dist2ori + 0.1*dist2target_curr
        goals_left = len(wpath) - self.data['curr_target_wpath'][0]  # encourage to reach goals, without this human can go around last target but not reaching, because switching to next target will suddenly increase dist2target
        # self.quality = dist2gp + dist2skat + dist2target_curr + goals_left * 10
        self.quality = dist2gp + dist2skat + weight_ori*dist2ori + weight_target * dist2target_curr + goals_left * 100


    def evaluate_quality_collision(self,
                        terrian_rotmat=np.eye(3),
                        terrian_transl=np.zeros(3),
                        wpath=None,
                        floor_height=0,
                        scene=None,
                        obj_sdf=None,
                        obj_transform=None,
                        obj_points=None,
                        coap_model=None,
                        collision_mode=1,
                        weight_target=0.1,
                        weight_pene=1,
                        weight_ori=0.1,
                        ):
        '''
        - The evaluation is based on body ground contact, in the local coordinate of terrian
        - rotmat and transl of terrian is from its local to world
        - target_node is used for planning
        - start_node is also for planning, but mainly used in A* to calculate the path cost from start to current. Set to None
        '''
        # terrian_transl[-1] = wpath[0][-1]
        #----transform markers to the world coordinate
        Y_l = self.data['markers_proj'].reshape((-1, 67, 3)) #[t,p,3]
        Y_w = np.einsum('ij,tpj->tpi', self.data['transf_rotmat'][0], Y_l)+self.data['transf_transl']
        #----transform markers to the local terrian coordinate
        Y_wr = np.einsum('ij, tpj->tpi', terrian_rotmat.T, Y_w-terrian_transl[None,None,...])

        #----select motion index of proper contact with the ground
        Y_wz = Y_wr[:,:,-1] #[t, P]
        Y_w_speed = np.linalg.norm(Y_w[1:]-Y_w[:-1], axis=-1)*40 #[t=9,P=67]

        '''evaluate contact soft'''
        self.dist2g = dist2gp = max(np.abs(Y_wz.min())-0.05, 0)
        self.dist2skat = dist2skat = max(np.abs(Y_w_speed.min())-0.075,0)
        q_contact = np.exp(-dist2gp) * np.exp(-dist2skat )

        '''evaluate the distance to the final target'''
        R0 = self.data['transf_rotmat']
        T0 = self.data['transf_transl']
        target_wpath = wpath[-1]
        target_wpath_l = np.einsum('ij,j->i', R0[0].T, target_wpath-T0[0,0])[:3]
        self.dist2target=dist2target=np.linalg.norm(target_wpath_l-self.data['pelvis_loc'][-1,:3])
        q_2target = np.exp(-dist2target)

        curr_target_wpath = self.data['curr_target_wpath'][1]
        curr_target_wpath_l = np.einsum('ij,j->i', R0[0].T, curr_target_wpath - T0[0, 0])[:3]
        self.dist2target_curr = dist2target_curr = np.linalg.norm(curr_target_wpath_l - self.data['pelvis_loc'][-1, :3])

        '''evaluate facing orientation'''
        # if self.data['curr_target_wpath'][0] < len(wpath) - 1:
        if False:
            joints = self.data['joints']
            joints_end = joints[-1] #[p,3]
            x_axis = joints_end[2,:] - joints_end[1,:]
            x_axis[-1] = 0
            x_axis = x_axis / np.linalg.norm(x_axis,axis=-1,keepdims=True)
            z_axis = np.array([0,0,1])
            y_axis = np.cross(z_axis, x_axis)
            b_ori = y_axis[:2]
            t_ori = curr_target_wpath_l[:2]-joints_end[0,:2]
            t_ori = t_ori/np.linalg.norm(t_ori, axis=-1, keepdims=True)
            self.dist2ori_curr = dist2ori_curr = 1-np.einsum('i,i->', t_ori, b_ori)
        else:
            from scipy.spatial.transform import Rotation as R
            target_orient = R.from_rotvec(scene['wpath_orients'][self.data['curr_target_wpath'][0] - 1]).as_matrix()
            target_orient = np.dot(self.data['transf_rotmat'][0].T, target_orient)
            # print(self.data['smplx_params'].shape)
            body_orient = R.from_rotvec(self.data['smplx_params'][0, -1, 3:6]).as_matrix()
            # theta = np.arccos((np.trace(np.dot(body_orient, target_orient.T)) - 1) / 2)
            self.dist2ori_curr = dist2ori_curr = 1 - ((np.trace(np.dot(body_orient, target_orient.T)) - 1) / 2)

        # penetration penalty
        if collision_mode == 1:
            # markers = np.einsum('ij,tpj->tpi', obj_transform[:3, :3].T, Y_w - obj_transform[:3, 3])  # [T, P, 3], transform to object space
            markers = Y_w
            markers = torch.tensor(markers, dtype=torch.float32, device='cuda')
            sdf_values = calc_sdf(markers, obj_sdf)
            self.pene_value = pene_value = 0.0 if sdf_values.lt(0.0).sum().item() < 1 else torch.mean(sdf_values[sdf_values < 0].abs()).cpu().numpy()
        elif collision_mode == 2:
            with torch.no_grad():
                bparams = torch.FloatTensor(self.data['smplx_params']).to('cuda')[0]  # [t, 93]
                betas = torch.FloatTensor(self.data['betas']).to('cuda').unsqueeze(0)
                scene_points = torch.FloatTensor(obj_points).to('cuda').unsqueeze(0)  # [1, p, 3]
                # print(bparams.shape, betas.shape, scene_points.shape)
                R0 = torch.FloatTensor(R0).to('cuda')
                T0 = torch.FloatTensor(T0).to('cuda')
                scene_points = torch.einsum('bij,bpj->bpi', R0.permute(0, 2, 1), scene_points - T0)
                smpl_output = coap_model(transl=bparams[:, :3], global_orient=bparams[:, 3:6],
                                         body_pose=bparams[:, 6:69],
                                         left_hand_pose=bparams[:, 69:81], right_hand_pose=bparams[:, 81:93],
                                         betas=betas.expand(bparams.shape[0], -1),
                                         return_verts=True, return_full_pose=True)
                loss_pene, _collision_mask = coap_model.coap.collision_loss(scene_points.expand(bparams.shape[0], -1, -1), smpl_output,
                                                                            ret_collision_mask=True)
            self.pene_value = pene_value = loss_pene.sum().item()
        else:
            self.pene_value = pene_value = 0.0

        # self.quality = dist2gp+dist2skat + 0.1*dist2ori + 0.1*dist2target_curr
        # print(dist2gp, dist2skat, dist2target_curr, pene_value)
        goals_left = len(wpath) - self.data['curr_target_wpath'][0]  # encourage to reach goals, without this human can go around last target but not reaching, because switching to next target will suddenly increase dist2target
        self.quality = dist2gp + dist2skat + weight_target * dist2target_curr + weight_ori * dist2ori_curr + weight_pene * pene_value + goals_left * 10


    def evaluate_quality_hard_contact_wpath(self,
                        terrian_rotmat=np.eye(3),
                        terrian_transl=np.zeros(3),
                        wpath=None):
        '''
        - The evaluation is based on body ground contact, in the local coordinate of terrian
        - rotmat and transl of terrian is from its local to world
        - target_node is used for planning
        - start_node is also for planning, but mainly used in A* to calculate the path cost from start to current. Set to None
        '''
        # terrian_transl[-1] = wpath[0][-1]
        #----transform markers to the world coordinate
        Y_l = self.data['markers_proj'].reshape((-1, 67, 3)) #[t,p,3]
        Y_w = np.einsum('ij,tpj->tpi', self.data['transf_rotmat'][0], Y_l)+self.data['transf_transl']
        #----transform markers to the local terrian coordinate
        Y_wr = np.einsum('ij, tpj->tpi', terrian_rotmat.T, Y_w-terrian_transl[None,None,...])

        #----select motion index of proper contact with the ground
        Y_wz = Y_wr[:,:,-1] #[t, P]
        Y_w_speed = np.linalg.norm(Y_w[1:]-Y_w[:-1], axis=-1)*40 #[t=9,P=67]

        '''evaluate contact soft'''
        # self.dist2g = dist2gp = max(np.abs(Y_wz.min())-0.05, 0)
        # self.dist2skat = dist2skat = max(np.abs(Y_w_speed.min())-0.075,0)
        if np.abs(Y_wz.min())<=0.05 and Y_w_speed.min()<=0.075:
            q_contact = 1
        else:
            q_contact = 0

        '''evaluate the distance to the final target'''
        R0 = self.data['transf_rotmat']
        T0 = self.data['transf_transl']
        target_wpath = wpath[-1]
        target_wpath_l = np.einsum('ij,j->i', R0[0].T, target_wpath-T0[0,0])[:2]
        self.dist2target=dist2target=np.linalg.norm(target_wpath_l-self.data['pelvis_loc'][-1,:2])
        q_2target = np.exp(-dist2target)

        '''evaluate facing orientation'''
        joints = self.data['joints']
        joints_end = joints[-1] #[p,3]
        x_axis = joints_end[2,:] - joints_end[1,:]
        x_axis[-1] = 0
        x_axis = x_axis / np.linalg.norm(x_axis,axis=-1,keepdims=True)
        z_axis = np.array([0,0,1])
        y_axis = np.cross(z_axis, x_axis)
        b_ori = y_axis[:2]
        t_ori = target_wpath_l[:2]-joints_end[0,:2]
        t_ori = t_ori/np.linalg.norm(t_ori, axis=-1, keepdims=True)
        dist2ori = 1-np.einsum('i,i->', t_ori, b_ori)

        curr_target_wpath = self.data['curr_target_wpath'][1]
        curr_target_wpath_l = np.einsum('ij,j->i', R0[0].T, curr_target_wpath-T0[0,0])[:2]
        self.dist2target_curr=dist2target_curr=np.linalg.norm(curr_target_wpath_l-self.data['pelvis_loc'][-1,:2])
        self.quality = q_contact*(0.1*dist2ori + 0.1*dist2target_curr)

class MPTNodeTorch(object):
    def __init__(self, gender, betas, transf_rotmat, transf_transl, pelvis_loc, joints,
                markers, markers_proj, smplx_params, mp_latent=None, mp_type='2-frame', timestamp=-1,
                curr_target_wpath=None):
        '''
        A MPT node contains (data, parent, children list, quality)
        '''

        self.data = {}
        self.data['gender'] = gender
        self.data['betas'] = betas
        self.data['transf_rotmat'] = transf_rotmat #amass2world
        self.data['transf_transl'] = transf_transl #amass2world
        self.data['mp_type'] = mp_type
        self.data['markers'] = markers
        self.data['markers_proj'] = markers_proj
        self.data['pelvis_loc'] = pelvis_loc
        self.data['joints'] = joints
        self.data['smplx_params'] = smplx_params
        self.data['mp_latent'] = mp_latent
        self.data['timestamp'] = timestamp
        self.data['curr_target_wpath']= curr_target_wpath #(pt_idx, pt_loc)

        self.parent = None
        self.children = []
        self.quality = 0
        self.to_target_dist = 1e6
        self.motion_naturalness = 1e6
        self.q_me = 0

    def __lt__(self, other):
        '''
        note that this definition is to flip the order in the python heapq (a min heap)
        '''
        return self.quality > other.quality


    def add_child(self, child):
        '''
        child - MPTNode
        '''
        if child.quality != 0:
            child.parent = self
            self.children.append(child)
        # else:
        #     # print('[INFO searchop] cannot add low-quality children. Do nothing.')
        #     pass

    def set_parent(self, parent):
        '''
        parent - MPTNode
        '''
        if self.quality != 0:
            self.parent = parent
            return True
        else:
            return False







