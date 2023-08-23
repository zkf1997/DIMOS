from __future__ import division

import sys
sys.path.append('..')
from configuration.config import *

import numpy as np
import scipy.sparse
import torch
import smplx
import trimesh
import os.path as osp
# import pytorch3d
# from pytorch3d.loss import mesh_laplacian_smoothing
# from pytorch3d.structures import Meshes

from interaction.posa_utils import get_graph_params
class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)

def downsample_vertices(D, last_valid_vertices):
    new_dim, last_dim = D.shape
    last_valid_vertices = set(last_valid_vertices)
    # print(last_valid_vertices)
    valid_vertices = set()
    for vertex in range(new_dim):
        last_vertices = torch.nonzero(D[vertex] > 0, as_tuple=False).flatten()
        # print(vertex, last_vertices)
        if set(last_vertices.tolist()).issubset(last_valid_vertices):
            valid_vertices.add(vertex)
    # print(valid_vertices)
    return list(valid_vertices)

# level of smplx meshes
class Mesh(object):
    """Mesh object that is used for handling certain graph operations."""
    def __init__(self, filename=mesh_operation_file,
                 num_downsampling=1, nsize=1, device=torch.device('cuda')):
        self.num_downsampling = num_downsampling
        self._A, self._U, self._D, self.meshes = [], [], [], []
        self._A.append(get_graph_params(mesh_ds_folder, 0, use_cuda=True))
        for level in range(5):
            A, U, D = get_graph_params(mesh_ds_folder, level + 1, use_cuda=True)
            self._A.append(A)
            self._U.append(U)
            self._D.append(D)
        self.num_vertices = []
        for level in range(6):
            m = trimesh.load(osp.join(str(mesh_ds_folder), 'mesh_{}.obj'.format(level)), process=False)
            self.meshes.append(m)
            self.num_vertices.append(m.vertices.shape[0])

        ref_vertices = torch.tensor(self.meshes[0].vertices, dtype=torch.float32)
        center = 0.5*(ref_vertices.max(dim=0)[0] + ref_vertices.min(dim=0)[0])[None]
        ref_vertices -= center
        ref_vertices /= ref_vertices.abs().max().item()

        self._ref_vertices = ref_vertices.to(device)
        self.faces = self.meshes[self.num_downsampling].faces
        self.device = device
        self.body_part_vertices = self.downsample_body_part_vertices(body_part_vertices)
        self.body_part_vertices_full = self.downsample_body_part_vertices(body_part_vertices_full)

    def downsample_body_part_vertices(self, body_part_vertices):
        """
        Get vertices list of each body part on downsampled meshes
        """
        downsample_level = self.num_downsampling
        last_body_part_vertices = body_part_vertices
        for level in range(downsample_level):
            D = self._D[level].cpu().to_dense()
            new_body_part_vertices = {}
            for part in last_body_part_vertices:
                new_body_part_vertices[part] = downsample_vertices(D, last_body_part_vertices[part])
            last_body_part_vertices = new_body_part_vertices

        return last_body_part_vertices

    @property
    def adjmat(self):
        """Return the graph adjacency matrix at the specified subsampling level."""
        return self._A[self.num_downsampling].float()

    @property
    def ref_vertices(self):
        """Return the template vertices at the specified subsampling level."""
        ref_vertices = torch.tensor(self.meshes[self.num_downsampling].vertices, dtype=torch.float32, device=self.device)
        center = 0.5 * (ref_vertices.max(dim=0)[0] + ref_vertices.min(dim=0)[0])[None]
        ref_vertices -= center
        ref_vertices /= ref_vertices.abs().max().item()
        return ref_vertices

    def ref_vertices_by_level(self, num_downsampling):
        """Return the template vertices at the specified subsampling level."""
        ref_vertices = torch.tensor(self.meshes[num_downsampling].vertices, dtype=torch.float32, device=self.device)
        center = 0.5 * (ref_vertices.max(dim=0)[0] + ref_vertices.min(dim=0)[0])[None]
        ref_vertices -= center
        ref_vertices /= ref_vertices.abs().max().item()
        return ref_vertices

    def downsample(self, x, n1=0, n2=None):
        """Downsample mesh."""
        if n2 is None:
            n2 = self.num_downsampling
        if x.ndimension() < 3:
            for i in range(n1, n2):
                # print(self._D[i].shape, x.shape)
                x = spmm(self._D[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in range(n1, n2):
                    y = spmm(self._D[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

    def upsample(self, x, n1=None, n2=0):
        """Upsample mesh."""
        if n1 is None:
            n1 = self.num_downsampling
        if x.ndimension() < 3:
            for i in reversed(range(n2, n1)):
                x = spmm(self._U[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in reversed(range(n2, n1)):
                    y = spmm(self._U[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

if __name__ == '__main__':
    import pylab
    color_map = pylab.get_cmap('hsv')
    for level in range(1, 6):
        mesh = Mesh(num_downsampling=level)
        print('faces:', mesh.faces.shape)
        for part in mesh.body_part_vertices:
            print(part, len(mesh.body_part_vertices[part]))
        print(mesh._A[-1])
        vertices = mesh._ref_vertices
        down_sampled = mesh.downsample(vertices)
        up_sampled = mesh.upsample(down_sampled)
        print(down_sampled.shape, up_sampled.shape)
        import trimesh
        colors = np.ones((mesh.num_vertices[mesh.num_downsampling], 4), dtype=np.float32) * 0.8
        # for idx, part in enumerate(mesh.body_part_vertices_full):
        #     colors[mesh.body_part_vertices_full[part], :] = color_map(idx / len(mesh.body_part_vertices_full))
        # for idx, part in enumerate(mesh.body_part_vertices):
        #     colors[mesh.body_part_vertices[part], :] = color_map(idx / len(mesh.body_part_vertices))
        downsampled = trimesh.Trimesh(
            vertices=down_sampled.cpu().numpy(),
            faces=mesh.faces,
            vertex_colors=colors
        )
        downsampled.show()
        colors = np.ones((mesh.num_vertices[0], 4), dtype=np.float32) * 0.8
        # for idx, part in enumerate(body_part_vertices_full):
        #     colors[body_part_vertices_full[part], :] = color_map(idx / len(body_part_vertices_full))
        # for idx, part in enumerate(body_part_vertices):
        #     colors[body_part_vertices[part], :] = color_map(idx / len(body_part_vertices))
        reconstructed = trimesh.Trimesh(
            vertices=up_sampled.cpu().numpy(),
            faces=mesh.meshes[0].faces,
            vertex_colors=colors
        )
        reconstructed.show()

