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

from __future__ import division
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import openmesh as om
from sklearn.neighbors import KDTree
import trimesh


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i, i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat


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


class ds_us(nn.Module):
    """docstring for ds_us."""

    def __init__(self, M):
        super(ds_us, self).__init__()
        self.M = M

    def forward(self, x):
        """Upsample/downsample mesh. X: B*C*N"""
        out = []
        x = x.transpose(1, 2)
        for i in range(x.shape[0]):
            y = x[i]
            y = spmm(self.M, y)
            out.append(y)
        x = torch.stack(out, dim=0)
        return x.transpose(2, 1)


def scipy_to_pytorch(x):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    x = scipy.sparse.coo_matrix(x)
    i = torch.LongTensor(np.array([x.row, x.col]))
    v = torch.FloatTensor(x.data)
    return torch.sparse.FloatTensor(i, v, x.shape)


def get_graph_params(ds_us_dir, layer=1, use_cuda=False, **kwargs):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    device = torch.device("cuda" if use_cuda else "cpu")

    if layer > 0:
        A = scipy.sparse.load_npz(osp.join(ds_us_dir, 'A_{}.npz'.format(layer)))
        D = scipy.sparse.load_npz(osp.join(ds_us_dir, 'D_{}.npz'.format(layer)))
        U = scipy.sparse.load_npz(osp.join(ds_us_dir, 'U_{}.npz'.format(layer)))
    else:
        A = np.load(osp.join(ds_us_dir, 'A_{}.npz'.format(layer)), allow_pickle=True, encoding='latin1')['A'].all()
        # print(type(A))
        D = scipy.sparse.csr_matrix([])  # null
        U = scipy.sparse.csr_matrix([])

    D = scipy_to_pytorch(D).to(device)
    U = scipy_to_pytorch(U).to(device)
    A = adjmat_sparse(A).to(device)
    return A, U, D


def _next_ring(mesh, last_ring, other):
    res = []

    def is_new_vertex(idx):
        return (idx not in last_ring and idx not in other and idx not in res)

    for vh1 in last_ring:
        vh1 = om.VertexHandle(vh1)
        after_last_ring = False
        for vh2 in mesh.vv(vh1):
            if after_last_ring:
                if is_new_vertex(vh2.idx()):
                    res.append(vh2.idx())
            if vh2.idx() in last_ring:
                after_last_ring = True
        for vh2 in mesh.vv(vh1):
            if vh2.idx() in last_ring:
                break
            if is_new_vertex(vh2.idx()):
                res.append(vh2.idx())
    return res


def extract_spirals(mesh, seq_length, dilation=1):
    # output: spirals.size() = [N, seq_length]
    spirals = []
    for vh0 in mesh.vertices():
        reference_one_ring = []
        for vh1 in mesh.vv(vh0):
            reference_one_ring.append(vh1.idx())
        spiral = [vh0.idx()]
        one_ring = list(reference_one_ring)
        last_ring = one_ring
        next_ring = _next_ring(mesh, last_ring, spiral)
        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(mesh, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            kdt = KDTree(mesh.points(), metric='euclidean')
            spiral = kdt.query(np.expand_dims(mesh.points()[spiral[0]],
                                              axis=0),
                               k=seq_length * dilation,
                               return_distance=False).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(spiral[:seq_length * dilation][::dilation])
    return spirals


def create_spiral(template_path, seq_length, dilation=1):
    m = trimesh.load(template_path)
    m_om = om.TriMesh(m.vertices, m.faces)
    spirals = extract_spirals(m_om, seq_length, dilation)
    return spirals