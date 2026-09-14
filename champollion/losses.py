#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
import torch
import torch.nn as nn
import torch.nn.functional as func
from champollion.utils import logs

log = logs.set_file_logger(__file__)

class BarlowTwinsLoss(nn.Module):

    def __init__(self, device, correlation='cross', lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device
        self.correlation = correlation

    def forward(self, z_a, z_b):
        # normalize repr. along the batch dimension
        # beware: normalization is not robust to batch of size 1
        # if it happens, it will return a nan loss
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)
        lbd = self.lambda_param / D

        if self.correlation=='cross':
            # cross-correlation matrix
            c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
            # loss
            c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
            # multiply off-diagonal elems of c_diff by lambda
            c_diff[~torch.eye(D, dtype=bool)] *= lbd
            loss_invariance = c_diff[torch.eye(D, dtype=bool)].sum()
            loss_redundancy = c_diff[~torch.eye(D, dtype=bool)].sum()
            loss = loss_invariance + loss_redundancy
        elif self.correlation=='auto':
            # auto-correlation matrix
            c1 = torch.mm(z_a_norm.T, z_a_norm) / N # DxD
            c2 = torch.mm(z_b_norm.T, z_b_norm) / N # DxD
            c = (c1.pow(2) + c2.pow(2)) / 2
            c[torch.eye(D, dtype=bool)]=0
            redundancy_loss = c.sum()
            # cross-correlation matrix
            c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
            # loss
            c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
            c_diff[~torch.eye(D, dtype=bool)]=0
            loss_invariance = c_diff.sum()
            loss_redundancy = self.lbd*redundancy_loss
            loss = loss_invariance + loss_redundancy
        else:
            raise ValueError("Wrong correlation specified in BarlowTwins\
                             config: use cross or auto.")

        return(loss, loss_invariance, loss_redundancy)


class NTXenLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, sim_zii, sim_zjj

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)
