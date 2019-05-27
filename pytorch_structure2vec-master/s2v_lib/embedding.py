from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from s2v_lib import S2VLIB
from pytorch_util import weights_init, gnn_spmm

class EmbedMeanField(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv = 3):
        super(EmbedMeanField, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

        self.max_lv = max_lv

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(latent_dim, output_dim)

        self.conv_params = nn.Linear(latent_dim, latent_dim)
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat): 
        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)
        if type(node_feat) is torch.cuda.FloatTensor:
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)

        h = self.mean_field(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp)
        
        return h

    def mean_field(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp):
        input_node_linear = self.w_n2l(node_feat)
        input_message = input_node_linear
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            input_message += e2npool_input
        input_potential = F.relu(input_message)

        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            n2npool = gnn_spmm(n2n_sp, cur_message_layer)
            node_linear = self.conv_params( n2npool )
            merged_linear = node_linear + input_message

            cur_message_layer = F.relu(merged_linear)
            lv += 1
        if self.output_dim > 0:
            out_linear = self.out_params(cur_message_layer)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = cur_message_layer
            
        y_potential = gnn_spmm(subg_sp, reluact_fp)

        return F.relu(y_potential)

class EmbedLoopyBP(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv = 3):
        super(EmbedLoopyBP, self).__init__()
        self.latent_dim = latent_dim
        self.max_lv = max_lv

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        self.out_params = nn.Linear(latent_dim, output_dim)

        self.conv_params = nn.Linear(latent_dim, latent_dim)
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat): 
        n2e_sp, e2e_sp, e2n_sp, subg_sp = S2VLIB.PrepareLoopyBP(graph_list)
        if type(node_feat) is torch.cuda.FloatTensor:
            n2e_sp = n2e_sp.cuda()
            e2e_sp = e2e_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
        node_feat = Variable(node_feat)
        edge_feat = Variable(edge_feat)
        n2e_sp = Variable(n2e_sp)
        e2e_sp = Variable(e2e_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)

        h = self.loopy_bp(node_feat, edge_feat, n2e_sp, e2e_sp, e2n_sp, subg_sp)
        
        return h

    def loopy_bp(self, node_feat, edge_feat, n2e_sp, e2e_sp, e2n_sp, subg_sp):
        input_node_linear = self.w_n2l(node_feat)
        input_edge_linear = self.w_e2l(edge_feat)

        n2epool_input = gnn_spmm(n2e_sp, input_node_linear)
        
        input_message = input_edge_linear + n2epool_input
        input_potential = F.relu(input_message)

        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            e2epool = gnn_spmm(e2e_sp, cur_message_layer)
            edge_linear = self.conv_params(e2epool)                    
            merged_linear = edge_linear + input_message

            cur_message_layer = F.relu(merged_linear)
            lv += 1

        e2npool = gnn_spmm(e2n_sp, cur_message_layer)
        hidden_msg = F.relu(e2npool)
        out_linear = self.out_params(hidden_msg)
        reluact_fp = F.relu(out_linear)

        y_potential = gnn_spmm(subg_sp, reluact_fp)

        return F.relu(y_potential)
