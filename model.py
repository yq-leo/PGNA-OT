import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import scipy


class BRIGHT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BRIGHT, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(input_dim + hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, G1_data, G2_data):
        x1, x2 = G1_data.x, G2_data.x
        pos_emd1 = torch.cat([x1, self.act(self.lin1(x1))], dim=1)
        pos_emd2 = torch.cat([x2, self.act(self.lin1(x2))], dim=1)
        pos_emd1 = self.lin2(pos_emd1)
        pos_emd2 = self.lin2(pos_emd2)
        pos_emd1 = F.normalize(pos_emd1, p=2, dim=1)
        pos_emd2 = F.normalize(pos_emd2, p=2, dim=1)
        return pos_emd1, pos_emd2


class PGNNLayer(torch.nn.Module):
    def __init__(self, input_dim, anchor_dim, output_dim, dist_trainable=False, use_hidden=False,
                 mcf_type='anchor', agg_type='mean', **kwargs):
        """
        One PGNA Layer
        :param input_dim: input feature dimension
        :param anchor_dim: num of anchor nodes
        :param output_dim: output feature dimension
        :param dist_trainable: whether to use trainable distance metric scores
        :param mcf_type: type of message computation function (e.g. default, concat, mean, etc.)
        :param agg_type: type of message aggregation function (e.g. mean, sum, max, etc.)
        :param use_hidden: whether to use SLP after message computation function F
        :param kwargs: optional arguments
        """
        super(PGNNLayer, self).__init__()

        self.input_dim = input_dim
        self.anchor_dim = anchor_dim
        self.message_dim = input_dim if mcf_type != 'concat' else input_dim * 2
        self.output_dim = output_dim
        self.dist_trainable = dist_trainable
        self.mcf_type = mcf_type
        self.agg_type = agg_type
        self.use_hidden = use_hidden

        self.linear_hidden = nn.Linear(self.message_dim, self.input_dim) if self.use_hidden else None
        self.linear_final = nn.Linear(self.anchor_dim, self.output_dim)
        self.act = nn.ReLU()

    def forward(self, x1, x2, dists_max_1, dists_max_2, dists_argmax_1, dists_argmax_2):
        if self.dist_trainable:
            dists_max_1 = self.dist_compute(dists_max_1.unsqueeze(-1)).squeeze()
            dists_max_2 = self.dist_compute(dists_max_2.unsqueeze(-1)).squeeze()

        anchor_features_1 = x1[dists_argmax_1, :]
        self_features_1 = x1.unsqueeze(1).repeat(1, dists_max_1.shape[1], 1)
        messages_1 = self.mcf(self_features_1, anchor_features_1, dists_max_1)
        del anchor_features_1, self_features_1

        anchor_features_2 = x2[dists_argmax_2, :]
        self_features_2 = x2.unsqueeze(1).repeat(1, dists_max_2.shape[1], 1)
        messages_2 = self.mcf(self_features_2, anchor_features_2, dists_max_2)
        del anchor_features_2, self_features_2

        if self.use_hidden:
            assert self.linear_hidden is not None, 'Hidden layer is not defined'
            messages_1 = self.linear_hidden(messages_1).squeeze()
            messages_1 = self.act(messages_1)
            messages_2 = self.linear_hidden(messages_2).squeeze()
            messages_2 = self.act(messages_2)

        out1_position = self.linear_final(torch.sum(messages_1, dim=2))  # zv (output)
        out1_structure = self.agg(messages_1)  # hv (feed to the next layer)
        out2_position = self.linear_final(torch.sum(messages_2, dim=2))
        out2_structure = self.agg(messages_2)

        return out1_position, out1_structure, out2_position, out2_structure

    def mcf(self, node_feat, anchor_feat, distances):
        """
        Message Computation Function F
        :param node_feat: node features (hv)
        :param anchor_feat: anchorset features (hu)
        :param distances: distances metric scores (s(v, u))
        :return:
            messages: messages F(v, u, hv, hu)
        """
        assert self.mcf_type in ['anchor', 'concat', 'sum', 'mean', 'max', 'min'], 'Invalid MCF type'

        if self.mcf_type == 'anchor':
            return distances.unsqueeze(-1) * anchor_feat
        elif self.mcf_type == 'concat':
            return distances.unsqueeze(-1) * torch.cat((node_feat, anchor_feat), dim=-1)
        elif self.mcf_type == 'sum':
            return distances.unsqueeze(-1) * torch.sum(torch.stack((node_feat, anchor_feat), dim=0), dim=0)
        elif self.mcf_type == 'mean':
            return distances.unsqueeze(-1) * torch.mean(torch.stack((node_feat, anchor_feat), dim=0), dim=0)
        elif self.mcf_type == 'max':
            return distances.unsqueeze(-1) * torch.max(torch.stack((node_feat, anchor_feat), dim=0), dim=0)[0]
        elif self.mcf_type == 'min':
            return distances.unsqueeze(-1) * torch.min(torch.stack((node_feat, anchor_feat), dim=0), dim=0)[0]

    def agg(self, messages):
        """
        Message Aggregation Function AGG
        :param messages: message matrix Mv
        :return:
            out: aggregated messages
        """
        assert self.agg_type in ['mean', 'sum', 'max', 'min'], 'Invalid AGG type'

        if self.agg_type == 'mean':
            return torch.mean(messages, dim=1)
        elif self.agg_type == 'sum':
            return torch.sum(messages, dim=1)
        elif self.agg_type == 'max':
            return torch.max(messages, dim=1)[0]
        elif self.agg_type == 'min':
            return torch.min(messages, dim=1)[0]


class PGNA(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, anchor_dim, hidden_dim, output_dim,
                 feature_pre=False, num_layers=1, use_dropout=False, **kwargs):
        super(PGNA, self).__init__()
        self.feature_pre = feature_pre
        self.num_layers = num_layers
        self.use_dropout = use_dropout
        if num_layers == 1:
            hidden_dim = output_dim

        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNNLayer(feature_dim, anchor_dim, hidden_dim, **kwargs)
        else:
            self.conv_first = PGNNLayer(input_dim, anchor_dim, hidden_dim, **kwargs)

        if num_layers > 1:
            self.conv_hidden = nn.ModuleList(
                [PGNNLayer(hidden_dim, anchor_dim, hidden_dim, **kwargs)] * (num_layers - 2))
            self.conv_out = PGNNLayer(hidden_dim, anchor_dim, output_dim, **kwargs)

    def forward(self, G1_data, G2_data):
        x1, x2 = G1_data.x, G2_data.x
        dists_max_1, dists_max_2 = G1_data.rwr, G2_data.rwr

        device = x1.device
        num_anchors = G1_data.rwr.shape[1]
        n1, n2 = x1.shape[0], x2.shape[0]
        dists_argmax_1 = torch.arange(num_anchors).repeat(n1, 1).to(device)
        dists_argmax_2 = torch.arange(num_anchors).repeat(n2, 1).to(device)

        if self.feature_pre:
            x1 = self.linear_pre(x1)
            x2 = self.linear_pre(x2)
        x1_position, x1, x2_position, x2 = self.conv_first(x1, x2, dists_max_1, dists_max_2, dists_argmax_1,
                                                           dists_argmax_2)
        x1, x2 = F.sigmoid(x1), F.sigmoid(x2)

        if self.num_layers == 1:
            x1_position = F.normalize(x1_position, p=2, dim=-1)
            x2_position = F.normalize(x2_position, p=2, dim=-1)
            return x1_position, x2_position

        if self.use_dropout:
            x1 = F.dropout(x1, training=self.training)
            x2 = F.dropout(x2, training=self.training)

        for i in range(self.num_layers - 2):
            _, x1, _, x2 = self.conv_hidden[i](x1, x2, dists_max_1, dists_max_2, dists_argmax_1, dists_argmax_2)
            x1, x2 = F.sigmoid(x1), F.sigmoid(x2)
            if self.use_dropout:
                x1 = F.dropout(x1, training=self.training)
                x2 = F.dropout(x2, training=self.training)

        x1_position, x1, x2_position, x2 = self.conv_out(x1, x2, dists_max_1, dists_max_2, dists_argmax_1,
                                                         dists_argmax_2)
        x1_position = F.normalize(x1_position, p=2, dim=-1)
        x2_position = F.normalize(x2_position, p=2, dim=-1)
        return x1_position, x2_position


class RWRNet(torch.nn.Module):
    def __init__(self, num_layers, input_dim, output_dim):
        super(RWRNet, self).__init__()
        self.num_layers = num_layers
        self.beta = 0.5
        self.lin = torch.nn.Linear(input_dim, output_dim)
        self.gcn_in = GCNConv(input_dim, output_dim)
        self.gcn = nn.ModuleList([GCNConv(output_dim, output_dim)] * (num_layers - 1))
        self.act = nn.ReLU()

    def forward(self, G1_data, G2_data):
        x1, x2 = G1_data.x, G2_data.x
        init_emb1 = self.lin(x1)
        init_emb2 = self.lin(x2)
        pos_emd1 = self.act(self.gcn_in(x1, G1_data.edge_index))
        pos_emd2 = self.act(self.gcn_in(x2, G2_data.edge_index))
        for i in range(self.num_layers - 1):
            pos_emd1 = self.act(self.gcn[i]((1 - self.beta) * pos_emd1 + self.beta * init_emb1, G1_data.edge_index))
            pos_emd2 = self.act(self.gcn[i]((1 - self.beta) * pos_emd2 + self.beta * init_emb2, G2_data.edge_index))
        pos_emd1 = F.normalize((1 - self.beta) * pos_emd1 + self.beta * init_emb1, p=2, dim=1)
        pos_emd2 = F.normalize((1 - self.beta) * pos_emd2 + self.beta * init_emb2, p=2, dim=1)

        return pos_emd1, pos_emd2


class FusedGWLoss(torch.nn.Module):
    def __init__(self, G1_tg, G2_tg, lambda_edge=0, lambda_total=1e-2, in_iter=5, out_iter=10):
        super().__init__()
        self.device = G1_tg.x.device
        self.lambda_edge = lambda_edge
        self.lambda_total = lambda_total
        self.in_iter = in_iter
        self.out_iter = out_iter

        x1, x2 = F.normalize(G1_tg.x, p=2, dim=1), F.normalize(G2_tg.x, p=2, dim=1)
        self.n1, self.n2 = G1_tg.num_nodes, G2_tg.num_nodes
        self.intra_c1 = (torch.exp(-(x1 @ x1.T)) * G1_tg.adj).to(self.device)
        self.intra_c2 = (torch.exp(-(x2 @ x2.T)) * G2_tg.adj).to(self.device)

    def forward(self, out1, out2):
        inter_c = torch.exp(-(out1 @ out2.T))
        with torch.no_grad():
            s = sinkhorn(inter_c, self.intra_c1, self.intra_c2, self.in_iter, self.out_iter,
                         self.lambda_edge, self.lambda_total, self.device)
        return -torch.sum(inter_c * s)


def sinkhorn(inter_c, intra_c1, intra_c2, in_iter=5, out_iter=10, lambda_e=0, lambda_t=1e-2, device='cpu'):
    n1, n2 = inter_c.shape
    # marginal distribution
    a = torch.ones(n1).to(torch.float64).to(device) / n1
    b = torch.ones(n2).to(torch.float64).to(device) / n2
    # lagrange multiplier
    u = torch.ones(n1).to(torch.float64).to(device) / n1
    v = torch.ones(n2).to(torch.float64).to(device) / n2
    # transport plan
    s = torch.ones((n1, n2)).to(torch.float64).to(device) / (n1 * n2)

    for i in range(out_iter):
        L = (intra_c1 ** 2 @ a.view(-1, 1) @ torch.ones((1, n2)).to(torch.float64).to(device) +
             torch.ones((n1, 1)).to(torch.float64).to(device) @ b.view(1, -1) @ intra_c2 ** 2 -
             2 * intra_c1 @ s @ intra_c2.T)
        cost = inter_c + lambda_e * L

        K = torch.exp(-cost / lambda_t)
        for j in range(in_iter):
            u = a / (K @ v)
            v = b / (K.T @ u)
        s = 0.05 * s + 0.95 * torch.diag(u) @ K @ torch.diag(v)

    return s






