# from Generator import DAM_COL_CASE
import numpy as np
from Particles import Particles
from Operator import Divergence, Laplacian
import torch
from torch.nn.modules import Module
from torch import nn
from torch.nn.parameter import Parameter
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, SAGEConv


class LagrangianConv2D(Module):

    def __init__(self, in_size, out_size, scale=None, bias=True):
        super(LagrangianConv2D, self).__init__()
        self.in_size = in_size
        self.weight = Parameter(torch.FloatTensor(in_size, out_size))
        if scale is None:
            scale = 1. / np.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-scale, scale)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_size))
            self.bias.data.uniform_(-scale, scale)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class LagrangianConv3D(Module):

    def __init__(self, in_size, out_size, scale=None, bias=True, keep_dim=False, dim_wise=False):
        super(LagrangianConv3D, self).__init__()
        self.in_size = in_size
        if dim_wise and in_size != 3:
            raise Exception("Under dimension wise mode, only accept three channel input with in_size=3")
        if keep_dim and out_size != 1:
            raise Exception("Under keep dimension mode, only accept 1 channel output with out_size=1")
        self.weight = Parameter(torch.FloatTensor(3, 1, out_size)) if dim_wise\
                        else Parameter(torch.FloatTensor(3, in_size, out_size))
        if scale is None:
            scale = 1. / np.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-scale, scale)
        if bias:
            self.bias = Parameter(torch.FloatTensor(3, out_size))
            self.bias.data.uniform_(-scale, scale)
        else:
            self.register_parameter('bias', None)
        self.keep_dim = keep_dim
        self.dim_wise = dim_wise

    def forward(self, input, adj3d):
        adj_x, adj_y, adj_z = adj3d
        if self.dim_wise:
            support_x = torch.mm(input[:, 0].view(-1, 1), self.weight[0])
            support_y = torch.mm(input[:, 1].view(-1, 1), self.weight[1])
            support_z = torch.mm(input[:, 2].view(-1, 1), self.weight[2])
        else:
            support_x = torch.mm(input, self.weight[0])
            support_y = torch.mm(input, self.weight[1])
            support_z = torch.mm(input, self.weight[2])

        if not self.keep_dim:
            output = torch.sparse.mm(adj_x, support_x) + torch.sparse.mm(adj_y, support_y) + torch.sparse.mm(adj_z, support_z)
            if self.bias is not None:
                return output + torch.sum(self.bias, dim=0)
            else:
                return output
        else:
            if self.bias is not None:
                out_x = torch.sparse.mm(adj_x, support_x) + self.bias[0]
                out_y = torch.sparse.mm(adj_y, support_y) + self.bias[1]
                out_z = torch.sparse.mm(adj_z, support_z) + self.bias[2]
            else:
                out_x = torch.sparse.mm(adj_x, support_x)
                out_y = torch.sparse.mm(adj_y, support_y)
                out_z = torch.sparse.mm(adj_z, support_z)
            output = torch.cat((out_x, out_y, out_z), dim=1)
            return output


class SmoothGCN2D(nn.Module):
    def __init__(self, layer_settings, GCN_layer=1):
        super(SmoothGCN2D, self).__init__()

        layers = []
        self.GCN_layer = GCN_layer
        for layer_num, layer_setting in enumerate(layer_settings):
            in_feat = layer_setting[0]
            out_feat = layer_setting[1]
            if layer_num < GCN_layer:
                if 3 > len(layer_setting) > 2:
                    scale = layer_setting[2]
                    gcn_layer = LagrangianConv2D(in_feat, out_feat, scale)
                elif len(layer_setting) == 3:
                    scale = layer_setting[2]
                    bias = layer_setting[3]
                    gcn_layer = LagrangianConv2D(in_feat, out_feat, scale, bias)
                else:
                    gcn_layer = LagrangianConv2D(in_feat, out_feat)
                layers.append(gcn_layer)
            else:
                # layers.append(nn.LayerNorm(in_feat))
                layers.append(nn.Linear(in_feat, out_feat))

            non_linear = nn.ReLU()
            if layer_num != len(layer_settings) - 1:
                layers.append(non_linear)
        self.model = nn.ModuleList(layers)

    def forward(self, input, adj):
        x = input.float()
        adj_model = adj.float()

        for l, layer in enumerate(self.model):
            if l % 2 == 0 and l < self.GCN_layer:
                x = layer(x, adj_model)
            else:
                x = layer(x)
        return x


class BFModel(SmoothGCN2D):
    def __init__(self, layer_settings, GCN_layer=1):
        super(BFModel, self).__init__(layer_settings, GCN_layer)

    def forward(self, input, adj):
        x = input.float()
        adj_model = adj.float()

        for l, layer in enumerate(self.model):
            if l % 2 == 0 and l/2 < self.GCN_layer:
                x = layer(x, adj_model)
            else:
                x = layer(x)
        return x


# simplified version of collision net
class ColModel(nn.Module):
    def __init__(self):
        super(ColModel, self).__init__()

        edgeEncoder = []
        edgeEncoder += [nn.Linear(6, 128), nn.ReLU()]
        edgeEncoder += [nn.LayerNorm(128)]
        edgeEncoder += [nn.Linear(128, 64), nn.ReLU()]
        edgeEncoder += [nn.LayerNorm(64)]
        edgeEncoder += [nn.Linear(64, 32), nn.ReLU()]
        edgeEncoder += [nn.LayerNorm(32)]
        edgeEncoder += [nn.Linear(32, 3)]
        self.edgeEncoder = nn.Sequential(*edgeEncoder)
        self.LagConv = LagrangianConv3D(1, 1, keep_dim=True)

    def forward(self, input, adj3d, v):

        x = input.float()
        edge_attr, edge_idx, tot_size = adj3d[0].float(), adj3d[1].long(), adj3d[2]
        enc_edge_attr = self.edgeEncoder(edge_attr)

        attr_x = enc_edge_attr[:, 0]
        attr_y = enc_edge_attr[:, 1]
        attr_z = enc_edge_attr[:, 2]

        enc_adj_x = torch.sparse.FloatTensor(edge_idx.t(), attr_x.view(-1,),
                                         torch.Size([tot_size, tot_size]))
        enc_adj_y = torch.sparse.FloatTensor(edge_idx.t(), attr_y.view(-1, ),
                                             torch.Size([tot_size, tot_size]))
        enc_adj_z = torch.sparse.FloatTensor(edge_idx.t(), attr_z.view(-1, ),
                                             torch.Size([tot_size, tot_size]))
        enc_adj = enc_adj_x, enc_adj_y, enc_adj_z
        output = self.LagConv(x, enc_adj)
        return output + v



class EdgeMessage(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(EdgeMessage, self).__init__()
        edgeProcessor = []
        edgeProcessor += [nn.Linear(edge_dim + node_dim * 2, 64), nn.ReLU()]
        edgeProcessor += [nn.Linear(64, 64), nn.ReLU()]
        edgeProcessor += [nn.Linear(64, 64), nn.LayerNorm(64)]
        self.edgeProcessor = nn.Sequential(*edgeProcessor)

        nodeProcessor = []
        nodeProcessor += [nn.Linear(node_dim + edge_dim, 64), nn.ReLU()]
        nodeProcessor += [nn.Linear(64, 64), nn.ReLU()]
        nodeProcessor += [nn.Linear(64, 64), nn.LayerNorm(64)]
        self.nodeProcessor = nn.Sequential(*nodeProcessor)

    def aggregate(self, edge_feat, edge_idx, node_size):
        device = edge_feat.device
        agg_edge_feat = torch.zeros((node_size, edge_feat.size(1)), dtype=torch.float32, device=device)
        scatter_idx = edge_idx[:, 0]
        scatter(edge_feat, scatter_idx, dim=0, out=agg_edge_feat)
        return agg_edge_feat

    def forward(self, node_feat, edge_feat, edge_idx, node_size):
        center_idx = edge_idx[:, 0]    # i
        source_idx = edge_idx[:, 1]    # j
        stacked_rs = torch.cat((node_feat[center_idx, :], node_feat[source_idx, :]), dim=1)

        stacked_edge_feat = torch.cat((edge_feat, stacked_rs), dim=1)

        proc_edge_feat = self.edgeProcessor(stacked_edge_feat)
        nodal_edge_feat = self.aggregate(proc_edge_feat, edge_idx, node_size)
        stacked_node_feat = torch.cat((node_feat, nodal_edge_feat), dim=1)
        proc_node_feat = self.nodeProcessor(stacked_node_feat)

        return proc_edge_feat, proc_node_feat


# full version of collision net
class DeepColModel(nn.Module):
    def __init__(self, block_num=8):
        super(DeepColModel, self).__init__()

        edgeEncoder = []
        edgeEncoder += [nn.Linear(4, 64), nn.ReLU()]
        edgeEncoder += [nn.Linear(64, 128), nn.ReLU()]
        edgeEncoder += [nn.Linear(128, 128), nn.LayerNorm(128)]

        self.edgeEncoder = nn.Sequential(*edgeEncoder)

        edgeDecoder = []
        edgeDecoder += [nn.Linear(128, 64), nn.ReLU()]
        edgeDecoder += [nn.Linear(64, 32), nn.ReLU()]
        edgeDecoder += [nn.Linear(32, 3)]

        self.edgeDecoder = nn.Sequential(*edgeDecoder)

        GN = []
        for block in range(block_num):
            GN += [EdgeMessage(128, 128)]
        self.GN = nn.ModuleList(GN)

    def forward(self, edge_feat, edge_idx, node_size):
        enc_edge_feat = self.edgeEncoder(edge_feat)
        enc_node_feat = self.GN[0].aggregate(enc_edge_feat, edge_idx, node_size)
        last_node = enc_node_feat
        last_edge = enc_edge_feat
        for block in self.GN:
            current_edge, current_node = block.forward(last_node, last_edge, edge_idx, node_size)
            last_edge += current_edge
            last_node += current_node
        last_edge = self.edgeDecoder(last_edge)
        prediction = self.GN[0].aggregate(last_edge, edge_idx, node_size)
        return prediction


class GNBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim):
        super(GNBlock, self).__init__()
        edgeProcessor = []
        edgeProcessor += [nn.Linear(edge_dim + node_dim * 2, 64), nn.ReLU()]
        edgeProcessor += [nn.Linear(64, 64), nn.ReLU()]
        edgeProcessor += [nn.Linear(64, 64), nn.LayerNorm(64)]
        self.edgeProcessor = nn.Sequential(*edgeProcessor)

        nodeProcessor = []
        nodeProcessor += [nn.Linear(node_dim + edge_dim + global_dim, 64), nn.ReLU()]
        nodeProcessor += [nn.Linear(64, 64), nn.ReLU()]
        nodeProcessor += [nn.Linear(64, 64), nn.LayerNorm(64)]
        self.nodeProcessor = nn.Sequential(*nodeProcessor)

    def aggregate(self, edge_feat, edge_idx, node_size):
        device = edge_feat.device
        agg_edge_feat = torch.zeros((node_size, edge_feat.size(1)), dtype=torch.float32, device=device)
        scatter_idx = edge_idx[:, 0]
        scatter(edge_feat, scatter_idx, dim=0, out=agg_edge_feat)
        return agg_edge_feat

    def forward(self, node_feat, edge_feat, global_feat, edge_idx, node_size):
        receiver_idx = edge_idx[:, 0]
        sender_idx = edge_idx[:, 1]
        stacked_rs = torch.cat((node_feat[receiver_idx, :], node_feat[sender_idx, :]), dim=1)

        stacked_edge_feat = torch.cat((edge_feat, stacked_rs), dim=1)

        proc_edge_feat = self.edgeProcessor(stacked_edge_feat)
        nodal_edge_feat = self.aggregate(proc_edge_feat, edge_idx, node_size)
        stacked_node_feat = torch.cat((node_feat, nodal_edge_feat, global_feat), dim=1)
        proc_node_feat = self.nodeProcessor(stacked_node_feat)

        return proc_edge_feat, proc_node_feat


class GNS(nn.Module):
    def __init__(self, block_num=8):
        super(GNS, self).__init__()
        nodeEncoder = []  # constant, particle type, vel_0, vel_1, vel_2, vel_3, vel_4, vel_5
        nodeEncoder += [nn.Linear(17, 64), nn.ReLU()]
        nodeEncoder += [nn.Linear(64, 64), nn.ReLU()]
        nodeEncoder += [nn.Linear(64, 64), nn.LayerNorm(64)]

        self.nodeEncoder = nn.Sequential(*nodeEncoder)

        edgeEncoder = []
        edgeEncoder += [nn.Linear(4, 64), nn.ReLU()]
        edgeEncoder += [nn.Linear(64, 64), nn.ReLU()]
        edgeEncoder += [nn.Linear(64, 64), nn.LayerNorm(64)]

        self.edgeEncoder = nn.Sequential(*edgeEncoder)

        nodeDecoder = []
        nodeDecoder += [nn.Linear(64, 64), nn.ReLU()]
        nodeDecoder += [nn.Linear(64, 32), nn.ReLU()]
        nodeDecoder += [nn.Linear(32, 3)]

        self.nodeDecoder = nn.Sequential(*nodeDecoder)

        GN = []
        for block in range(block_num):
            GN += [GNBlock(64, 64, 2)]
        self.GN = nn.ModuleList(GN)

    def forward(self, node_feat, edge_feat, global_feat, edge_idx, node_size):
        enc_node_feat = self.nodeEncoder(node_feat)
        enc_edge_feat = self.edgeEncoder(edge_feat)
        last_node = enc_node_feat
        last_edge = enc_edge_feat
        for block in self.GN:
            current_edge, current_node = block.forward(last_node, last_edge, global_feat, edge_idx, node_size)
            last_edge += current_edge
            last_node += current_node
        prediction = self.nodeDecoder(last_node)
        return prediction


class PrsModel(SmoothGCN2D):
    def __init__(self, layer_settings, GCN_layer=1):
        super(PrsModel, self).__init__(layer_settings, GCN_layer)

    def forward(self, input, adj):
        x = input.float()
        for l, layer in enumerate(self.model):
            if l % 2 == 0 and l/2 < self.GCN_layer:
                x = layer(x, adj)
            else:
                x = layer(x)
        return x


class GeneralNet(torch.nn.Module):
    def __init__(self, layer_settings, GCN_type, GCN_layer=1):
        super(GeneralNet, self).__init__()
        gcn_layers = []
        for l in range(GCN_layer):
            in_channels = layer_settings[l]['in_channels']
            out_channels = layer_settings[l]['out_channels']
            if GCN_type == 'SAGE':
                gcn_layers += [SAGEConv(in_channels=in_channels,
                                        out_channels=out_channels)]
            elif GCN_type == 'GCN':
                gcn_layers += [GCNConv(in_channels=in_channels,
                                       out_channels=out_channels)]
        self.gcn_layers = nn.ModuleList(gcn_layers)
        in_feat = layer_settings[0]['in_channels']
        mlp_in_feat = layer_settings[GCN_layer]['in_channels']
        self.skip_fc = nn.Linear(in_feat, mlp_in_feat)
        linear_layers = []
        for l in range(GCN_layer, len(layer_settings)-1):
            in_channels = layer_settings[l]['in_channels']
            out_channels = layer_settings[l]['out_channels']
            linear_layers += [nn.Linear(in_channels, out_channels),
                              nn.ReLU()]
        in_channels = layer_settings[-1]['in_channels']
        out_channels = layer_settings[-1]['out_channels']
        linear_layers += [nn.Linear(in_channels, out_channels)]
        self.linear_layers = nn.Sequential(*linear_layers)

    def forward(self, x, edge_idxes):
        input = x.clone()
        for l, graph_conv in enumerate(self.gcn_layers):
            x = graph_conv(x, edge_idxes[l])
        x += self.skip_fc(input)
        for linear_layer in self.linear_layers:
            x = linear_layer(x)

        return x

