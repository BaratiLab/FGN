import numpy as np
import torch
from torch import nn
import frnn
import dgl
from dgl.utils import expand_as_pair
import dgl.function as fn

def build_shared_mlp(mlp_spec, act_first=False, use_layer_norm=False, act_fn=nn.LeakyReLU(True)):
    layers = []
    for i in range(1, len(mlp_spec)):
        if act_first:
            layers.append(act_fn)
        layers.append(
            nn.Linear(mlp_spec[i - 1], mlp_spec[i], bias=not use_layer_norm)
        )
        if use_layer_norm:
            layers.append(
                nn.LayerNorm(mlp_spec[i])
            )
        if not act_first and i != len(mlp_spec) - 1:
            layers.append(act_fn)

    return nn.Sequential(*layers)


def LIN_KERNEL(r: torch.Tensor, re):
    w = (re / r) - 1.0
    mask = torch.logical_or(r > re, r < 1e-8)
    w[mask] = 0.
    return w


def IDT_KERNEL(r: torch.Tensor, re):
    return torch.ones_like(r)


class FixedRadiusGraph(torch.nn.Module):
    def __init__(self, store_weights=False, kernel_fn=LIN_KERNEL):
        super(FixedRadiusGraph, self).__init__()
        self.store_weights = store_weights
        self.kernel_fn = kernel_fn

    @torch.no_grad()
    def build_fixed_radius_graph(self, pos_tsr: torch.Tensor, cutoff_radius):
        dists, nbr_idxs, _, _ = frnn.frnn_grid_points(
            pos_tsr[None, ...], pos_tsr[None, ...],
            K=128,
            r=cutoff_radius,
            grid=None, return_nn=False, return_sorted=True
        )
        nbr_idx = nbr_idxs.squeeze(0)
        center_idx = nbr_idx.clone()
        center_idx[:] = torch.arange(pos_tsr.shape[0]).to(pos_tsr.device).reshape(-1, 1)
        mask = nbr_idx != -1
        nbr_idx = nbr_idx[mask]
        center_idx = center_idx[mask]
        graph = dgl.graph((nbr_idx, center_idx))

        if self.store_weights:
            dists = dists.squeeze(0)
            dists = dists[mask]
            w = self.kernel_fn(torch.sqrt(dists), cutoff_radius)
            graph.edata['w'] = w
        return graph

    def forward(self, pos: torch.Tensor, cutoff) -> dgl.graph:
        return self.build_fixed_radius_graph(pos, cutoff)


class SmoothGCN(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat):
        super().__init__()
        self.mlp = build_shared_mlp([in_feat//2, out_feat], act_first=True)
        self.node_linear = nn.Linear(in_feat, in_feat//2)
        self.edge_linear = nn.Linear(in_feat, in_feat//2)

    def forward(self, x, g):
        with g.local_scope():
            h_src, h_dst = expand_as_pair(x)
            g.srcdata['h'] = h_src
            g.dstdata['h'] = h_dst
            g.update_all(fn.src_mul_edge('h', 'w', 'm'), fn.sum('m', 'h'))
            message = g.ndata['h']
        x = self.mlp(self.node_linear(x) + self.edge_linear(message))
        return x


class EdgeMessage(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat):
        super().__init__()
        self.node_mlp = build_shared_mlp([in_feat//2, in_feat//2, out_feat], act_first=True)
        self.edge_mlp = build_shared_mlp([in_feat//2, out_feat], act_first=True)

        self.node_linear1 = nn.Linear(in_feat, in_feat//2)
        self.edge_linear = nn.Linear(in_feat, in_feat//2)

        self.node_linear2 = nn.Linear(in_feat, in_feat//2)
        self.msg_linear = nn.Linear(in_feat, in_feat//2)

    def forward(self, g, x=None, update_node=True):
        if x is None:
            g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'h'))
            x = g.ndata['h']
        else:
            h_src, h_dst = expand_as_pair(x)
            g.srcdata['h'] = h_src
            g.dstdata['h'] = h_dst

        g.apply_edges(lambda edge: {'e': self.node_mlp(
                self.node_linear1(edge.src['h']) + self.edge_linear(edge.data['e']))})
        if update_node:
            g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'h'))
            message = g.ndata['h']
            x = self.edge_mlp(self.node_linear2(x) + self.msg_linear(message))
        return g, x


class NodeNetwork(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 layers_of_mp,
                 cutoff):
        super().__init__()
        self.node_encoder = build_shared_mlp([in_feat, in_feat*8, 64])
        self.node_decoder = build_shared_mlp([64, out_feat*8, out_feat])
        self.cutoff = cutoff
        self.graph_fn = FixedRadiusGraph(store_weights=True)

        self.mp_layers = nn.ModuleList([])
        for l in range(layers_of_mp):
            self.mp_layers.append(nn.ModuleList([nn.LayerNorm(64), SmoothGCN(64, 64)]))

    def forward(self, x, pos, g=None):
        x = self.node_encoder(x)
        if g is None:
            g = self.graph_fn(pos, self.cutoff)
        for norm, mp_layer in self.mp_layers:
            x = norm(x)
            x = mp_layer(x, g) + x
        x = self.node_decoder(x)
        return x

    def forward_with_timimg(self, x, pos, g=None):
        log_info = {}
        # build graph
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if g is None:
            g = self.graph_fn(pos, self.cutoff)
        end.record()
        torch.cuda.synchronize()
        log_info['graph_time'] = start.elapsed_time(end)

        # inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        x = self.node_encoder(x)
        for norm, mp_layer in self.mp_layers:
            x = norm(x)
            x = mp_layer(x, g) + x
        x = self.node_decoder(x)
        end.record()
        torch.cuda.synchronize()
        log_info['inference_time'] = start.elapsed_time(end)
        return x, log_info


class EdgeNetwork(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 layers_of_mp,
                 cutoff):
        super().__init__()
        self.edge_encoder = build_shared_mlp([6, in_feat*8, 64, 64], use_layer_norm=True, act_fn=nn.ReLU(True))
        self.edge_decoder = build_shared_mlp([64, out_feat*8, out_feat], act_fn=nn.ReLU(True))
        self.cutoff = cutoff
        self.graph_fn = FixedRadiusGraph(store_weights=False)

        self.mp_layers = nn.ModuleList([])
        for l in range(layers_of_mp):
            self.mp_layers.append(nn.ModuleList([nn.LayerNorm(64), EdgeMessage(64, 64)]))

    @torch.no_grad()
    def build_graph(self, x, pos):
        g = self.graph_fn(pos, self.cutoff)
        g = g.remove_self_loop()
        raw_node_feat = torch.cat([x, pos], dim=1)
        with g.local_scope():
            g.ndata['h'] = raw_node_feat
            g.apply_edges(fn.u_sub_v('h', 'h', 'e'))
            raw_edge_feat = g.edata['e']
            raw_edge_feat[:, 3:] /= torch.norm(raw_edge_feat[:, 3:], dim=1).view(-1, 1)
        raw_edge_feat = self.edge_encoder(raw_edge_feat)
        g.edata['e'] = raw_edge_feat

        return g

    def forward(self, x, pos):
        # x is usually the velocity
        x_in = x
        g = self.build_graph(x, pos)
        if g.edges()[0].shape[0] > 0:
            flag = True
            for l, (norm, mp_layer) in enumerate(self.mp_layers):
                g.edata['e'] = norm(g.edata['e'])
                update_node = (l == len(self.mp_layers) - 1)
                if l == 0:
                    g, x = mp_layer(g, update_node=update_node)
                else:
                    g, x = mp_layer(g, x, update_node=update_node)
            g.apply_edges(lambda edge: {'e': self.edge_decoder(edge.data['e'])})
            g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'h'))
            x = g.ndata['h'] + x_in[:, :3]
        else:
            flag = False
            x = x_in[:, :3]
        return x, flag

    def forward_with_timing(self, x, pos):
        # x is usually the velocity
        x_in = x
        log_info = {}
        # build graph
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        g = self.build_graph(x, pos)

        end.record()
        torch.cuda.synchronize()
        log_info['graph_time'] = start.elapsed_time(end)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if g.edges()[0].shape[0] > 0:
            flag = True
            for l, (norm, mp_layer) in enumerate(self.mp_layers):
                g.edata['e'] = norm(g.edata['e'])
                update_node = (l == len(self.mp_layers) - 1)
                if l == 0:
                    g, x = mp_layer(g, update_node=update_node)
                else:
                    g, x = mp_layer(g, x, update_node=update_node)
            g.apply_edges(lambda edge: {'e': self.edge_decoder(edge.data['e'])})
            g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'h'))
            x = g.ndata['h'] + x_in[:, :3]
        else:
            flag = False
            x = x_in[:, :3]

        end.record()
        torch.cuda.synchronize()
        log_info['inference_time'] = start.elapsed_time(end)
        return x, flag, log_info



if __name__ == '__main__':
    x = torch.randn(1000, 4).cuda()
    pos = torch.randn(1000, 3).cuda()
    vel = torch.randn(1000, 3).cuda()
    # prs_net = NodeNetwork(4, 1, 2, 0.1).cuda()
    # print(prs_net(x, pos).shape)
    col_net = EdgeNetwork(3, 3, 1, 0.05).cuda()
    print(col_net(vel, pos).shape)

