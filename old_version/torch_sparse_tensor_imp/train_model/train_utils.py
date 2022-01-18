import torch
import pickle
import random
from Constants import *
from operator import itemgetter
from sklearn.preprocessing import RobustScaler
from sklearn.externals import joblib
import numpy as np
from Particles import Particles, sparse_tensor


def grid2box_normals(grid_filename, output_dir):
    with open(grid_filename, 'r') as gf:
        fluid_pos = np.empty((0, 3), dtype=np.float32)

        pos = np.empty((0, 3), dtype=np.float32)
        ptype = np.empty((0,), dtype=np.int64)
        for line in gf.readlines():
            info = line.strip().split(',')
            x, y, z = float(info[0]), float(info[1]), float(info[2])
            tpe = int(info[6])
            if tpe == 1:
                fluid_pos = np.concatenate((fluid_pos, np.array([[x, y, z]])), axis=0)
                continue
            pos = np.concatenate((pos, np.array([[x, y, z]])), axis=0)
            if tpe == 2:
                ptype = np.concatenate((ptype, np.array([1])))
            else:
                ptype = np.concatenate((ptype, np.array([3])))
    vel = np.zeros_like(pos)
    case = Particles(pos, vel, ptype, G, NU, RHO, DT)
    box, normals = case.get_normals()
    return box, normals
    # np.save(output_dir+'box.npy', box)
    # np.save(output_dir+'box_normals.npy', normals)
    # np.save(output_dir+'fluid.npy', fluid_pos)


def sparse_diagonalizer(arrs):
    batch_size = len(arrs)
    idxs = []
    vals = []
    m = 0

    for i in range(batch_size):
        adj = arrs[i]
        idx = adj._indices().t() + m
        idxs.append(idx)
        vals.append(adj._values())
        m += adj.size(0)
    idxs = torch.cat(idxs, dim=0)
    vals = torch.cat(vals, dim=0)

    block_diag_arr = torch.sparse.FloatTensor(idxs.t(), vals, torch.Size([m, m]))
    return block_diag_arr


def stack_sparse_arr(adjs):
    batch_size = len(adjs)
    idxs = []
    vals = []
    m = 0

    for i in range(batch_size):
        adj = adjs[i]
        idx = adj[0] + m
        val = adj[1]
        # if isinstance(idx, np.ndarray):
        #     idx = torch.from_numpy(idx)
        #     val = torch.from_numpy(val)
        idxs.append(idx)
        vals.append(val)
        m += adj[2]
    idxs = np.concatenate(idxs, axis=0)
    vals = np.concatenate(vals, axis=0)
    idxs = torch.from_numpy(idxs)
    vals = torch.from_numpy(vals)

    block_diag_arr = torch.sparse.FloatTensor(idxs.t(), vals, torch.Size([m, m]))
    return block_diag_arr


def stack_idx(adjs, ids_lst):
    stack_lst = []
    past = 0
    for i in range(len(adjs)):
        arr_size = adjs[i][2]
        stack_lst.append(ids_lst[i].reshape(-1,) + past)
        past += arr_size
    return np.concatenate(stack_lst, axis=0)


def stack_edge_idx(adjs):
    batch_idx = []
    past = 0
    for i in range(len(adjs)):
        arr_size = adjs[i][1]
        batch_idx.append(adjs[i][0].reshape(-1, 2) + past)
        past += arr_size
    return np.concatenate(batch_idx, axis=0).transpose()


def stack_edge(edge_attrs, edge_idxs, edge_sizes):
    batch_idx = []
    past = 0
    for i in range(len(edge_idxs)):
        global_idx = edge_idxs[i] + past
        batch_idx.append(global_idx)
        past += edge_sizes[i]
    batch_edge_attr = torch.cat(edge_attrs, dim=0)
    batch_edge_idx = torch.cat(batch_idx, dim=0)
    tot_size = past
    return batch_edge_attr, batch_edge_idx, tot_size


def load_data(filename):
    data = np.load(filename)
    return data


class DataLoader(object):
    def __init__(self, adjs, batch_size=64, train_test_split=0.95, only_idx=False):

        self.adjs = adjs
        self.batch_size = batch_size
        self.only_idx = only_idx

        self.total_num = len(adjs)
        self.train_num = int(self.total_num * train_test_split)
        self.test_num = self.total_num - self.train_num
        self.ids = list(range(self.total_num))

        random.shuffle(self.ids)
        self.train_ids = self.ids[:self.train_num][:]
        self.test_ids = self.ids[self.train_num:][:]

        self.feat_scaler = RobustScaler()
        self.gt_scaler = RobustScaler()
        self.feature, self.gt, self.scale_feature, self.scale_gt = [], [], [], []

        self.all_data = None
        self.feat_selected = None
        self.reach_all = False
        self.cursor = 0
        self.feat_prev = -1

    def get_train_size(self):
        return len(self.train_ids)

    def get_all(self, scale=True, feat_idx=-1):
        if scale:
            if self.feat_scaler is None or self.gt_scaler is None:
                raise Exception("scaler missing")
        random.shuffle(self.train_ids)
        feat = self.scale_feature if scale else self.feature

        if self.feat_selected is None or (feat_idx != self.feat_prev):
            self.feat_selected = feat if feat_idx == -1 else [f[:, feat_idx].view(-1, 1) for f in feat]
            self.feat_prev = feat_idx

        return itemgetter(*self.train_ids)(self.adjs), itemgetter(*self.train_ids)(self.feat_selected), itemgetter(
            *self.train_ids)(self.scale_gt)

    def get_testset(self, scale=True):
        if scale:
            return itemgetter(*self.test_ids)(self.adjs), itemgetter(*self.test_ids)(self.scale_feature), itemgetter(
                *self.test_ids)(self.scale_gt)
        else:
            return itemgetter(*self.test_ids)(self.adjs), itemgetter(*self.test_ids)(self.feature), itemgetter(
                *self.test_ids)(self.gt)

    def dump_scaler(self, feature_pt, gt_pt):
        joblib.dump(self.feat_scaler, feature_pt)
        joblib.dump(self.gt_scaler, gt_pt)

    def scale_data(self):
        raise NotImplementedError

    def init_batch(self, scale, feat_idx):
        if self.reach_all or self.all_data is None:
            self.reach_all = False
            self.cursor = 0
            random.shuffle(self.train_ids)
            self.all_data = self.get_all(scale, feat_idx)

    def get_batch(self, scale=True, feat_idx=-1, no_adj=False):
        self.init_batch(scale, feat_idx)
        adjs, feat, gt = self.all_data
        temp = self.cursor + self.batch_size
        if temp < len(self.train_ids):
            if not no_adj and not self.only_idx:
                adj_batch = stack_sparse_arr(adjs[self.cursor:temp])
            elif self.only_idx:
                adj_batch = torch.from_numpy(stack_idx(adjs[self.cursor:temp]))
                adj_batch = adj_batch.long()
            feat_batch = torch.cat(feat[self.cursor:temp], dim=0)
            gt_batch = torch.cat(gt[self.cursor:temp], dim=0)
            self.cursor = temp
        else:
            if not no_adj and not self.only_idx:
                adj_batch = stack_sparse_arr(adjs[self.cursor:])
            elif self.only_idx:
                adj_batch = torch.from_numpy(stack_idx(adjs[self.cursor:]))
                adj_batch = adj_batch.long()
            feat_batch = torch.cat(feat[self.cursor:], dim=0)
            gt_batch = torch.cat(gt[self.cursor:], dim=0)
            self.cursor = len(self.train_ids)
            self.reach_all = True
        if not no_adj:
            return adj_batch, feat_batch, gt_batch, self.cursor, self.reach_all
        else:
            return feat_batch, gt_batch, self.cursor, self.reach_all


class PressureDataLoader(DataLoader):
    def __init__(self, adjs, d_feat, v_feat, prs_gt, prs_scaler, density_scaler,
                 batch_size=64, train_test_split=0.95, only_idx=False):
        super(PressureDataLoader, self).__init__(adjs, batch_size, train_test_split, only_idx=only_idx)
        self.d_feat, self.v_feat, self.prs_gt = \
            d_feat, v_feat, prs_gt
        self.gt_scaler = prs_scaler
        self.feat_scaler = density_scaler
        self.scale_data()

    def scale_data(self):
        # d_feat_all = np.concatenate(self.d_feat, axis=0).reshape(-1, 1)
        # v_feat_all = np.concatenate(self.v_feat, axis=0).reshape(-1, 3)
        # acc_all = np.concatenate(self.p_gt, axis=0)
        # feature_all = np.concatenate((d_feat_all, v_feat_all), axis=1)
        #
        # self.feat_scaler.fit(d_feat_all)
        # self.gt_scaler.fit(p_all)
        #
        for i in range(len(self.ids)):
            temp_f = np.concatenate((self.d_feat[i].reshape(-1, 1)
                                     , self.v_feat[i].reshape(-1, 3)), axis=1).astype(np.float32)
            self.feature.append(torch.from_numpy(temp_f))

            temp_f = np.concatenate(
                (self.feat_scaler.transform(self.d_feat[i].reshape(-1, 1)),self.v_feat[i].reshape(-1, 3)),
                axis=1).astype(np.float32)
            self.scale_feature.append(torch.from_numpy(temp_f))

            gt = self.prs_gt[i].reshape(-1, 1).astype(np.float32)
            self.gt.append(torch.from_numpy(gt))

            # self.scale_gt.append(torch.from_numpy(temp_gt))
            gt = self.gt_scaler.transform(gt)
            self.scale_gt.append(torch.from_numpy(gt))


class CollisionDataLoader(DataLoader):
    def __init__(self, adjs, ids, v_feat, v_gt, batch_size=64, train_test_split=0.95):
        super(CollisionDataLoader, self).__init__(adjs, batch_size, train_test_split)
        self.ids, self.v_feat, self.v_gt = [np.array(idx) for idx in ids], v_feat, v_gt
        self.ids_choice = None
        self.adjs_selected = None
        self.scale_adjs = None
        self.scale_data()

    def scale_data(self):

        # v_feat_all = np.concatenate(self.v_feat, axis=0).reshape(-1, 3)
        # v_gt_all = np.concatenate(self.v_gt, axis=0).reshape(-1, 3)
        # v_all = np.concatenate((v_feat_all, v_gt_all), axis=0)
        #
        # self.feat_scaler.fit(v_all)
        # self.gt_scaler = self.feat_scaler

        adjs = []
        # scale_adjs = []
        for adj in self.adjs:
            edge_attr, edge_idx, edge_size = adj[0], adj[1], adj[2]
            #if len(edge_attr) == 0:
            #    scale_edge_attr = edge_attr
            # else:
            #     v = edge_attr[:, 0:5:2]
            #     p = edge_attr[:, 1:6:2]
            #     scale_v = self.feat_scaler.transform(v)
            #     scale_edge_attr = np.concatenate((scale_v[:, 0].reshape(-1, 1), p[:, 0].reshape(-1, 1),
            #                                       scale_v[:, 1].reshape(-1, 1), p[:, 1].reshape(-1, 1),
            #                                       scale_v[:, 2].reshape(-1, 1), p[:, 2].reshape(-1, 1)), axis=1)
            # scale_edge_attr = torch.from_numpy(scale_edge_attr)
            edge_attr = torch.from_numpy(edge_attr)
            edge_idx = torch.from_numpy(edge_idx)
            adjs.append((edge_attr, edge_idx, edge_size))
            #scale_adjs.append((scale_edge_attr, edge_idx, edge_size))
        self.adjs = adjs
        self.scale_adjs = adjs

        self.feature = [torch.from_numpy(v) for v in self.v_feat]
        self.gt = [torch.from_numpy(v) for v in self.v_gt]

        self.scale_feature = self.feature[:]
        self.scale_gt = [torch.from_numpy(v) for v in self.v_gt]
        #self.gt_scaler.transform(v)) for v in self.v_gt]

    def get_all(self, scale=True, feat_idx=-1):
        if scale:
            if self.feat_scaler is None or self.gt_scaler is None:
                raise Exception("scaler missing")
        random.shuffle(self.train_ids)
        adjs = self.scale_adjs if scale else self.adjs
        feat = self.scale_feature if scale else self.feature
        if self.feat_selected is None or (feat_idx != self.feat_prev):
            self.feat_selected = feat if feat_idx == -1 else [f[:, feat_idx].reshape(-1, 1) for f in feat]
            self.feat_prev = feat_idx
        if self.adjs_selected is None:
            self.adjs_selected = adjs
        return itemgetter(*self.train_ids)(self.adjs_selected), itemgetter(*self.train_ids)(self.feat_selected), \
               itemgetter(*self.train_ids)(self.scale_gt)

    def init_batch(self, scale, feat_idx):
        if self.reach_all or self.all_data is None:
            self.reach_all = False
            self.cursor = 0
            random.shuffle(self.train_ids)
            self.all_data = self.get_all(scale, feat_idx)
            self.ids_choice = itemgetter(*self.train_ids)(self.ids)

    def get_batch(self, scale=True, feat_idx=-1):
        self.init_batch(scale, feat_idx)
        adjs, feat, gt = self.all_data
        temp = self.cursor + self.batch_size
        if temp < len(self.train_ids):
            ids_batch = stack_idx(adjs[self.cursor:temp], self.ids_choice[self.cursor:temp])
            edge_attrs = [adj[0] for adj in adjs[self.cursor:temp]]
            edge_idxs = [adj[1] for adj in adjs[self.cursor:temp]]
            edge_sizes = [adj[2] for adj in adjs[self.cursor:temp]]
            adj_batch = stack_edge(edge_attrs, edge_idxs, edge_sizes)

            feat_batch = torch.cat(feat[self.cursor:temp], dim=0)
            gt_batch = torch.cat(gt[self.cursor:temp], dim=0)
            self.cursor = temp
        else:
            ids_batch = stack_idx(adjs[self.cursor:], self.ids_choice[self.cursor:])
            edge_attrs = [adj[0] for adj in adjs[self.cursor:]]
            edge_idxs = [adj[1] for adj in adjs[self.cursor:]]
            edge_sizes = [adj[2] for adj in adjs[self.cursor:]]
            adj_batch = stack_edge(edge_attrs, edge_idxs, edge_sizes)

            feat_batch = torch.cat(feat[self.cursor:], dim=0)
            gt_batch = torch.cat(gt[self.cursor:], dim=0)
            self.cursor = len(self.train_ids)
            self.reach_all = True

        return adj_batch, feat_batch, gt_batch, self.cursor, self.reach_all, ids_batch


class BFDataLoader(DataLoader):
    def __init__(self, adjs, v_feat, acc, gravity=G, viscosity=NU, batch_size=64, train_test_split=0.95):
        super(BFDataLoader, self).__init__(adjs, batch_size, train_test_split)
        self.v_feat, self.acc = v_feat, acc
        self.g = gravity
        if isinstance(viscosity, list):
            self.nu = viscosity
        else:
            self.nu = [viscosity for _ in range(len(v_feat))]
        self.scale_data()

    def scale_data(self):
        feature = []
        #scale_feature = []
        #v_all = np.concatenate(self.v_feat, axis=0)
        #acc_all = np.concatenate(self.acc, axis=0)
        # self.feat_scaler.fit(v_all)
        # self.gt_scaler.fit(acc_all)

        for i, v in enumerate(self.v_feat):
            g_append = np.ones((v.shape[0], 1)) * self.g
            visc_append = np.ones((v.shape[0], 1)) * self.nu[i]
            feature.append(torch.from_numpy(np.concatenate((v, g_append, visc_append), axis=1)))
            # scale_feature.append(torch.from_numpy(np.concatenate((
            #                         self.feat_scaler.transform(v), g_append, visc_append), axis=1)))
        self.feature = feature
        self.gt = [torch.from_numpy(a) for a in self.acc]
        # self.scale_feature = scale_feature
        # self.scale_gt = [torch.from_numpy(self.gt_scaler.transform(a)) for a in self.acc]
        self.scale_feature = self.feature[:]
        self.scale_gt = self.gt[:]


class GCNPressureDataLoader(DataLoader):
    def __init__(self, adjs, d_feat, v_feat, prs_gt, prs_scaler, density_scaler,
                 batch_size=8, train_test_split=0.95):
        super(GCNPressureDataLoader, self).__init__(adjs, batch_size, train_test_split, only_idx=True)
        self.d_feat, self.v_feat, self.prs_gt = \
            d_feat, v_feat, prs_gt
        self.gt_scaler = prs_scaler
        self.feat_scaler = density_scaler
        self.scale_data()

    def scale_data(self):

        for i in range(len(self.ids)):
            temp_f = np.concatenate((self.d_feat[i].reshape(-1, 1)
                                     , self.v_feat[i].reshape(-1, 3)), axis=1).astype(np.float32)
            self.feature.append(torch.from_numpy(temp_f))

            temp_f = np.concatenate(
                (self.feat_scaler.transform(self.d_feat[i].reshape(-1, 1)),self.v_feat[i].reshape(-1, 3)),
                axis=1).astype(np.float32)
            self.scale_feature.append(torch.from_numpy(temp_f))

            gt = self.prs_gt[i].reshape(-1, 1).astype(np.float32)
            self.gt.append(torch.from_numpy(gt))

            # self.scale_gt.append(torch.from_numpy(temp_gt))
            gt = self.gt_scaler.transform(gt)
            self.scale_gt.append(torch.from_numpy(gt))

    def get_batch(self, scale=True, feat_idx=-1, no_adj=False):
            self.init_batch(scale, feat_idx)
            adjs, feat, gt = self.all_data
            temp = self.cursor + self.batch_size
            if temp < len(self.train_ids):
                adjs_large = [a[0] for a in adjs[self.cursor:temp]]
                adjs_small = [a[1] for a in adjs[self.cursor:temp]]
                adj_batch = torch.from_numpy(stack_edge_idx(adjs_large)),\
                            torch.from_numpy(stack_edge_idx(adjs_small))
                feat_batch = torch.cat(feat[self.cursor:temp], dim=0)
                gt_batch = torch.cat(gt[self.cursor:temp], dim=0)
                self.cursor = temp
            else:
                adjs_large = [a[0] for a in adjs[self.cursor:]]
                adjs_small = [a[1] for a in adjs[self.cursor:]]
                adj_batch = torch.from_numpy(stack_edge_idx(adjs_large)),\
                            torch.from_numpy(stack_edge_idx(adjs_small))
                feat_batch = torch.cat(feat[self.cursor:], dim=0)
                gt_batch = torch.cat(gt[self.cursor:], dim=0)
                self.cursor = len(self.train_ids)
                self.reach_all = True

            return adj_batch, feat_batch, gt_batch, self.cursor, self.reach_all

