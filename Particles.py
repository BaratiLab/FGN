import numpy as np
import torch
import numba as nb
import open3d as o3d
import frnn
import dgl
from numba import types, deferred_type
from numba.typed import Dict, List
import pickle
from Constants import *
from Operator import *
from cuHelper import *
import time

I = 16  # 2^I is for hashing function
Large = 2**I


@nb.njit(parallel=True)
def arr2adj(idx_arr, val_arr, count, tot_size):
    i = np.zeros((tot_size, 2), dtype=np.int64)
    v = np.zeros((tot_size,), dtype=np.float32)
    for row in nb.prange(idx_arr.shape[0]):
        neighbor_num = idx_arr[row, 0]
        if row == 0:
            cursor = 0
        else:
            cursor = count[row-1]
        i[cursor:cursor+neighbor_num, 0] = row
        i[cursor:cursor+neighbor_num, 1] = idx_arr[row, 1:neighbor_num+1]
        v[cursor:cursor+neighbor_num] = val_arr[row, 1:neighbor_num+1]

    return i, v


@nb.njit(parallel=True, fastmath=True)
def get_density(cell, ids, control_radius):
    density = np.zeros((len(ids), ), dtype=np.float32)
    for i in nb.prange(len(ids)):
        id = ids[i]
        neighbors = cell.get_cell_neighbor(id)
        center_sum = 0.
        for neighbor in neighbors:
            r = cell.distance(id, neighbor)
            if r < control_radius:
                w = MPS_KERNEL(r, control_radius)
                center_sum += w
        density[i] = center_sum
    return density


@nb.njit(parallel=True)
def get_gradient(cell, ids, grad_val, control_radius):
    gradient = np.zeros((len(ids), 3), dtype=np.float32)
    val_min = cell.get_neighbor_min(grad_val, ids, control_radius)
    for i in nb.prange(len(ids)):
        id = ids[i]
        neighbors = cell.get_cell_neighbor(id)
        for neighbor in neighbors:
            if neighbor != id:
                r = cell.distance(id, neighbor)
                if r < control_radius:
                    r_vec = cell.pos[neighbor] - cell.pos[id]
                    w = MPS_KERNEL(r, control_radius)
                    val_diff = grad_val[neighbor] - val_min[id]
                    grad = w * val_diff / r ** 2
                    for dim in range(3):
                        gradient[i, dim] += grad * r_vec[dim]
    return gradient


@nb.njit
def get_all_collision_neighbor(cell, ids, vel, local_map, ptype, control_radius):
    A_ele = np.zeros((len(ids), 64), dtype=np.float32)
    A_idx = np.zeros((len(ids), 64), dtype=np.int64)

    for row in range(len(ids)):
        id = ids[row]
        if ptype[id] != 1:
            continue
        neighbors = cell.get_cell_neighbor(id)
        center_sum = 0.0

        for neighbor in neighbors:
            if neighbor != id:
                coll_term = 0.0
                r = cell.distance(id, neighbor)
                r_vec = cell.pos[neighbor] - cell.pos[id]
                v_vec = vel[id] - vel[neighbor]
                if r < control_radius:
                    for dim in range(3):
                        coll_term += v_vec[dim] * r_vec[dim]
                if coll_term > 0.0:
                    col = local_map.get(neighbor)
                    coll_term /= r ** 2
                    center_sum += coll_term
                    if not (col is None):
                        A_idx[row, 0] += 1
                        cursor = A_idx[row, 0]
                        A_ele[row, cursor] = -coll_term
                        A_idx[row, cursor] = col

        A_idx[row, 0] += 1
        cursor = A_idx[row, 0]
        A_ele[row, cursor] = center_sum
        A_idx[row, cursor] = row

    return A_ele, A_idx


@nb.njit(parallel=True)
def get_lap_neighbor(cell, ids, local_map, control_radius, PD=False):
    A_ele = np.zeros((len(ids), 384), dtype=np.float32)
    A_idx = np.zeros((len(ids), 384), dtype=np.int64)
    for row in nb.prange(len(ids)):
        id = ids[row]
        neighbors = cell.get_cell_neighbor(id)
        center_sum = 0.
        for neighbor in neighbors:
            if neighbor != id:
                r = cell.distance(id, neighbor)
                col = local_map.get(neighbor)
                if r < control_radius:
                    w = MPS_KERNEL(r, control_radius)
                    if PD:
                        w = -w
                    center_sum += w
                    if not (col is None):
                        A_idx[row, 0] += 1
                        cursor = A_idx[row, 0]
                        A_ele[row, cursor] = w
                        A_idx[row, cursor] = col

        A_idx[row, 0] += 1
        cursor = A_idx[row, 0]
        A_ele[row, cursor] = -center_sum
        A_idx[row, cursor] = row

    return A_ele, A_idx


@nb.njit(parallel=True, fastmath=True)
def get_gcn_neighbor(cell, ids, local_map, control_radius):
    A_ele = np.zeros((len(ids), 384), dtype=np.float32)
    A_idx = np.zeros((len(ids), 384), dtype=np.int64)

    for row in nb.prange(len(ids)):
        id = ids[row]
        neighbors = cell.get_cell_neighbor(id)
        center_sum = 0.
        for neighbor in neighbors:
            if neighbor != id:
                r = cell.distance(id, neighbor)
                col = local_map.get(neighbor)
                if r < control_radius and col is not None:
                    w = MPS_KERNEL(r, control_radius)
                    center_sum += w

                    A_idx[row, 0] += 1
                    cursor = A_idx[row, 0]
                    A_ele[row, cursor] = w
                    A_idx[row, cursor] = col
        A_idx[row, 0] += 1
        cursor = A_idx[row, 0]
        A_ele[row, cursor] = 1
        A_idx[row, cursor] = row
        A_ele[row, 1:cursor] /= center_sum
    return A_ele, A_idx


def sparse_tensor(idx, val, arr_size):
    if isinstance(idx, np.ndarray):
        i = torch.from_numpy(idx).view(-1, 2)
        v = torch.from_numpy(val).view(-1, )
    else:
        i = torch.tensor(idx, dtype=torch.int64).view(-1, 2)
        v = torch.tensor(val, dtype=torch.float32).view(-1, )
    return torch.sparse.FloatTensor(i.t(), v, torch.Size([arr_size, arr_size]))


class Particles:
    def __init__(self, positions, velocity, ptype, gravity, visc_coeff, rho, DT):
        """
        """
        self.pos = positions.astype(np.float32).reshape(-1, 3)
        self.vel = velocity.astype(np.float32).reshape(-1, 3)

        self.ptype = ptype.reshape(-1,)
        self.fluid_ids = np.argwhere(self.ptype == 1).astype(np.int64).reshape(-1, )
        self.nfs_ids = None    # non free surface particles
        self.non_dum_ids = np.argwhere(self.ptype != 3).astype(np.int64).reshape(-1, )
        self.total_num = ptype.shape[0]

        self.g = gravity
        self.nu = visc_coeff
        self.rho = rho

        self.pres = np.zeros((self.total_num, ))
        self.N0 = None
        self.N0_lap = None
        self.lam = None
        self.DT = DT
        self.device = None

        self.cache = {}

    def init_params(self, show_info=True, pass_in=None):
        if pass_in is None:
            # calculate the constant density of fluid field
            self.N0 = np.max(self.get_density(only_fluid=True))

            # calculate the density parameter for Laplacian operator
            self.N0_lap = np.max(self.get_density(control_radius=LAP_RADIUS, only_fluid=True))

            _, sqd_dist = self.get_nbr(LAP_RADIUS)

            self.lam = torch.max(torch.sum(sqd_dist, dim=1)).item()/self.N0_lap

            self.update_nfs()
        else:
            self.N0 = pass_in['N0']
            self.N0_lap = pass_in['N0_lap']
            self.lam = pass_in['lam']

        if show_info:
            print("fluid particle number: %i" % self.fluid_ids.shape[0])
            print("non dummy particle number: %i" % self.non_dum_ids.shape[0])
            print("total particle number: %i" % self.total_num)
            print("constant density N0: %.8f" % self.N0)
            print("constant density of Laplacian operator: %.8f" % self.N0_lap)
            print("Laplacian normalizer: %.8f" % self.lam)

    def get_nbr(self, cutoff, return_pic=False):
        pos_tsr = torch.from_numpy(self.pos).cuda()
        fluid_pos_tsr = pos_tsr[self.fluid_ids]
        sqd_dist, nbr_idx, _, _ = frnn.frnn_grid_points(
                                    fluid_pos_tsr[None, ...], pos_tsr[None, ...],
                                    K=128,
                                    r=cutoff,
                                    grid=None, return_nn=False, return_sorted=True
                                )
        nbr_idx = nbr_idx.squeeze(0)
        center_idx = nbr_idx.clone()
        center_idx[:] = torch.from_numpy(self.fluid_ids).to(pos_tsr.device).reshape(-1, 1)
        mask = nbr_idx != -1
        nbr_idx = nbr_idx[mask]
        center_idx = center_idx[mask]
        if return_pic:
            return torch.unique(nbr_idx).cpu().numpy()
        nbr_lst = torch.cat((center_idx.view(-1, 1), nbr_idx.view(-1, 1)), dim=1)
        sqd_dist = sqd_dist.squeeze(0)
        sqd_dist[sqd_dist < 1e-8] = 0.
        return nbr_lst.cpu().numpy(), sqd_dist

    def get_graph(self, cutoff, kernel_fn=None, with_col_pic=False):
        pos_tsr = torch.from_numpy(self.pos).cuda()
        fluid_pos_tsr = pos_tsr[self.fluid_ids]
        sqd_dist, nbr_idx, _, _ = frnn.frnn_grid_points(
            fluid_pos_tsr[None, ...], pos_tsr[None, ...],
            K=128,
            r=cutoff,
            grid=None, return_nn=False, return_sorted=True
        )
        nbr_idx = nbr_idx.squeeze(0)
        center_idx = nbr_idx.clone()
        center_idx[:] = torch.from_numpy(self.fluid_ids).to(pos_tsr.device).reshape(-1, 1)
        mask = nbr_idx != -1

        nbr_idx = nbr_idx[mask]
        center_idx = center_idx[mask]
        pic_index = torch.unique(nbr_idx).cpu().numpy()
        if with_col_pic:
            # col_cutoff = 2.1 * COL_RADIUS
            # col_pic_mask = sqd_dist.squeeze(0) < col_cutoff**2
            # valid_mask = torch.logical_and(col_pic_mask, mask)
            # col_pic = torch.unique(nbr_idx[valid_mask]).cpu().numpy()
            col_pic = pic_index

        mapped_nbr_idx = self.find_idx(pic_index, nbr_idx.cpu().numpy())
        mapped_center_idx = self.find_idx(pic_index, center_idx.cpu().numpy())
        graph = dgl.graph((torch.from_numpy(mapped_nbr_idx).cuda(), torch.from_numpy(mapped_center_idx).cuda()))
        if kernel_fn is not None:
            sqd_dist = sqd_dist.squeeze(0)
            sqd_dist = sqd_dist[mask]
            w = kernel_fn(torch.sqrt(sqd_dist), cutoff)
            graph.edata['w'] = w

        if with_col_pic:
            return graph, pic_index, col_pic
        return graph, pic_index

    def add_particles(self, vel, pos, verbose=True):
        add_pnum = vel.shape[0]
        self.vel = np.concatenate((self.vel, vel), axis=0)
        self.pos = np.concatenate((self.pos, pos), axis=0)
        self.ptype = np.concatenate((self.ptype, np.ones((add_pnum, ))), axis=0).reshape(-1, )
        self.fluid_ids = np.concatenate((self.fluid_ids,
                                            np.arange(add_pnum) + self.total_num),
                                        axis=0).reshape(-1, )

        self.total_num = self.ptype.shape[0]
        self.pres = np.zeros((self.total_num, ))
        if verbose:
            print("Add %i particles to fluid field" % add_pnum)

    def remove_particle(self, boundary, which_axis=0, verbose=True, right=True):
        if right:  # particles inside boundary
            inbound_fluid = np.logical_and(self.ptype == 1, self.pos[:, which_axis] < boundary)
        else:    # particles outside boundary
            inbound_fluid = np.logical_and(self.ptype == 1, self.pos[:, which_axis] > boundary)
        wall = self.ptype != 1
        mask = np.logical_or(inbound_fluid, wall)

        # remove out of boundary fluid particles from data structure
        prev_num = self.total_num
        remove_num = prev_num - np.sum(mask)
        if verbose:
            print("Remove %i particles from fluid field" % remove_num)
        if remove_num > 0:
            self.vel = self.vel[mask].copy()
            self.pos = self.pos[mask].copy()
            self.ptype = self.ptype[mask].copy()
            self.fluid_ids = np.argwhere(self.ptype == 1).astype(np.int64).reshape(-1, )
            self.total_num = self.ptype.shape[0]
            self.pres = np.zeros((self.total_num,))

    def get_pos(self, ids=None):
        return self.pos[self.ptype == 1] if ids is None else self.pos[ids]

    def get_vel(self, ids=None):
        return self.vel[self.ptype == 1] if ids is None else self.vel[ids]

    def move_boundary(self, vel: np.ndarray, dt):
        # here, assert the whole system move
        self.pos[:] += vel*dt

    def get_lap_adjacency(self, control_radius, mode, **kwargs):
        # sparse adjacency matrix for laplacian
        support_kw = ['ids_type',
                      'pass_in_ids',
                      'neg_lap',
                      'return_sparse_ele']

        supported_mode = ['laplacian']

        if mode not in supported_mode:
            raise Exception(mode + ' is not a supported mode')
        for key in kwargs.keys():
            if key not in support_kw:
                raise Exception(key + ' is not a supported operation')

        ids_type = kwargs.get('ids_type')
        return_sparse_ele = kwargs.get('return_sparse_ele')

        fluid_ids = self.fluid_ids
        nd_ids = self.non_dum_ids
        nfs_ids = self.nfs_ids

        if ids_type == 'fluid' or ids_type is None:
            ids = fluid_ids
        elif ids_type == 'pass in':
            pass_ids = kwargs.get('pass_in_ids')
            if pass_ids is None:
                raise Exception('There should be ids passed in under pass in ids type')
            ids = pass_ids
        elif ids_type == 'nfs':  # non free surface particles
            ids = nfs_ids
        elif ids_type == 'nds':  # non dummy particles
            ids = nd_ids
        elif ids_type == 'all':  # all particles
            ids = np.arange(0, self.total_num, 1).astype(np.int64)
        else:
            raise Exception("unsupported particle type selecting")

        if control_radius != LAP_RADIUS:
            print('Default radius of nfs_lap mode should be LAP_RADIUS')
        negative = kwargs.get('neg_lap')
        arr_size = len(ids)
        val_arr, idx_arr, tot_size, count = get_laplacian_cuda(self.pos, self.ptype, ids, control_radius)
        if not negative:   # negative to make the matrix positive definite, which is necessary for CG method
            val_arr = - val_arr
        idx, val = arr2adj(idx_arr, val_arr, count, tot_size)

        if return_sparse_ele:
            return idx, val, arr_size

        return sparse_tensor(idx, val, arr_size)

    def fluid2id(self, p_in_cell, fld_ids=None):
        if fld_ids is None:
            fld_ids = self.fluid_ids
        p_in_cell = np.array(p_in_cell, dtype=np.int64)
        return self._index(p_in_cell, fld_ids)

    def get_density(self, control_radius=GRAD_RADIUS, only_fluid=False):

        density = get_density_cuda(self.pos, self.ptype, control_radius)
        if not only_fluid:
            return density
        else:
            return density[self.fluid_ids]

    def get_vel_divergence(self, control_radius=GRAD_RADIUS, only_fluid=False):
        div = get_vel_div_cuda(self.vel, self.pos, self.ptype, control_radius)
        if not only_fluid:
            return div * (-self.N0 * control_radius) / self.rho
        else:
            return div[self.fluid_ids] * (-self.N0 * control_radius) / self.rho

    def update_nfs(self, return_feat=False, clamp_fs=False, only_nfs=False):
        density_ = self.get_density()
        self.nfs_ids = np.argwhere(density_ > 0.97*self.N0).astype(np.int64).reshape(-1,)

        if clamp_fs:
            density = (np.ones_like(self.ptype, dtype=np.float32) * self.N0).reshape(-1,)
            density[self.nfs_ids] = density_[self.nfs_ids]
        else:
            density = density_

        if return_feat:
            return density / self.N0

        if only_nfs:
            return density[self.nfs_ids]
        return density

    def predict(self, cache_train_feat=False):
        Adj_lap = self.get_lap_adjacency(LAP_RADIUS, mode="laplacian")

        with torch.no_grad():
            # ==============apply body force================
            visc = self.nu * Laplacian(self.vel[self.fluid_ids], Adj_lap, self.N0_lap, self.lam).numpy()

            body_force = np.array([0, 0, G])
            bf_acc = visc + body_force
            vel_bf_prev = self.vel.copy()
            pos_bf_prev = self.pos.copy()
            self.vel[self.fluid_ids] += bf_acc * self.DT
            self.pos[self.fluid_ids] += self.vel[self.fluid_ids] * self.DT

            if cache_train_feat:
                assert len(self.cache) == 0
                self.cache['fluid_idx'] = self.fluid_ids
                bf_pic = self.get_nbr(LAP_RADIUS, return_pic=True)
                bf_train_feat = {'pos': pos_bf_prev[bf_pic],
                                 'vel': vel_bf_prev[bf_pic],
                                 'acc': bf_acc,
                                 'in_cell_idx': bf_pic}
                self.cache['bf'] = bf_train_feat

                col_pic = self.get_nbr(COL_RADIUS, return_pic=True)

                col_train_feat = {'pos': self.pos[col_pic],
                                  'vel_prev': self.vel[col_pic],
                                  'in_cell_idx': col_pic,
                                  }

            vel_cu, pos_cu = get_collision_cuda(self.vel, self.pos, self.ptype, COL_RADIUS)
            self.vel = vel_cu
            self.pos = pos_cu

            if cache_train_feat:
                col_train_feat['vel_after'] = self.vel[col_pic]
                self.cache['col'] = col_train_feat

            density = self.update_nfs()
            div = self.get_vel_divergence()
        source_term = -0.80 * (div[self.nfs_ids] / self.DT) \
                      + 0.20 * (density[self.nfs_ids] - self.N0) / (self.DT**2 * self.N0)

        if cache_train_feat:
            dns_feat = self.update_nfs(return_feat=True)
            prs_train_feat = {
                              'pos': self.pos[self.nfs_ids],
                              'dns': dns_feat[self.nfs_ids],
                              'vel': self.vel[self.nfs_ids],
                              'nfs_idx': self.nfs_ids}
            self.cache['prs'] = prs_train_feat
        return source_term

    def predict_with_timing(self):
        log = {}
        visc_nbr_start = time.time()
        Adj_lap = self.get_lap_adjacency(LAP_RADIUS, mode="laplacian")
        visc_nbr_end = time.time()
        log['adv_nbr'] = visc_nbr_end - visc_nbr_start
        with torch.no_grad():
            # ==============apply body force================
            visc_calc_start = time.time()
            visc = self.nu * Laplacian(self.vel[self.fluid_ids], Adj_lap, self.N0_lap, self.lam).numpy()

            body_force = np.array([0, 0, G])
            bf_acc = visc + body_force
            self.vel[self.fluid_ids] += bf_acc * self.DT
            self.pos[self.fluid_ids] += self.vel[self.fluid_ids] * self.DT
            visc_calc_end = time.time()
            log['adv_calc'] = visc_calc_end - visc_calc_start

        col_start = time.time()
        vel_cu, pos_cu = get_collision_cuda(self.vel, self.pos, self.ptype, COL_RADIUS)
        col_end = time.time()
        log['col'] = col_end - col_start
        self.vel = vel_cu
        self.pos = pos_cu

        source_start = time.time()
        density = self.update_nfs()
        div = self.get_vel_divergence()
        source_term = -0.80 * (div[self.nfs_ids] / self.DT) \
                  + 0.20 * (density[self.nfs_ids] - self.N0) / (self.DT**2 * self.N0)
        source_end = time.time()
        log['prs_source'] = source_end - source_start
        return source_term, log

    def predict_with_dynamic_bound(self, vel, dt):
        Adj_lap = self.get_lap_adjacency(LAP_RADIUS, mode="laplacian")
        with torch.no_grad():
            # ==============apply body force================
            visc = self.nu * Laplacian(self.vel[self.fluid_ids], Adj_lap, self.N0_lap, self.lam).numpy()
            body_force = np.array([0, 0, G])
            bf_acc = visc + body_force
            self.vel[self.fluid_ids] += bf_acc * self.DT
            self.pos[self.fluid_ids] += self.vel[self.fluid_ids] * self.DT

        mask = np.ones((self.pos.shape[0], ))
        mask[self.fluid_ids] = 0
        self.pos[mask == 1] += vel*dt

        vel_cu, pos_cu = get_collision_cuda(self.vel, self.pos, self.ptype, COL_RADIUS)
        self.vel = vel_cu
        self.pos = pos_cu

        density = self.update_nfs()
        div = self.get_vel_divergence()
        source_term = -0.80 * (div[self.nfs_ids] / self.DT) \
                  + 0.20 * (density[self.nfs_ids] - self.N0) / (self.DT**2 * self.N0)
        return source_term

    def move_bound(self, vel, dt):
        mask = np.ones((self.pos.shape[0],))
        mask[self.fluid_ids] = 0
        self.pos[mask == 1] += vel * dt

    def get_prs_feat(self):
        dns_feat = self.update_nfs(return_feat=True)
        prs_feat = {
            'pos': self.pos[self.nfs_ids],
            'dns': dns_feat[self.nfs_ids],
            'vel': self.vel[self.nfs_ids],
            'nfs_idx': self.nfs_ids}
        return prs_feat

    def collision(self):
        vel_cu, pos_cu = get_collision_cuda(self.vel, self.pos, self.ptype, COL_RADIUS)
        self.vel = vel_cu
        self.pos = pos_cu

    def correct(self, pred_pres, cache_train_feat=False):
        self.pres = np.zeros_like(self.pres)
        self.pres[self.nfs_ids] = np.where(pred_pres > 0, pred_pres, 0).reshape(-1, )
        pres_grad = get_gradient_cuda(self.pres, self.pos, self.ptype, self.fluid_ids, GRAD_RADIUS)

        acc = -1./self.rho * pres_grad * 3./self.N0
        dt = self.DT
        self.pos[self.fluid_ids] += acc * (dt**2)
        self.vel[self.fluid_ids] += acc * dt
        if cache_train_feat:
            assert 'prs' in self.cache.keys()
            self.cache['prs']['prs'] = self.pres[self.nfs_ids]

    def weak_compress_pressure(self):
        density = self.update_nfs(only_nfs=True)
        pres = 22.0**2 / self.N0 * (density > self.N0) * (density - self.N0) * self.rho
        return pres

    def advect(self, ext_force):
        self.vel[self.fluid_ids] += ext_force * self.DT
        self.pos[self.fluid_ids] += self.vel[self.fluid_ids] * self.DT

    def col_correct(self, col_vel):
        self.pos[self.fluid_ids] += (col_vel - self.vel[self.fluid_ids]) * self.DT
        self.vel[self.fluid_ids] = col_vel

    def pres_predict(self):
        density_feat = self.update_nfs(return_feat=True)
        vel_feat = self.vel
        return density_feat, vel_feat

    def set_pos(self, pos, ids=None, update_nfs=True):
        ids = self.fluid_ids if ids is None else ids
        pos_ = pos.copy()
        self.pos[ids] = pos_
        if update_nfs:
            self.update_nfs()

    def set_vel(self, vel, ids=None):
        ids = self.fluid_ids if ids is None else ids
        vel_ = vel.copy()
        self.vel[ids] = vel_

    def set_state(self, pos, vel, ids=None):
        self.set_pos(pos, ids)
        self.set_vel(vel, ids)

    def write_info(self, output_path, time_step,
                   x_boundary=None, y_boundary=None, z_boundary=None, write_status=False):

        pos = self.pos
        vel = self.vel
        mask = self.fluid_ids

        def mask_out_boundary(mask, bound):
            mask = np.intersect1d(mask, np.argwhere(pos[:, 0]) >= bound[0])
            mask = np.intersect1d(mask, np.argwhere(pos[:, 0]) <= bound[1])
            return mask

        if x_boundary is not None:
            mask = mask_out_boundary(mask, x_boundary)
        if y_boundary is not None:
            mask = mask_out_boundary(mask, y_boundary)
        if z_boundary is not None:
            mask = mask_out_boundary(mask, z_boundary)

        info = {'pos': pos[mask], 'vel': vel[mask]}
        if write_status:
            density = self.get_density(GRAD_RADIUS)
            pres = self.pres
            vel = self.vel
            vel_div = self.get_vel_divergence(GRAD_RADIUS)
            info['dns'] = density[mask]
            info['prs'] = pres[mask]
            info['vel'] = vel[mask]
            info['vel_div'] = vel_div[mask]
        np.savez_compressed(output_path + 'sim_info' + str(time_step) + '.npz', **info)

    def write_data(self, output_path, time_step, dump_only=None):
        with open(output_path + 'data' + str(time_step) + '.pkl', 'wb') as pickle_file:
            if dump_only is None:
                pickle.dump(self.cache, pickle_file)
            else:
                assert self.cache.get(dump_only) is not None
                self.cache[dump_only]['fluid_idx'] = self.cache['fluid_idx']
                pickle.dump(self.cache[dump_only], pickle_file)
        del self.cache
        self.cache = {}

    @staticmethod
    @nb.njit
    def _find_min_max(val, h):
        min_ = 1e10
        max_ = -1e10
        for i in range(val.shape[0]):
            if min_ > val[i]:
                min_ = val[i]
            if max_ < val[i]:
                max_ = val[i]
        return (min_ - 2 * h), (max_ + 2 * h)

    def gen_sparse_map(self, ids):
        arr_size = len(ids)
        local_ids = np.arange(arr_size).astype(dtype=np.int64)
        sparse_map = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        sparse_map = self._gen_sparse_map(sparse_map, ids, local_ids)
        return sparse_map

    @staticmethod
    @nb.njit
    def _gen_sparse_map(sp_map, global_ids, local_ids):
        for i in range(len(global_ids)):
            sp_map[global_ids[i]] = local_ids[i]
        return sp_map

    @staticmethod
    @nb.njit
    def _index(ids1, ids2):
        """ids1: sparse encoding,
           ids2: fluid ids"""
        idx_map = []
        cursor = 0
        for i, id1 in enumerate(ids1):
            id2 = ids2[cursor]
            if id2 == id1:
                idx_map.append(i)
                cursor += 1
            if cursor == len(ids2):
                break
        return idx_map

    @staticmethod
    def find_idx(to_find_arr, key):
        """
        Copied from
        https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
        """
        xsorted_idx = np.argsort(to_find_arr)
        ypos = np.searchsorted(to_find_arr[xsorted_idx], key)
        indices = xsorted_idx[ypos]
        return indices









