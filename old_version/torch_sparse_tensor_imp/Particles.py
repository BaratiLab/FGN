import numpy as np
import torch
import numba as nb
import open3d as o3d
from numba import types, deferred_type
from numba.typed import Dict, List

from Constants import *
from Operator import *
from cuHelper import *
import time

I = 16  # 2^I is for hashing function
Large = 2**I


@nb.njit
def arr2adj(idx_arr, val_arr, tot_size):
    i = np.zeros((tot_size, 2), dtype=np.int64)
    v = np.zeros((tot_size,), dtype=np.float32)
    count = 0
    for row in range(idx_arr.shape[0]):
        neighbor_num = idx_arr[row, 0]
        for j in range(1, neighbor_num+1):
            col = idx_arr[row, j]
            val = val_arr[row, j]
            i[count] = np.array([row, col])
            v[count] = val
            count += 1
    assert(count == tot_size)
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
        self.cell_cache = None

    def init_params(self, show_info=True, pass_in=None):
        if pass_in is None:
            # calculate the constant density of fluid field
            self.N0 = np.max(self.get_density(only_fluid=True))

            # calculate the density parameter for Laplacian operator
            self.N0_lap = np.max(self.get_density(control_radius=LAP_RADIUS, only_fluid=True))

            Adj_sqd = self.get_adjacency(LAP_RADIUS, 'square distance', ids_type='fluid')
            self.lam = torch.max(SumOp.forward(Adj_sqd, to_numpy=False)).item()/self.N0_lap

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

    def cell_sort(self, radius=GRAD_RADIUS):
        fluid_pos = self.get_pos()
        l0 = 1.05*radius
        x_min, x_max = self._find_min_max(fluid_pos[:, 0], l0)
        y_min, y_max = self._find_min_max(fluid_pos[:, 1], l0)
        z_min, z_max = self._find_min_max(fluid_pos[:, 2], l0)

        cells = Cells(self.pos,
                      x_max, x_min,
                      y_max, y_min,
                      z_max, z_min,
                      l0)
        cells.add_particles(self.total_num)
        # cells.test()
        return cells

    def get_pos(self, ids=None):
        return self.pos[self.ptype == 1] if ids is None else self.pos[ids]

    def get_vel(self, ids=None):
        return self.vel[self.ptype == 1] if ids is None else self.vel[ids]

    def get_particle_in_cell(self, control_radius, cells=None):
        cells = self.cell_sort(control_radius) if cells is None else cells
        ids = self.fluid_ids
        p_in_cell = cells.get_neighbors(ids, control_radius)
        p_in_cell = list(set(p_in_cell))
        p_in_cell.sort()
        return np.array(p_in_cell).astype(np.int64)

    def get_adjacency(self, control_radius, mode, **kwargs):

        support_kw = ['ids_type',
                      'pass_in_ids',
                      'grad_val',
                      'neg_lap',
                      'return_sparse_ele']

        supported_mode = ['collision',
                          'laplacian',
                          'gradient',
                          'square distance',
                          'gcn average',
                          'collision feature',
                          'gns feature',
                          'edge idx']

        if mode not in supported_mode:
            raise Exception(mode + ' is not a supported mode')
        for key in kwargs.keys():
            if key not in support_kw:
                raise Exception(key + ' is not a supported operation')
        grad_val = kwargs.get('grad_val')
        if mode == 'gradient' and grad_val is None:
            raise Exception('Gradient value is missed')

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
        elif ids_type == 'p in cell':   # particle in cell
            cells = self.cell_sort(control_radius)
            ids = self.get_particle_in_cell(control_radius, cells)
        else:
            raise Exception("unsupported particle type selecting")

        arr_size = self.total_num

        if mode == 'collision feature':
            edge_feat_cu, edge_idx_cu, pic = get_collision_feature_cuda(self.vel, self.pos, self.ptype, COL_RADIUS)
            edge_feat = torch.from_numpy(edge_feat_cu)
            edge_idx = torch.from_numpy(edge_idx_cu)
            arr_size = len(pic)
            if return_sparse_ele:
                return edge_feat_cu, edge_idx_cu, pic, arr_size
            return edge_feat, edge_idx, pic, arr_size
        elif mode == 'collision':
            cells = self.cell_sort(control_radius)
            arr_size = len(ids)
            sparse_map = self.gen_sparse_map(ids)
            val_arr, idx_arr = get_all_collision_neighbor(cells, ids, self.vel, sparse_map, self.ptype, COL_RADIUS)
            tot_size = np.sum(idx_arr[:, 0])
            idx, val = arr2adj(idx_arr, val_arr, tot_size)

            if return_sparse_ele:
                return idx, val, arr_size
            return sparse_tensor(idx, val, arr_size), ids
        elif mode == 'gns feature':
            edge_feat_cu, edge_idx_cu, pic = get_gns_feature_cuda(self.pos, self.ptype, control_radius)
            edge_feat = torch.from_numpy(edge_feat_cu)
            edge_idx = torch.from_numpy(edge_idx_cu)
            arr_size = len(pic)
            if return_sparse_ele:
                return edge_feat_cu, edge_idx_cu, pic, arr_size
            return edge_feat, edge_idx, pic, arr_size
        elif mode == 'edge idx':
            edge_idx_cu = get_edge_idx_cuda(ids, self.pos, self.ptype, control_radius)
            edge_idx = torch.from_numpy(edge_idx_cu)
            arr_size = len(ids)
            if return_sparse_ele:
                return edge_idx_cu, arr_size
            return edge_idx, arr_size

        if mode == 'square distance':
            cells = self.cell_sort(control_radius)
            val, idx = cells.get_squared_distance_neighbor(ids, control_radius)
        elif mode == 'laplacian':
            if control_radius != LAP_RADIUS:
                print('Default radius of nfs_lap mode should be LAP_RADIUS')
            negative = kwargs.get('neg_lap')
            arr_size = len(ids)
            val_arr, idx_arr, tot_size = get_laplacian_cuda(self.pos, self.ptype, ids, control_radius)
            if not negative:   # negative to make the matrix positive definite, which is necessary for CG method
                val_arr = - val_arr
            idx, val = arr2adj(idx_arr, val_arr, tot_size)

            if return_sparse_ele:
                return idx, val, arr_size

        elif mode == 'gcn average':
            arr_size = len(ids)
            cu_val, cu_idx, cu_tot_size = get_gcn_average_cuda(self.pos, self.ptype, ids, control_radius)
            idx_cu, val_cu = arr2adj(cu_idx, cu_val, cu_tot_size)
            idx, val = idx_cu, val_cu

            if return_sparse_ele:
                return idx, val, arr_size
        else:
            raise Exception("unsupported operation mode")

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

    def predict(self, return_train_feat=False, return_bc_data=False):
        Adj_lap = self.get_adjacency(LAP_RADIUS, mode="laplacian")

        with torch.no_grad():
            # ==============apply body force================
            visc = self.nu * Laplacian(self.vel[self.fluid_ids], Adj_lap, self.N0_lap, self.lam).numpy()

            body_force = np.array([0, 0, G])

            if return_train_feat:
                vel_bf_prev = self.vel[self.fluid_ids].copy()
                gcn_idx, gcn_val, arr_size = self.get_adjacency(LAP_RADIUS, mode='gcn average', return_sparse_ele=True)

            bf_acc = visc + body_force
            self.vel[self.fluid_ids] += bf_acc * self.DT
            self.pos[self.fluid_ids] += self.vel[self.fluid_ids] * self.DT

            if return_train_feat:
                bf_train_feat = {'gcn_idx': gcn_idx, 'gcn_val': gcn_val, 'arr_size': arr_size,
                                 'vel_prev': vel_bf_prev,
                                 'acc': bf_acc}
                edge_attr, edge_idx, p_in_cell, edge_size = self.get_adjacency(COL_RADIUS, mode='collision feature')
                vel_col_prev = self.vel[p_in_cell].copy()
                col_train_feat = {'edge_idx': edge_idx, 'edge_attr': edge_attr, 'edge_size': edge_size,
                                  'vel_prev': vel_col_prev,
                                  'fld_ids': self.fluid2id(p_in_cell, self.fluid_ids)
                                  }

            vel_cu, pos_cu = get_collision_cuda(self.vel, self.pos, self.ptype, COL_RADIUS)
            self.vel = vel_cu
            self.pos = pos_cu

            if return_train_feat:
                vel_col_gt = self.vel[p_in_cell].copy()
                col_train_feat['vel_gt'] = vel_col_gt

            density = self.update_nfs()
            div = self.get_vel_divergence()
        source_term = -0.80 * (div[self.nfs_ids] / self.DT) \
                      + 0.20 * (density[self.nfs_ids] - self.N0) / (self.DT**2 * self.N0)

        if return_train_feat:
            dns_feat = self.update_nfs(return_feat=True)[self.nfs_ids]
            vel_feat = self.vel[self.nfs_ids]
            gcn_idx, gcn_val, arr_size = self.get_adjacency(LAP_RADIUS,
                                                            mode='gcn average',
                                                            ids_type='nfs',
                                                            return_sparse_ele=True)
            prs_train_feat = {'gcn_idx': gcn_idx, 'gcn_val': gcn_val, 'arr_size': arr_size,
                              'dns_feat': dns_feat,
                              'vel_feat': vel_feat}
            return bf_train_feat, col_train_feat, prs_train_feat, source_term
        elif return_bc_data:
            dns_feat = self.update_nfs(return_feat=True)[self.nfs_ids]
            vel_feat = self.vel[self.nfs_ids]
            return dns_feat, vel_feat, source_term
        else:
            return source_term

    def correct(self, pred_pres, return_train_feat=False):
        self.pres = np.zeros_like(self.pres)
        self.pres[self.nfs_ids] = np.where(pred_pres>0, pred_pres, 0).reshape(-1, )
        pres_grad = get_gradient_cuda(self.pres, self.pos, self.ptype, self.fluid_ids, GRAD_RADIUS)

        acc = -1./self.rho * pres_grad * 3./self.N0
        dt = self.DT
        self.pos[self.fluid_ids] += acc * (dt**2)
        self.vel[self.fluid_ids] += acc * dt
        if return_train_feat:
            return self.pres[self.nfs_ids]

    def weak_compress_pressure(self):
        density = self.update_nfs(only_nfs=True)
        pres = 22.0**2 / self.N0 * (density > self.N0) * (density - self.N0) * self.rho
        return pres

    def get_density_gradient(self):
        density = self.get_density(control_radius=1.05*BASE_RADIUS)
        N0_wall = 0.31
        N0_obs = 0.83
        print(N0_wall)
        wall_ids = self.fluid_ids   # use fluid particle to mimic wall particles
        density_grad = get_gradient_cuda(density, self.pos, self.ptype, wall_ids, 1.9*BASE_RADIUS)

        wall = self.pos[wall_ids]
        wall_density = density[wall_ids]
        mask1 = np.logical_and(wall_density > 0.80 * N0_wall, wall_density < 0.95 * N0_wall)
        mask2 = np.logical_and(wall_density > 0.70 * N0_obs, wall_density < 0.90 * N0_obs)
        mask = np.logical_or(mask1, mask2)

        def normalize(vector3d):
            return vector3d / np.sqrt(np.sum(vector3d**2, axis=1)).reshape(-1, 1)

        density_grad = density_grad[mask]
        density_grad_ = normalize(density_grad)
        normalized_density_grad = np.where(np.abs(density_grad_) < 0.1, 0, density_grad_)
        normalized_density_grad = normalize(normalized_density_grad)

        wall_surface = wall[mask]
        wall_normals = -normalized_density_grad
        return wall_surface, wall_normals

    def get_normals(self):
        if self.fluid_ids.shape[0] == 0:
            raise Exception("Currently do not support scene without fluid particles transformation")
        return self.get_density_gradient()

    def col_predict(self, ext_force):

        self.vel[self.fluid_ids] += ext_force * self.DT
        self.pos[self.fluid_ids] += self.vel[self.fluid_ids] * self.DT

        return self.vel.copy()

    def col_correct(self, col_vel, ids):
        vel_temp = np.zeros_like(self.vel)
        fld2ids = self.fluid2id(ids, self.fluid_ids)
        if len(fld2ids) != len(self.fluid_ids):
            print(len(ids))
            print(len(self.fluid_ids))
            print(len(fld2ids))
        vel_temp[self.fluid_ids] = col_vel[fld2ids]
        self.pos[self.fluid_ids] += (vel_temp[self.fluid_ids] - self.vel[self.fluid_ids]) * self.DT
        self.vel[self.fluid_ids] = vel_temp[self.fluid_ids]

    def pres_predict(self):
        density_feat = self.update_nfs(return_feat=True)[self.nfs_ids]
        vel_feat = self.vel[self.nfs_ids]
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

        info = {'pos': pos[mask]}
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









