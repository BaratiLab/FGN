import numba as nb
import numpy as np
from numba import cuda
from Constants import COL_COEF, DT, BASE_RADIUS
import math
import cupy as cp


THREADS = 256


@cuda.jit(device=True)
def weight(r, re):
    if r < 1e-8:
        return 0.0
    else:
        return (re/r) - 1.


@cuda.jit('float32(float32[:, :], int32, int32)', device=True)
def distance(pos, i, j):
    rx = pos[i, 0] - pos[j, 0]
    ry = pos[i, 1] - pos[j, 1]
    rz = pos[i, 2] - pos[j, 2]
    return math.sqrt(rx**2 + ry**2 + rz**2)


@cuda.jit('boolean(float32[:, :], float32[:], int32)', device=True)
def check_in_range(pos, boundary, i):
    rx = pos[i, 0]
    ry = pos[i, 1]
    rz = pos[i, 2]
    max_x, min_x, max_y, min_y, max_z, min_z = \
        boundary[0], boundary[1], boundary[2], boundary[3], boundary[4], boundary[5]
    return (min_x < rx < max_x and
            min_y < ry < max_y and
            min_z < rz < max_z)


@cuda.jit(device=True)
def __get_neighbor_min__(val, pos, cell_fst, cell_next, boundary,
                         i, control_radius, nx, nxy):
    cell_len = control_radius * 1.05
    min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
    rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
    ix = int((rx - min_x) / cell_len) + 1
    iy = int((ry - min_y) / cell_len) + 1
    iz = int((rz - min_z) / cell_len) + 1
    val_min = val[i]
    for jz in range(iz - 1, iz + 2):
        for jy in range(iy - 1, iy + 2):
            for jx in range(ix - 1, ix + 2):
                cell_idx_j = jz * nxy + jy * nx + jx
                j = cell_fst[cell_idx_j]
                if j == -1:
                    continue
                while True:
                    dist = distance(pos, i, j)
                    if dist < control_radius:
                        if val[j] < val_min:
                            val_min = val[j]
                    j = cell_next[j]
                    if j == -1:
                        break
    return val_min


@cuda.jit
def cu_init_int_array(arr, param, tot_num):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < tot_num:
        arr[i] = param


@cuda.jit
def cu_init_float_array(arr, param, tot_num):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < tot_num:
        arr[i] = param


@nb.njit
def __check_in_range__(pos, boundary, i):
    rx = pos[i, 0]
    ry = pos[i, 1]
    rz = pos[i, 2]
    max_x, min_x, max_y, min_y, max_z, min_z = \
        boundary[0], boundary[1], boundary[2], boundary[3], boundary[4], boundary[5]
    return (min_x < rx < max_x and
            min_y < ry < max_y and
            min_z < rz < max_z)


@nb.njit
def cell_sort(pos, boundary, control_radius, nx, nxy, nxyz, tot_num):
    cell_fst = -np.ones((nxyz, ), dtype=np.int32)
    cell_lst = -np.ones((nxyz, ), dtype=np.int32)
    cell_next = -np.ones((tot_num, ), dtype=np.int32)

    for i in range(tot_num):
        if not __check_in_range__(pos, boundary, i):
            continue
        cell_len = control_radius * 1.05
        min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
        rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
        ix = int((rx - min_x) / cell_len) + 1
        iy = int((ry - min_y) / cell_len) + 1
        iz = int((rz - min_z) / cell_len) + 1
        cell_idx = iz * nxy + iy * nx + ix
        j = cell_lst[cell_idx]
        cell_lst[cell_idx] = i
        if j == -1:
            cell_fst[cell_idx] = i
        else:
            cell_next[j] = i
    return cell_fst, cell_next


@nb.njit(parallel=True)
def get_p_in_cell(pos, ids, cell_fst, cell_next, min_x, min_y, min_z, nxy, nx, control_radius):
    p_in_cell = -np.ones((pos.shape[0], ), dtype=np.int32)
    cell_len = control_radius * 1.05
    for i in nb.prange(ids.shape[0]):
        idx = ids[i]
        p_in_cell[idx] = 1
        ix = int((pos[idx, 0] - min_x) / cell_len) + 1
        iy = int((pos[idx, 1] - min_y) / cell_len) + 1
        iz = int((pos[idx, 2] - min_z) / cell_len) + 1
        for jz in range(iz - 1, iz + 2):
            for jy in range(iy - 1, iy + 2):
                for jx in range(ix - 1, ix + 2):
                        cell_idx_j = jz * nxy + jy * nx + jx
                        j = cell_fst[cell_idx_j]
                        if j == -1:
                            continue
                        while True:
                            rx = pos[j, 0] - pos[idx, 0]
                            ry = pos[j, 1] - pos[idx, 1]
                            rz = pos[j, 2] - pos[idx, 2]
                            dist = math.sqrt(rx**2 + ry**2 + rz**2)
                            if dist < control_radius:
                                p_in_cell[j] = 1
                            j = cell_next[j]
                            if j == -1:
                                break
    return p_in_cell


@nb.njit
def stack_col_feature(attr_arr, idx_arr, tot_size):
    v = np.zeros((tot_size, 6), dtype=np.float32)
    i = np.zeros((tot_size, 2), dtype=np.int64)
    count = 0
    for row in range(idx_arr.shape[0]):
        neighbor_num = idx_arr[row, 0]
        for j in range(1, neighbor_num + 1):
            col = idx_arr[row, j]
            i[count] = np.array([row, col])
            for n in range(6):
                v[count, n] = attr_arr[row, 6*j + n]
            count += 1
    assert (count == tot_size)
    return i, v


@nb.njit
def stack_edge_idx(idx_arr, tot_size):
    i = np.zeros((tot_size, 2), dtype=np.int64)
    count = 0
    for row in range(idx_arr.shape[0]):
        neighbor_num = idx_arr[row, 0]
        for j in range(1, neighbor_num + 1):
            col = idx_arr[row, j]
            i[count] = np.array([row, col])
            count += 1
    assert (count == tot_size)
    return i


@nb.njit
def stack_gns_feature(attr_arr, idx_arr, tot_size):
    v = np.zeros((tot_size, 4), dtype=np.float32)
    i = np.zeros((tot_size, 2), dtype=np.int64)
    count = 0
    for row in range(idx_arr.shape[0]):
        neighbor_num = idx_arr[row, 0]
        for j in range(1, neighbor_num + 1):
            col = idx_arr[row, j]
            i[count] = np.array([row, col])
            for n in range(4):
                v[count, n] = attr_arr[row, 4*j + n]
            count += 1
    assert (count == tot_size)
    return i, v


@cuda.jit
def cu_get_laplacian(lap_val, lap_idx, i2row, pos, cell_fst, cell_next, boundary,
                  control_radius, nx, nxy, tot_num):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < tot_num :
        row = i2row[i]
        if check_in_range(pos, boundary, i) and row != -1:
            # identify the cell particle locate in
            cell_len = control_radius * 1.05
            min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
            rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
            ix = int((rx - min_x) / cell_len) + 1
            iy = int((ry - min_y) / cell_len) + 1
            iz = int((rz - min_z) / cell_len) + 1

            center_sum = 0.
            for jz in range(iz - 1, iz + 2):
                for jy in range(iy - 1, iy + 2):
                    for jx in range(ix - 1, ix + 2):

                            cell_idx_j = jz * nxy + jy * nx + jx
                            j = cell_fst[cell_idx_j]
                            if j == -1:
                                continue
                            while True:
                                dist = distance(pos, i, j)
                                if dist < control_radius  and j != i:
                                    w = weight(dist, control_radius)
                                    center_sum += w
                                    col = i2row[j]
                                    if col != -1:
                                        lap_idx[row, 0] += 1
                                        cursor = lap_idx[row, 0]
                                        lap_val[row, cursor] = -w
                                        lap_idx[row, cursor] = col
                                j = cell_next[j]
                                if j == -1:
                                    break
            lap_idx[row, 0] += 1
            cursor = lap_idx[row, 0]
            lap_val[row, cursor] = center_sum
            lap_idx[row, cursor] = row


@cuda.jit
def cu_get_density(density, ptype, pos, cell_fst, cell_next, boundary,
                  control_radius, nx, nxy, tot_num):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < tot_num :
        if check_in_range(pos, boundary, i) and ptype[i] != 3:
        # identify the cell particle locate in
            cell_len = control_radius * 1.05
            min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
            rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
            ix = int((rx - min_x) / cell_len) + 1
            iy = int((ry - min_y) / cell_len) + 1
            iz = int((rz - min_z) / cell_len) + 1

            center_sum = 0.
            for jz in range(iz - 1, iz + 2):
                for jy in range(iy - 1, iy + 2):
                    for jx in range(ix - 1, ix + 2):
                            cell_idx_j = jz * nxy + jy * nx + jx
                            j = cell_fst[cell_idx_j]
                            if j == -1:
                                continue
                            while True:
                                rx = pos[i, 0] - pos[j, 0]
                                ry = pos[i, 1] - pos[j, 1]
                                rz = pos[i, 2] - pos[j, 2]
                                dist = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
                                if dist < control_radius and j != i:
                                    w = weight(dist, control_radius)
                                    center_sum += w
                                j = cell_next[j]
                                if j == -1:
                                    break
            density[i] = center_sum


@cuda.jit
def cu_get_gradient(gradient, grad_val, i2row, pos, cell_fst, cell_next, boundary,
                  control_radius, nx, nxy, tot_num):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < tot_num:
        row = i2row[i]
        if check_in_range(pos, boundary, i) and row != -1:
            # identify the cell particle locate in
            cell_len = control_radius * 1.05
            min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
            rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
            ix = int((rx - min_x) / cell_len) + 1
            iy = int((ry - min_y) / cell_len) + 1
            iz = int((rz - min_z) / cell_len) + 1

            val_min = grad_val[i]
            for jz in range(iz - 1, iz + 2):
                for jy in range(iy - 1, iy + 2):
                    for jx in range(ix - 1, ix + 2):
                        cell_idx_j = jz * nxy + jy * nx + jx
                        j = cell_fst[cell_idx_j]
                        if j == -1:
                            continue
                        while True:
                            dist = distance(pos, i, j)
                            if dist < control_radius:
                                neigh_val = grad_val[j]
                                if val_min > neigh_val:
                                    val_min = neigh_val
                            j = cell_next[j]
                            if j == -1:
                                break

            for jz in range(iz - 1, iz + 2):
                for jy in range(iy - 1, iy + 2):
                    for jx in range(ix - 1, ix + 2):
                            cell_idx_j = jz * nxy + jy * nx + jx
                            j = cell_fst[cell_idx_j]
                            if j == -1:
                                continue
                            while True:
                                r_vec_x = pos[j, 0] - rx
                                r_vec_y = pos[j, 1] - ry
                                r_vec_z = pos[j, 2] - rz
                                r_2 = r_vec_x ** 2 + r_vec_y ** 2 + r_vec_z ** 2
                                dist = math.sqrt(r_2)
                                if dist < control_radius and j != i:
                                    w = weight(dist, control_radius)
                                    val_diff = grad_val[j] - val_min
                                    w *= val_diff / r_2

                                    gradient[3*row] += w * r_vec_x
                                    gradient[3*row + 1] += w * r_vec_y
                                    gradient[3*row + 2] += w * r_vec_z
                                j = cell_next[j]
                                if j == -1:
                                    break


@cuda.jit
def cu_get_collision(vel, pos, ptype, cell_fst, cell_next, boundary,
                  control_radius, nx, nxy, tot_num):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < tot_num:
        if check_in_range(pos, boundary, i) and ptype[i] == 1:
            # identify the cell particle locate in
            cell_len = control_radius * 1.05
            min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
            rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
            vx, vy, vz = vel[i, 0], vel[i, 1], vel[i, 2]
            vx_temp, vy_temp, vz_temp = vx, vy, vz
            ix = int((rx - min_x) / cell_len) + 1
            iy = int((ry - min_y) / cell_len) + 1
            iz = int((rz - min_z) / cell_len) + 1

            for jz in range(iz - 1, iz + 2):
                for jy in range(iy - 1, iy + 2):
                    for jx in range(ix - 1, ix + 2):
                        cell_idx_j = jz * nxy + jy * nx + jx
                        j = cell_fst[cell_idx_j]
                        if j == -1:
                            continue
                        while True:
                            if j != i and ptype[j] != 3:
                                r_vec_x = pos[j, 0] - rx
                                r_vec_y = pos[j, 1] - ry
                                r_vec_z = pos[j, 2] - rz
                                r_2 = r_vec_x ** 2 + r_vec_y ** 2 + r_vec_z ** 2
                                dist = math.sqrt(r_2)
                                if dist < control_radius:
                                    fdt = (vx - vel[j, 0]) * r_vec_x +\
                                          (vy - vel[j, 1]) * r_vec_y +\
                                          (vz - vel[j, 2]) * r_vec_z
                                    if fdt > 0.0:
                                        fdt *= (1.0 + COL_COEF) * 0.5 / r_2
                                        vx_temp -= r_vec_x * fdt
                                        vy_temp -= r_vec_y * fdt
                                        vz_temp -= r_vec_z * fdt
                            j = cell_next[j]
                            if j == -1:
                                break
            vel[i, 0] = vx_temp
            vel[i, 1] = vy_temp
            vel[i, 2] = vz_temp
            pos[i, 0] += (vx_temp - vx) * DT
            pos[i, 1] += (vy_temp - vy) * DT
            pos[i, 2] += (vz_temp - vz) * DT


@cuda.jit
def cu_get_gcn_average(gcn_val, gcn_idx, i2row, pos, cell_fst, cell_next, boundary,
                  control_radius, nx, nxy, tot_num):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < tot_num:
        row = i2row[i]
        if check_in_range(pos, boundary, i) and row != -1:
            # identify the cell particle locate in
            cell_len = control_radius * 1.05
            min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
            rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
            ix = int((rx - min_x) / cell_len) + 1
            iy = int((ry - min_y) / cell_len) + 1
            iz = int((rz - min_z) / cell_len) + 1

            center_sum = 0.
            for jz in range(iz - 1, iz + 2):
                for jy in range(iy - 1, iy + 2):
                    for jx in range(ix - 1, ix + 2):
                            cell_idx_j = jz * nxy + jy * nx + jx
                            j = cell_fst[cell_idx_j]
                            if j == -1:
                                continue
                            while True:
                                dist = distance(pos, i, j)
                                if dist < control_radius and j != i:
                                    w = weight(dist, control_radius)
                                    col = i2row[j]
                                    if col != -1:
                                        center_sum += w
                                        gcn_idx[row, 0] += 1
                                        cursor = gcn_idx[row, 0]
                                        gcn_val[row, cursor] = w
                                        gcn_idx[row, cursor] = col
                                j = cell_next[j]
                                if j == -1:
                                    break
            gcn_idx[row, 0] += 1
            cursor = gcn_idx[row, 0]
            gcn_val[row, cursor] = 1
            gcn_idx[row, cursor] = row
            col_num = cursor
            for n in range(1, col_num):
                gcn_val[row, n] /= center_sum


@cuda.jit
def cu_get_col_feature(edge_attr, edge_idx, vel, pos, i2row, ptype,
                       cell_fst, cell_next, boundary,
                       control_radius, nx, nxy, tot_num):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < tot_num:
        row = i2row[i]
        if check_in_range(pos, boundary, i) and row != -1 and ptype[i] == 1:
            # identify the cell particle locate in
            cell_len = control_radius * 1.05
            min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
            rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
            ix = int((rx - min_x) / cell_len) + 1
            iy = int((ry - min_y) / cell_len) + 1
            iz = int((rz - min_z) / cell_len) + 1

            for jz in range(iz - 1, iz + 2):
                for jy in range(iy - 1, iy + 2):
                    for jx in range(ix - 1, ix + 2):
                        cell_idx_j = jz * nxy + jy * nx + jx
                        j = cell_fst[cell_idx_j]
                        if j == -1:
                            continue
                        while True:
                            r_vec_x = rx - pos[j, 0]
                            r_vec_y = ry - pos[j, 1]
                            r_vec_z = rz - pos[j, 2]
                            r_2 = r_vec_x ** 2 + r_vec_y ** 2 + r_vec_z ** 2
                            dist = math.sqrt(r_2)
                            col = i2row[j]
                            if dist < control_radius and j != i and col != -1:
                                v_vec_x = vel[i, 0] - vel[j, 0]
                                v_vec_y = vel[i, 1] - vel[j, 1]
                                v_vec_z = vel[i, 2] - vel[j, 2]
                                edge_idx[row, 0] += 1
                                cursor = edge_idx[row, 0]
                                edge_attr[row, 6*cursor] = v_vec_x
                                edge_attr[row, 6*cursor + 1] = r_vec_x/dist
                                edge_attr[row, 6*cursor + 2] = v_vec_y
                                edge_attr[row, 6*cursor + 3] = r_vec_y/dist
                                edge_attr[row, 6*cursor + 4] = v_vec_z
                                edge_attr[row, 6*cursor + 5] = r_vec_z/dist
                                edge_idx[row, cursor] = col
                            j = cell_next[j]
                            if j == -1:
                                break


@cuda.jit
def cu_get_gns_feature(edge_attr, edge_idx, pos, i2row, ptype,
                       cell_fst, cell_next, boundary,
                       control_radius, nx, nxy, tot_num):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < tot_num:
        row = i2row[i]
        if check_in_range(pos, boundary, i) and row != -1 and ptype[i] == 1:
            # identify the cell particle locate in
            cell_len = control_radius * 1.05
            min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
            rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
            ix = int((rx - min_x) / cell_len) + 1
            iy = int((ry - min_y) / cell_len) + 1
            iz = int((rz - min_z) / cell_len) + 1

            for jz in range(iz - 1, iz + 2):
                for jy in range(iy - 1, iy + 2):
                    for jx in range(ix - 1, ix + 2):
                        cell_idx_j = jz * nxy + jy * nx + jx
                        j = cell_fst[cell_idx_j]
                        if j == -1:
                            continue
                        while True:
                            r_vec_x = rx - pos[j, 0]
                            r_vec_y = ry - pos[j, 1]
                            r_vec_z = rz - pos[j, 2]
                            r_2 = r_vec_x ** 2 + r_vec_y ** 2 + r_vec_z ** 2
                            dist = math.sqrt(r_2)
                            col = i2row[j]
                            if dist < control_radius and j != i and col != -1:
                                edge_idx[row, 0] += 1
                                cursor = edge_idx[row, 0]
                                edge_attr[row, 4*cursor] = r_vec_x
                                edge_attr[row, 4*cursor + 1] = r_vec_y
                                edge_attr[row, 4*cursor + 2] = r_vec_z
                                edge_attr[row, 4*cursor + 3] = dist
                                edge_idx[row, cursor] = col
                            j = cell_next[j]
                            if j == -1:
                                break


@cuda.jit
def cu_get_divergence(divergence, div_val, ptype,
                      pos, cell_fst, cell_next, boundary,
                      control_radius, nx, nxy, tot_num):
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < tot_num:
            if check_in_range(pos, boundary, i) and ptype[i] == 1:
                # identify the cell particle locate in
                cell_len = control_radius * 1.05
                min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
                rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
                ix = int((rx - min_x) / cell_len) + 1
                iy = int((ry - min_y) / cell_len) + 1
                iz = int((rz - min_z) / cell_len) + 1

                for jz in range(iz - 1, iz + 2):
                    for jy in range(iy - 1, iy + 2):
                        for jx in range(ix - 1, ix + 2):
                            cell_idx_j = jz * nxy + jy * nx + jx
                            j = cell_fst[cell_idx_j]
                            if j == -1:
                                continue
                            while True:
                                r_vec_x = pos[j, 0] - rx
                                r_vec_y = pos[j, 1] - ry
                                r_vec_z = pos[j, 2] - rz
                                r_2 = r_vec_x ** 2 + r_vec_y ** 2 + r_vec_z ** 2
                                dist = math.sqrt(r_2)
                                if dist < control_radius and j != i and ptype[j] != 3:
                                    w = weight(dist, control_radius)
                                    partial_x = r_vec_x * (div_val[j, 0] - div_val[i, 0])
                                    partial_y = r_vec_y * (div_val[j, 1] - div_val[i, 1])
                                    partial_z = r_vec_y * (div_val[j, 2] - div_val[i, 2])
                                    div = w*(partial_x + partial_y + partial_z) / r_2
                                    divergence[i] += div
                                j = cell_next[j]
                                if j == -1:
                                    break


@cuda.jit
def cu_get_edge_idx(edge_idx, ptype, i2row,
                      pos, cell_fst, cell_next, boundary,
                      control_radius, nx, nxy, tot_num):
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < tot_num:
            row = i2row[i]
            if check_in_range(pos, boundary, i) and ptype[i] == 1 and row != -1:
                # identify the cell particle locate in
                cell_len = control_radius * 1.05
                min_x, min_y, min_z = boundary[1], boundary[3], boundary[5]
                rx, ry, rz = pos[i, 0], pos[i, 1], pos[i, 2]
                ix = int((rx - min_x) / cell_len) + 1
                iy = int((ry - min_y) / cell_len) + 1
                iz = int((rz - min_z) / cell_len) + 1

                for jz in range(iz - 1, iz + 2):
                    for jy in range(iy - 1, iy + 2):
                        for jx in range(ix - 1, ix + 2):
                            cell_idx_j = jz * nxy + jy * nx + jx
                            j = cell_fst[cell_idx_j]
                            col = i2row[j]
                            if j == -1:
                                continue
                            while True:
                                r_vec_x = pos[j, 0] - rx
                                r_vec_y = pos[j, 1] - ry
                                r_vec_z = pos[j, 2] - rz
                                r_2 = r_vec_x ** 2 + r_vec_y ** 2 + r_vec_z ** 2
                                dist = math.sqrt(r_2)
                                if dist < control_radius and j != i and col != -1:
                                    edge_idx[row, 0] += 1
                                    cursor = edge_idx[row, 0]
                                    edge_idx[row, cursor] = col
                                j = cell_next[j]
                                if j == -1:
                                    break


@nb.njit
def gen_i2row_map(ids, tot_num):
    i2row = - np.ones((tot_num,), dtype=np.int32)
    row = 0
    for i in range(ids.shape[0]):
        idx = ids[i]
        i2row[idx] = row
        row += 1
    return i2row


@nb.njit
def find_min_max(val, h):
    min_ = 1e10
    max_ = -1e10
    for i in range(val.shape[0]):
        if min_ > val[i]:
            min_ = val[i]
        if max_ < val[i]:
            max_ = val[i]
    return (min_ - 2*h), (max_ + 2*h)


def calc_cell_params(pos, ptype, control_radius):
    cell_len = control_radius * 1.05
    fluid_pos = pos[ptype == 1]
    x_min, x_max = find_min_max(fluid_pos[:, 0], cell_len)
    y_min, y_max = find_min_max(fluid_pos[:, 1], cell_len)
    z_min, z_max = find_min_max(fluid_pos[:, 2], cell_len)

    # boundary position
    bound = np.array(
        [x_max, x_min, y_max, y_min, z_max, z_min], dtype=np.float32)

    # calculate the cell number along different axis
    nx, ny, nz = \
        int((x_max - x_min)/cell_len) + 3, int((y_max - y_min)/cell_len) + 3, int((z_max - z_min)/cell_len) + 3
    nxy, nxyz = nx * ny, nx * ny * nz

    return nx, nxy, nxyz, bound


def get_laplacian_cuda(pos, ptype, ids, control_radius):
    # python host wrapper for cuda kernel function "cu_get_laplacian"
    d_pos = cuda.to_device(pos)
    tot_num = pos.shape[0]
    i2row = gen_i2row_map(ids, tot_num)
    d_i2row = cuda.to_device(i2row)
    threads_pb = (THREADS, 1, 1)
    blocks_pg = (int(tot_num / THREADS) + 1, 1, 1)

    nx, nxy, nxyz, cell_bound = calc_cell_params(pos, ptype, control_radius)
    d_cell_bound = cuda.to_device(cell_bound)

    cell_fst, cell_next = cell_sort(pos, cell_bound, control_radius, nx, nxy, nxyz, tot_num)
    d_cell_fst = cuda.to_device(cell_fst)
    d_cell_next = cuda.to_device(cell_next)

    lap_val = np.zeros((len(ids), 384), dtype=np.float32)
    lap_idx = np.zeros((len(ids), 384), dtype=np.int32)
    d_lap_val = cuda.to_device(lap_val)
    d_lap_idx = cuda.to_device(lap_idx)

    cu_get_laplacian[blocks_pg, threads_pb](d_lap_val, d_lap_idx, d_i2row,
                                            d_pos, d_cell_fst, d_cell_next, d_cell_bound,
                                            control_radius, nx, nxy, tot_num)
    cp_lap_idx = cp.asarray(d_lap_idx[:, 0])
    tot_size = cp.sum(cp_lap_idx).item()
    lap_val = d_lap_val.copy_to_host()
    lap_idx = d_lap_idx.copy_to_host()
    return lap_val, lap_idx, tot_size


def get_density_cuda(pos, ptype, control_radius):
    # python host wrapper for cuda kernel function "cu_get_density"

    d_pos = cuda.to_device(pos)
    d_ptype = cuda.to_device(ptype)
    tot_num = pos.shape[0]
    threads_pb = (THREADS, 1, 1)
    blocks_pg = (int(tot_num / THREADS) + 1, 1, 1)

    nx, nxy, nxyz, cell_bound = calc_cell_params(pos, ptype, control_radius)
    d_cell_bound = cuda.to_device(cell_bound)

    cell_fst, cell_next = cell_sort(pos, cell_bound, control_radius, nx, nxy, nxyz, tot_num)
    d_cell_fst = cuda.to_device(cell_fst)
    d_cell_next = cuda.to_device(cell_next)

    density = np.zeros((tot_num, ), dtype=np.float32)
    d_density = cuda.to_device(density)
    cu_get_density[blocks_pg, threads_pb](d_density, d_ptype, d_pos,
                                          d_cell_fst, d_cell_next, d_cell_bound,
                                          control_radius, nx, nxy, tot_num)

    density = d_density.copy_to_host()
    return density


def get_gradient_cuda(grad_val, pos, ptype, ids, control_radius):
    # python host wrapper for cuda kernel function "cu_get_gradient"

    d_pos = cuda.to_device(pos)
    tot_num = pos.shape[0]
    i2row = gen_i2row_map(ids, tot_num)
    d_i2row = cuda.to_device(i2row)

    threads_pb = (THREADS, 1, 1)
    blocks_pg = (int(tot_num / THREADS) + 1, 1, 1)

    nx, nxy, nxyz, cell_bound = calc_cell_params(pos, ptype, control_radius)
    d_cell_bound = cuda.to_device(cell_bound)

    cell_fst, cell_next = cell_sort(pos, cell_bound, control_radius, nx, nxy, nxyz, tot_num)
    d_cell_fst = cuda.to_device(cell_fst)
    d_cell_next = cuda.to_device(cell_next)

    d_grad_val = cuda.to_device(grad_val)
    gradient = np.zeros((len(ids) * 3), dtype=np.float32)
    d_gradient = cuda.to_device(gradient)
    cu_get_gradient[blocks_pg, threads_pb](d_gradient, d_grad_val, d_i2row, d_pos,
                                          d_cell_fst, d_cell_next, d_cell_bound,
                                          control_radius, nx, nxy, tot_num)
    gradient = d_gradient.copy_to_host().reshape(-1, 3)

    return gradient


def get_collision_cuda(vel, pos, ptype, control_radius):

    d_pos = cuda.to_device(pos)
    d_vel = cuda.to_device(vel)
    d_ptype = cuda.to_device(ptype)
    tot_num = pos.shape[0]

    threads_pb = (THREADS, 1, 1)
    blocks_pg = (int(tot_num / THREADS) + 1, 1, 1)

    nx, nxy, nxyz, cell_bound = calc_cell_params(pos, ptype, control_radius)
    d_cell_bound = cuda.to_device(cell_bound)

    cell_fst, cell_next = cell_sort(pos, cell_bound, control_radius, nx, nxy, nxyz, tot_num)
    d_cell_fst = cuda.to_device(cell_fst)
    d_cell_next = cuda.to_device(cell_next)

    cu_get_collision[blocks_pg, threads_pb](d_vel, d_pos, d_ptype,
                                          d_cell_fst, d_cell_next, d_cell_bound,
                                          control_radius, nx, nxy, tot_num)

    vel = d_vel.copy_to_host()
    pos = d_pos.copy_to_host()
    return vel, pos


def get_gcn_average_cuda(pos, ptype, ids, control_radius):
    # python host wrapper for cuda kernel function "cu_get_gcn_average"
    d_pos = cuda.to_device(pos)
    tot_num = pos.shape[0]
    i2row = gen_i2row_map(ids, tot_num)
    d_i2row = cuda.to_device(i2row)
    threads_pb = (THREADS, 1, 1)
    blocks_pg = (int(tot_num / THREADS) + 1, 1, 1)

    nx, nxy, nxyz, cell_bound = calc_cell_params(pos, ptype, control_radius)
    d_cell_bound = cuda.to_device(cell_bound)

    cell_fst, cell_next = cell_sort(pos, cell_bound, control_radius, nx, nxy, nxyz, tot_num)
    d_cell_fst = cuda.to_device(cell_fst)
    d_cell_next = cuda.to_device(cell_next)

    gcn_val = np.zeros((len(ids), 384), dtype=np.float32)
    gcn_idx = np.zeros((len(ids), 384), dtype=np.int32)
    d_gcn_val = cuda.to_device(gcn_val)
    d_gcn_idx = cuda.to_device(gcn_idx)

    cu_get_gcn_average[blocks_pg, threads_pb](d_gcn_val, d_gcn_idx, d_i2row,
                                            d_pos, d_cell_fst, d_cell_next, d_cell_bound,
                                            control_radius, nx, nxy, tot_num)
    if len(ids) > 1:
        cp_gcn_idx = cp.asarray(d_gcn_idx[:, 0])
        tot_size = cp.sum(cp_gcn_idx).item()
    else:
        tot_size = d_gcn_idx.copy_to_host()[0, 0]
    gcn_val = d_gcn_val.copy_to_host()
    gcn_idx = d_gcn_idx.copy_to_host()
    return gcn_val, gcn_idx, tot_size


def get_collision_feature_cuda(vel, pos, ptype, control_radius):
    d_pos = cuda.to_device(pos)
    d_vel = cuda.to_device(vel)
    d_ptype = cuda.to_device(ptype)
    tot_num = pos.shape[0]

    threads_pb = (THREADS, 1, 1)
    blocks_pg = (int(tot_num / THREADS) + 1, 1, 1)

    nx, nxy, nxyz, cell_bound = calc_cell_params(pos, ptype, control_radius)
    d_cell_bound = cuda.to_device(cell_bound)

    cell_fst, cell_next = cell_sort(pos, cell_bound, control_radius, nx, nxy, nxyz, tot_num)
    fluid_ids = np.argwhere(ptype == 1).reshape(-1,)
    min_x, min_y, min_z = cell_bound[1], cell_bound[3], cell_bound[5]
    p_in_cell = get_p_in_cell(pos, fluid_ids, cell_fst, cell_next, min_x, min_y, min_z, nxy, nx, control_radius)
    p_in_cell = np.argwhere(p_in_cell == 1).astype(np.int32).reshape(-1,)
    i2row = gen_i2row_map(p_in_cell, tot_num)
    d_i2row = cuda.to_device(i2row)

    d_cell_fst = cuda.to_device(cell_fst)
    d_cell_next = cuda.to_device(cell_next)
    edge_attr = np.zeros((len(p_in_cell), 6*64), dtype=np.float32)
    edge_idx = np.zeros((len(p_in_cell), 64), dtype=np.int32)
    d_edge_attr = cuda.to_device(edge_attr)
    d_edge_idx = cuda.to_device(edge_idx)

    cu_get_col_feature[blocks_pg, threads_pb](d_edge_attr, d_edge_idx, d_vel, d_pos, d_i2row, d_ptype,
                       d_cell_fst, d_cell_next, d_cell_bound,
                       control_radius, nx, nxy, tot_num)
    cp_edge_idx = cp.asarray(d_edge_idx[:, 0])
    tot_size = cp.sum(cp_edge_idx).item()
    edge_attr_arr = d_edge_attr.copy_to_host()
    edge_idx_arr = d_edge_idx.copy_to_host()
    edge_idx, edge_attr = stack_col_feature(edge_attr_arr, edge_idx_arr, tot_size)
    edge_attr = edge_attr
    return edge_attr, edge_idx, p_in_cell


def get_gns_feature_cuda(pos, ptype, control_radius):
    d_pos = cuda.to_device(pos)
    d_ptype = cuda.to_device(ptype)
    tot_num = pos.shape[0]

    threads_pb = (THREADS, 1, 1)
    blocks_pg = (int(tot_num / THREADS) + 1, 1, 1)

    nx, nxy, nxyz, cell_bound = calc_cell_params(pos, ptype, control_radius)
    d_cell_bound = cuda.to_device(cell_bound)

    cell_fst, cell_next = cell_sort(pos, cell_bound, control_radius, nx, nxy, nxyz, tot_num)
    fluid_ids = np.argwhere(ptype == 1).reshape(-1,)
    min_x, min_y, min_z = cell_bound[1], cell_bound[3], cell_bound[5]

    p_in_cell = get_p_in_cell(pos, fluid_ids, cell_fst, cell_next, min_x, min_y, min_z, nxy, nx, control_radius)
    p_in_cell = np.argwhere(p_in_cell == 1).astype(np.int32).reshape(-1,)
    i2row = gen_i2row_map(p_in_cell, tot_num)
    d_i2row = cuda.to_device(i2row)

    d_cell_fst = cuda.to_device(cell_fst)
    d_cell_next = cuda.to_device(cell_next)
    edge_attr = np.zeros((len(p_in_cell), 4*128), dtype=np.float32)
    edge_idx = np.zeros((len(p_in_cell), 128), dtype=np.int32)
    d_edge_attr = cuda.to_device(edge_attr)
    d_edge_idx = cuda.to_device(edge_idx)

    cu_get_gns_feature[blocks_pg, threads_pb](d_edge_attr, d_edge_idx, d_pos, d_i2row, d_ptype,
                       d_cell_fst, d_cell_next, d_cell_bound,
                       control_radius, nx, nxy, tot_num)
    cp_edge_idx = cp.asarray(d_edge_idx[:, 0])
    tot_size = cp.sum(cp_edge_idx).item()
    edge_attr_arr = d_edge_attr.copy_to_host()
    edge_idx_arr = d_edge_idx.copy_to_host()
    edge_idx, edge_attr = stack_gns_feature(edge_attr_arr, edge_idx_arr, tot_size)
    edge_attr = edge_attr
    return edge_attr, edge_idx, p_in_cell


def get_vel_div_cuda(vel, pos, ptype, control_radius):
    d_vel = cuda.to_device(vel)
    d_pos = cuda.to_device(pos)
    d_ptype = cuda.to_device(ptype)
    tot_num = pos.shape[0]

    threads_pb = (THREADS, 1, 1)
    blocks_pg = (int(tot_num / THREADS) + 1, 1, 1)

    nx, nxy, nxyz, cell_bound = calc_cell_params(pos, ptype, control_radius)
    d_cell_bound = cuda.to_device(cell_bound)

    cell_fst, cell_next = cell_sort(pos, cell_bound, control_radius, nx, nxy, nxyz, tot_num)
    d_cell_fst = cuda.to_device(cell_fst)
    d_cell_next = cuda.to_device(cell_next)

    divergence = np.zeros((tot_num, ), dtype=np.float32)
    d_divergence = cuda.to_device(divergence)
    cu_get_divergence[blocks_pg, threads_pb](d_divergence, d_vel, d_ptype, d_pos,
                      d_cell_fst, d_cell_next, d_cell_bound,
                      control_radius, nx, nxy, tot_num)

    divergence = d_divergence.copy_to_host()

    return divergence


def get_edge_idx_cuda(ids, pos, ptype, control_radius):
    d_pos = cuda.to_device(pos)
    d_ptype = cuda.to_device(ptype)
    tot_num = pos.shape[0]

    threads_pb = (THREADS, 1, 1)
    blocks_pg = (int(tot_num / THREADS) + 1, 1, 1)

    nx, nxy, nxyz, cell_bound = calc_cell_params(pos, ptype, control_radius)
    d_cell_bound = cuda.to_device(cell_bound)

    cell_fst, cell_next = cell_sort(pos, cell_bound, control_radius, nx, nxy, nxyz, tot_num)

    i2row = gen_i2row_map(ids, tot_num)
    d_i2row = cuda.to_device(i2row)

    d_cell_fst = cuda.to_device(cell_fst)
    d_cell_next = cuda.to_device(cell_next)
    if 1.9*BASE_RADIUS >= control_radius >= 1.0*BASE_RADIUS:
        row_len = 64
    elif 0.0 < control_radius < 1.0*BASE_RADIUS:
        row_len = 32
    elif 3.1*BASE_RADIUS > control_radius > 1.9*BASE_RADIUS:
        row_len = 384
    else:
        raise Exception('Unsupported control radius')

    edge_idx = np.zeros((len(ids), row_len), dtype=np.int32)
    d_edge_idx = cuda.to_device(edge_idx)

    cu_get_edge_idx[blocks_pg, threads_pb](d_edge_idx, d_i2row, d_ptype, d_pos,
                       d_cell_fst, d_cell_next, d_cell_bound,
                       control_radius, nx, nxy, tot_num)
    cp_edge_idx = cp.asarray(d_edge_idx[:, 0])
    tot_size = cp.sum(cp_edge_idx).item()
    edge_idx_arr = d_edge_idx.copy_to_host()
    edge_idx = stack_edge_idx(edge_idx_arr, tot_size)
    return edge_idx