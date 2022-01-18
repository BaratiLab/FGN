#!usr/env/bin python3
import torch
import numpy as np
from sklearn.externals import joblib
import time
import pandas as pd
from gnn_model import *
from Particles import Particles, sparse_tensor
from Constants import *
from cg import cg_batch
import torch
from io_helper import load_df_grid, load_grid, load_npz_grid
import argparse
import os
from sklearn.preprocessing import RobustScaler
import pickle


def write_bf_data(out_pt, time_step, bf_train_feat):
    """write training data for training body force Graph Network"""
    print('writing training data for body force net')
    path = out_pt + 'bf_data' + str(time_step) + '.npz'
    np.savez_compressed(path, **bf_train_feat)


def write_col_data(out_pt, time_step, col_train_feat):
    """write training data for training collision Graph Network"""
    print('writing training data for collision net')
    path = out_pt + 'col_data' + str(time_step) + '.npz'
    np.savez_compressed(path, **col_train_feat)


def write_prs_data(out_pt, time_step, prs_train_feat):
    """write training data for training pressure Graph Network"""
    print('writing training data for pressure net')
    path = out_pt + 'prs_data' + str(time_step) + '.npz'
    np.savez_compressed(path, **prs_train_feat)


def write_bc_data(out_pt, time_step, gns_train_feat=None):
    # print('writing training data for benchmark')
    gns_path = out_pt + 'gns/gns_data' + str(time_step) + '.npz'
    # cconv_path = out_pt + 'cconv/cconv_data' + str(time_step) + '.npz'

    if time_step >= 5 and not gns_train_feat is None:   # as GNS need five previous velocities
        if not os.path.exists(out_pt + 'gns/'):
            os.mkdir(out_pt + 'gns/')
        np.savez_compressed(gns_path, **gns_train_feat)

    # if not os.path.exists(out_pt + 'cconv/'):
    #     os.mkdir(out_pt + 'cconv/')
    # np.savez_compressed(cconv_path, **cconv_train_feat)


def write_pnet_data(out_pt, time_step, pnet_train_feat):
    if not os.path.exists(out_pt + 'pnet/'):
        os.mkdir(out_pt + 'pnet/')
    pnet_path = out_pt + 'pnet/particle_data' + str(time_step) + '.npz'
    np.savez_compressed(pnet_path, **pnet_train_feat)


def run_sim(case, method, max_step, output_pt, write_out, write_train, write_bc, write_als):

    # visc_factor = np.random.randint(low=0, high=6)
    # visc_magn = np.random.uniform(low=0.1, high=1.0)
    # case.nu = NU*(10**visc_factor)*visc_magn
    # np.savez(os.path.join(output_pt, 'viscosity_info.npz'), visc_factor=visc_factor, visc_magn=visc_magn)
    print('Viscosity:%.6f' % case.nu)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('using device ' + device)
    # gns_radius = 2.1*BASE_RADIUS
    if method == 'GraphNet':
        dns_scaler = joblib.load('./train_model/scaler/dns_scaler.pt')
        prs_scaler = joblib.load('./train_model/scaler/prs_scaler.pt')
        pres_model_setting = [
            [4, 64],
            [64, 128],
            [128, 64],
            [64, 32],
            [32, 1]
        ]
        prs_net = PrsModel(pres_model_setting, GCN_layer=2)
        prs_net.load_state_dict(torch.load("./train_model/bc_state/prs2_model100.pt"))
        prs_net.to(device)
    gns_vel_cache = []
    rel_tols = []
    case.init_params()
    with torch.no_grad():
        for t in range(1, max_step+1):

            start = time.time()

            if write_bc:
                # for gns and cconv
                    vel_prev = case.get_vel()
                    pos_prev = case.get_pos()
                #     if t >= 5:
                #         gns_edge_feat, gns_edge_idx, pic, gns_edge_size =\
                #             case.get_adjacency(gns_radius, mode='gns feature', return_sparse_ele=True)
                #         gns_vel_cache.append(vel_prev)
                #
                #     else:
                #         gns_vel_cache.append(vel_prev)
                # gcn_train_feat = {}

            if write_train or method == 'GraphNet':
                bf_train_feat, col_train_feat, prs_train_feat, b = case.predict(return_train_feat=True)
            elif write_bc:
                dns_feat, vel_feat, b = case.predict(return_bc_data=True)
            else:
                b = case.predict(return_train_feat=False)

            # print('start pressure calculation')
            if method == 'MPS':
                # ============================ cg method ==============================
                b = torch.from_numpy(b).view(-1, 1).unsqueeze(0).to(device)
                A = case.get_adjacency(LAP_RADIUS, 'laplacian', ids_type='nfs', neg_lap='True')
                A = (2.0 * 3.0) / (case.rho * case.lam * case.N0_lap) * A
                A = A.to(device)

                A_bmm = lambda x: (torch.sparse.mm(A, x[0])).unsqueeze(0)
                cg_start = time.time()
                pres, info = cg_batch(A_bmm, b, atol=1.0e-8, maxiter=500)
                pres = pres.cpu()
                cg_end = time.time()

                cg_spend = cg_end - cg_start
                if (t % 10) == 0:
                    print('%i step, cg used time: %.6f second' % (t, cg_spend))
            elif method == 'WSPH':
                # ============== wsph =============
                pres = case.weak_compress_pressure()
            elif method == 'GraphNet':
                # ============== graph neural network ================
                idx, val, arr_size = prs_train_feat['gcn_idx'], prs_train_feat['gcn_val'], prs_train_feat['arr_size']
                gcn_adj = sparse_tensor(idx, val, arr_size).to(device)
                scale_dns = dns_scaler.transform(prs_train_feat['dns_feat'].reshape(-1, 1))
                dns_feat = torch.from_numpy(scale_dns).view(-1, 1)
                vel_prev = torch.from_numpy(prs_train_feat['vel_feat']).view(-1, 3)
                feat = torch.cat((dns_feat, vel_prev), dim=1).to(device)

                pres = prs_net.forward(feat, gcn_adj)
                pres = prs_scaler.inverse_transform(pres.cpu())
                with torch.no_grad():
                    b = torch.from_numpy(b).view(-1, 1).unsqueeze(0).to(device)
                    A = case.get_adjacency(LAP_RADIUS, 'laplacian', ids_type='nfs', neg_lap='True')
                    A = (2.0 * 3.0) / (case.rho * case.lam * case.N0_lap) * A
                    A = A.to(device)

                    A_bmm = lambda x: (torch.sparse.mm(A, x)).unsqueeze(0)
                    rel_tol = (torch.abs(torch.mean(A_bmm(torch.from_numpy(pres).view(-1, 1).float().to(device))-b))/
                                torch.mean(torch.abs(b))).cpu().numpy()
                    rel_tols += [rel_tol]
            prs = case.correct(pres, return_train_feat=True)

            if write_bc:
                # # for gns and cconv and pnet
                vel_gt = case.get_vel()
                # pos_gt = case.get_pos()
                acc_gt = (vel_gt - vel_prev) / case.DT

                # for gcn data
                # gcn_train_feat['dns_feat'] = dns_feat
                # gcn_train_feat['vel_feat'] = vel_feat
                # gcn_train_feat['prs_gt'] = prs
                # edge_idx_large, arr_size_large = case.get_adjacency(control_radius=1.6*BASE_RADIUS,
                #                                                     mode='edge idx',
                #                                                     ids_type='nfs',
                #                                                     return_sparse_ele=True)
                #
                # edge_idx_small, arr_size_small = case.get_adjacency(control_radius=1.1 * BASE_RADIUS,
                #                                                     mode='edge idx',
                #                                                     ids_type='nfs',
                #                                                     return_sparse_ele=True)
                # assert arr_size_large == arr_size_small
                # gcn_train_feat['adj_large'] = (edge_idx_large, arr_size_large)
                # gcn_train_feat['adj_small'] = (edge_idx_small, arr_size_small)
                # np.savez(output_pt+'gcn_data'+str(t)+'.npz', **gcn_train_feat)
                # if t == 1:
                #     print('writing gcn data')

            if (t % 10) == 0:
                end = time.time()
                print('%i step, total used time: %.6f second' % (t, end - start))
            if write_train:
                prs_train_feat['prs_gt'] = prs
                # write_bf_data(output_pt, t, bf_train_feat)
                # write_col_data(output_pt, t, col_train_feat)
                write_prs_data(output_pt, t, prs_train_feat)
            if write_out:
                case.write_info(output_pt, t, write_status=write_als)
            if write_bc:
                pnet_train_feat = {'vel': vel_prev,
                                     'pos': pos_prev,
                                     'acc': acc_gt}
                write_pnet_data(output_pt, t, pnet_train_feat)
                # if t >= 5:
                #     fluid_ids = case.fluid2id(pic, case.fluid_ids)
                #     assert(len(gns_vel_cache) == 5)
                #     gns_train_feat = {'vel_prev': gns_vel_cache,
                #                       'acc_gt': acc_gt,
                #                       'edge': gns_edge_feat,, data['gcn_val'], data['arr_size']
                #                       'edge_idx': gns_edge_idx,
                #                       'ids': fluid_ids,
                #                       'edge_size': gns_edge_size}
                #     write_bc_data(output_pt, t, gns_train_feat)
                #     gns_vel_cache = gns_vel_cache[1:]
                # else:
                #   write_bc_data(output_pt, t, cconv_train_feat
        # np.save(os.path.join(output_pt, 'rel_tolerance'), np.array(rel_tols))


def main():
    parser = argparse.ArgumentParser(
        description=
        "Using MPS or WSPH to simulate fluids"
    )

    parser.add_argument("--method",
                        type=str,
                        required=True,
                        help="Specify which numerical method to use (MPS/WSPH/GraphNet)")
    parser.add_argument("--max_step",
                        type=int,
                        default=250,
                        help="The number of simulation steps. Default is 250.")
    parser.add_argument("--grid_path",
                        type=str,
                        required=True,
                        help="The Path of the grid file")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="The output directory for the particle data.")
    parser.add_argument("--output_prefix",
                        type=str,
                        help="The prefix of the output file")
    parser.add_argument("--write_output",
                        action='store_true',
                        help="Export output data")
    parser.add_argument("--write_analysis_data",
                        action='store_true',
                        help='Data for analysis')
    parser.add_argument("--write_train_data",
                        action='store_true',
                        help="Export train data")
    parser.add_argument("--write_benchmark_data",
                        action='store_true',
                        help="Export train data for benchmark models")

    # to add later
    # parser.add_argument("--write-bgeo",
    #                     action='store_true',
    #                     help="Export particle data also as .bgeo sequence")
    args = parser.parse_args()
    print(args)
    method = args.method
    assert(method in ['MPS', 'WSPH', 'GraphNet'])
    max_step = args.max_step
    grid_pt = args.grid_path
    output_dir = args.output_dir
    if os.path.isdir(output_dir):
        print('Warning:'+output_dir+'already exists')
    else:
        os.mkdir(output_dir)
    output_prefix = args.output_prefix
    if output_prefix:
        output_pt = output_dir + output_prefix
    else:
        output_pt = output_dir
    write_out = args.write_output

    if grid_pt[-3:] == 'txt':
        case = load_grid(grid_pt)
    elif grid_pt[-3:] == 'csv':
        case = load_df_grid(grid_pt)
    elif grid_pt[-3:] == 'npz':
        case = load_npz_grid(grid_pt)
    else:
        raise Exception('Unsupported grid format')

    write_train = args.write_train_data
    write_bc = args.write_benchmark_data
    write_als = args.write_analysis_data

    run_sim(case, method, max_step, output_pt, write_out, write_train, write_bc, write_als)


if __name__ == '__main__':
    main()



