#!usr/env/bin python3
import torch
import numpy as np
import joblib
import time
import pandas as pd
from Particles import Particles, sparse_tensor
from Constants import *
from cg import cg_batch
import torch
from io_helper import load_df_grid, load_grid, load_npz_grid
import argparse
import os
from sklearn.preprocessing import RobustScaler
import pickle
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def run_sim(case, method, max_step, output_pt, write_out, write_train, write_als,
            dynamic_visc, end2end_data):
    if dynamic_visc:
        visc_factor = np.random.randint(low=4, high=7)
        visc_magn = np.random.uniform(low=0.1, high=0.9)
        case.nu = NU*(10**visc_factor)*visc_magn
        np.savez(output_pt + 'viscosity_info.npz', viscosity=case.nu)
    print('Viscosity:%.6f' % case.nu)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('using device ' + device)
    cache_data = write_train
    case.init_params()
    if end2end_data:
        with open(output_pt + 'case_info.pkl', 'wb') as pickle_file:
            pickle.dump(case, pickle_file)

    with torch.no_grad():
        for t in range(1, max_step+1):

            start = time.time()
            # print('start pressure calculation')
            if end2end_data:
                data = {}
                data['pos_prev'] = case.pos[case.fluid_ids]
                data['vel_prev'] = case.vel[case.fluid_ids]
            b = case.predict(cache_train_feat=cache_data and t % 4 == 0)

            if method == 'MPS':
                # ============================ cg method ==============================
                b = torch.from_numpy(b).view(-1, 1).unsqueeze(0).to(device)
                A = case.get_lap_adjacency(LAP_RADIUS, 'laplacian', ids_type='nfs', neg_lap='True')
                A = (2.0 * 3.0) / (case.rho * case.lam * case.N0_lap) * A
                A = A.to(device)

                A_bmm = lambda x: (torch.sparse.mm(A, x[0])).unsqueeze(0)
                cg_start = time.time()
                pres, info = cg_batch(A_bmm, b, atol=1.0e-6, maxiter=200)
                pres = pres.cpu()
                cg_end = time.time()

                cg_spend = cg_end - cg_start
                if (t % 10) == 0:
                    print('%i step, cg used time: %.6f second' % (t, cg_spend))
            elif method == 'WSPH':
                # ============== wsph =============
                pres = case.weak_compress_pressure()

            _ = case.correct(pres, cache_train_feat=cache_data and t % 4 == 0)

            if (t % 10) == 0:
                end = time.time()
                print('On %i step, used time: %.6f second' % (t, end - start))
            if end2end_data:
                data['pos_after'] = case.pos[case.fluid_ids]
                data['vel_after'] = case.vel[case.fluid_ids]
                with open(output_pt + f'end2end_data{t}.pkl', 'wb') as pickle_file:
                    pickle.dump(data, pickle_file)

            if t % 50 == 0:
                case.remove_particle(-5 * 0.050, which_axis=2, right=False)
                case.remove_particle(205 * 0.050, which_axis=2, right=True)

            if write_out:
                case.write_info(output_pt, t, write_status=write_als)

            if write_train:
                if not dynamic_visc:
                    case.write_data(output_pt, t)
                else:
                    case.write_data(output_pt, t, 'bf')


def main():
    parser = argparse.ArgumentParser(
        description=
        "Using MPS or WSPH to simulate fluids"
    )

    parser.add_argument("--method",
                        type=str,
                        required=True,
                        help="Specify which numerical method to use (MPS/WSPH)")
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
    parser.add_argument("--dynamic_visc",
                        action='store_true',
                        help="Different viscosity")
    parser.add_argument("--end_to_end",
                        action='store_true',
                        help="dump end2end data")

    # to add later
    # parser.add_argument("--write-bgeo",
    #                     action='store_true',
    #                     help="Export particle data also as .bgeo sequence")
    args = parser.parse_args()
    print(args)
    method = args.method
    assert(method in ['MPS', 'WSPH'])
    max_step = args.max_step
    grid_pt = args.grid_path
    output_dir = args.output_dir
    if output_dir is not None:
        if os.path.isdir(output_dir):
            print('Warning:'+output_dir+'already exists')
        else:
            os.mkdir(output_dir)
    output_prefix = args.output_prefix
    if output_prefix:
        output_pt = os.path.join(output_dir, output_prefix)
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
    write_als = args.write_analysis_data
    dy_visc = args.dynamic_visc
    e2e = args.end_to_end

    run_sim(case, method, max_step, output_pt, write_out, write_train, write_als, dy_visc, e2e)


if __name__ == '__main__':
    main()



