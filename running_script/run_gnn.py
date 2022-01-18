#!usr/env/bin python3
import torch
import numpy as np
import joblib
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from cg import cg_batch
from Particles import Particles
import torch
from io_helper import load_df_grid, load_grid, load_npz_grid
import argparse
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from nn_module_new import NodeNetwork, EdgeNetwork, LIN_KERNEL
from Constants import LAP_RADIUS, COL_RADIUS
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def predict_prs_with_gnn(prs_feat, model, prs_scaler, dns_scaler, vel_norm_scaler):
    pos = torch.from_numpy(prs_feat['pos']).cuda()
    vel = torch.from_numpy(prs_feat['vel']).cuda()
    dns = torch.from_numpy(prs_feat['dns']).cuda()
    with torch.no_grad():
        # normalize vel
        vel_norm = torch.norm(vel, dim=1).view(-1, 1) + 1e-8
        vel /= vel_norm
        vel_norm = (vel_norm - vel_norm_scaler.mean_.item()) / np.sqrt(vel_norm_scaler.var_.item())

        # normalize dns
        dns = (dns - dns_scaler.mean_.item()) / np.sqrt(dns_scaler.var_.item())
        feat = torch.cat((vel, vel_norm, dns.view(-1, 1)), dim=1)

        pred = model.forward(feat, pos)

        prs = pred.cpu().numpy() * np.sqrt(prs_scaler.var_.item()) + prs_scaler.mean_.item()
    return prs


def predict_adv_with_gnn(case: Particles, model: NodeNetwork, acc_scaler, vel_norm_scaler):

    fluid_idx = case.fluid_ids
    g, pic_index, col_pic = case.get_graph(0.6*LAP_RADIUS, LIN_KERNEL, True)
    pos = torch.from_numpy(case.pos[pic_index]).cuda()
    vel = torch.from_numpy(case.vel[pic_index]).cuda()
    pic2fluid = case.find_idx(pic_index, fluid_idx)
    ptype = np.zeros((pos.shape[0],))
    ptype[pic2fluid] = 1.
    ptype = torch.from_numpy(ptype.astype(np.float32)).cuda()

    with torch.no_grad():
        # normalize vel
        vel_norm = torch.norm(vel, dim=1).view(-1, 1) + 1e-8
        vel /= vel_norm
        vel_norm = (vel_norm - vel_norm_scaler.mean_.item()) / np.sqrt(vel_norm_scaler.var_.item())
        feat = torch.cat((vel, vel_norm, ptype.view(-1, 1)), dim=1)
        pred = model.forward(feat, pos, g)
        pred = pred[pic2fluid]
        acc = pred.cpu().numpy() * np.sqrt(acc_scaler.var_) + acc_scaler.mean_
    return acc, col_pic


def predict_col_with_gnn(case: Particles, model: EdgeNetwork, pic_index):

    fluid_idx = case.fluid_ids
    pos = torch.from_numpy(case.pos[pic_index]).cuda()
    vel = torch.from_numpy(case.vel[pic_index]).cuda()
    pic2fluid = case.find_idx(pic_index, fluid_idx)
    # ptype = np.zeros((pos.shape[0],))
    # ptype[pic2fluid] = 1.
    # ptype = torch.from_numpy(ptype.astype(np.float32)).cuda()

    with torch.no_grad():
        # normalize vel
        # vel_norm = torch.norm(vel, dim=1).view(-1, 1) + 1e-8
        # vel /= vel_norm
        # vel_norm_normalized = (vel_norm - vel_norm_scaler.mean_.item()) / np.sqrt(vel_norm_scaler.var_.item())
        # feat = torch.cat((vel, vel_norm_normalized, ptype.view(-1, 1)), dim=1)
        pred, _ = model.forward(vel, pos)
        pred = pred[pic2fluid]
        v = pred.cpu().numpy()
    return v


def run_sim(case, max_step, output_pt, write_out, write_als):

    print('Viscosity:%.6f' % case.nu)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('using device ' + device)

    # load pressure net
    break_point_pt = '../training/prs_net_small_state/ckpt_final.pkl'
    with open(break_point_pt, 'rb') as pickle_file:
        bp = pickle.load(pickle_file)

    prs_net = NodeNetwork(5, 1, 2, 0.55*LAP_RADIUS)
    prs_net.eval()
    prs_net.load_state_dict(bp['model_state'])
    prs_net.cuda()
    prs_vel_norm_scaler = bp['vel_scaler']
    prs_dns_scaler = bp['dns_scaler']
    prs_scaler = bp['prs_scaler']

    # load advection net
    break_point_pt = '../training/adv_net_state/ckpt_final.pkl'
    with open(break_point_pt, 'rb') as pickle_file:
        bp = pickle.load(pickle_file)

    adv_net = NodeNetwork(5, 3, 2, 0.6*LAP_RADIUS)
    adv_net.eval()
    adv_net.load_state_dict(bp['model_state'])
    adv_net.cuda()
    adv_vel_norm_scaler = bp['vel_scaler']
    adv_acc_scaler = bp['acc_scaler']

    # load collision net
    break_point_pt = '../training/col_net_state/ckpt_final.pkl'
    with open(break_point_pt, 'rb') as pickle_file:
        bp = pickle.load(pickle_file)

    col_net = EdgeNetwork(3, 3, 1, COL_RADIUS)
    col_net.eval()
    col_net.load_state_dict(bp['model_state'])
    col_net.cuda()
    # col_vel_norm_scaler = bp['vel_scaler']

    case.init_params()
    with torch.no_grad():
        for t in range(1, max_step+1):

            start = time.time()
            # print('start pressure calculation')
            adv_acc, col_pic = predict_adv_with_gnn(case, adv_net, adv_acc_scaler, adv_vel_norm_scaler)
            case.advect(adv_acc)

            new_v = predict_col_with_gnn(case, col_net, col_pic)
            case.col_correct(new_v)
            prs_feat = case.get_prs_feat()
            pres = predict_prs_with_gnn(prs_feat, prs_net, prs_scaler, prs_dns_scaler, prs_vel_norm_scaler)
            _ = case.correct(pres, cache_train_feat=False)
            case.cache = {}
            if (t % 10) == 0:
                end = time.time()
                print('On %i step, used time: %.6f second' % (t, end - start))

            if write_out:
                case.write_info(output_pt, t, write_status=write_als)

            if t % 50 == 0:
                case.remove_particle(-5 * 0.050, which_axis=2, right=False)
                case.remove_particle(205 * 0.050, which_axis=2, right=True)


def main():
    parser = argparse.ArgumentParser(
        description=
        "Using GNN to simulate fluids"
    )
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

    args = parser.parse_args()
    print(args)

    max_step = args.max_step
    grid_pt = args.grid_path
    output_dir = args.output_dir
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

    write_als = args.write_analysis_data

    run_sim(case, max_step, output_pt, write_out, write_als)


if __name__ == '__main__':
    main()

