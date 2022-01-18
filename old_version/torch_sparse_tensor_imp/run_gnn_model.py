import torch
from torch import nn
from torch.optim import Adam
from gnn_model import *
from Particles import Particles
from io_helper import load_grid
# from Generator import DAM_COL_CASE
from Constants import *
import pandas as pd
from sklearn.externals import joblib
import time
import pickle
import os


def main():

    TSTEP = 500
    output_path = "output/dam_col_FGNdt4/"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    grid_file = "./grid/dam_col_grid.txt"

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    pres_model_setting = [
                [4, 64],
                [64, 128],
                [128, 64],
                [64, 32],
                [32, 1]
                ]

    bf_model_setting = [
                [5, 128],
                [128, 64],
                [64, 32],
                [32, 3]
                ]

    case = load_grid(grid_file)
    case.init_params()

    bf_model = BFModel(bf_model_setting)
    bf_model.load_state_dict(torch.load('./train_model/model/bf_model200.pt'))
    col_model = ColModel()
    col_model.load_state_dict(torch.load("./train_model/model/col_model400.pt"))

    pres_model = PrsModel(pres_model_setting, GCN_layer=2)
    pres_model.load_state_dict(torch.load("./train_model/bc_state/prs2_model100.pt"))

    dns_scaler = joblib.load('./train_model/scaler/dns_scaler.pt')
    pres_scaler = joblib.load('./train_model/scaler/prs_scaler.pt')


    bf_model.to(device)
    col_model.to(device)
    pres_model.to(device)
    # general_pres_model.to(device)
    bf_model.eval()
    col_model.eval()
    pres_model.eval()
    # general_pres_model.eval()
    case.DT = 0.004
    N0_lap, N0, dt, total_num = case.N0_lap, case.N0, case.DT, case.total_num

    inference_time = []
    with torch.no_grad():
        for t in range(TSTEP):
            print("Current time step %i" % (t+1))
            step_start = time.time()
            vel = torch.from_numpy(case.get_vel())

            bf_feat = torch.cat((
                vel, torch.ones((vel.size(0), 1))*case.g, torch.ones((vel.size(0), 1))*case.nu
            ), dim=1).to(device)

            bf_gcn_avg = case.get_adjacency(LAP_RADIUS,
                                            mode='gcn average').to(device)

            ext_force = bf_model.forward(bf_feat, bf_gcn_avg)
            vel = case.col_predict(ext_force.cpu().numpy())

            edge_attr, edge_idx, p_in_cell, edge_size = case.get_adjacency(COL_RADIUS, mode='collision feature')
            v = vel[p_in_cell]
            if edge_attr.size(0) != 0:
                v = torch.from_numpy(v)
                v = v.to(device)
                col_feature = torch.ones((v.shape[0], 1)).to(device)
                edge_attr, edge_idx = edge_attr.to(device), edge_idx.to(device)
                col_gcn_avg = edge_attr, edge_idx, edge_size
                pred_vel = col_model.forward(col_feature, col_gcn_avg, v).detach().cpu().numpy()
                case.col_correct(pred_vel, p_in_cell)
            dns, vel_temp = case.pres_predict()
            if dns.shape[0] != 0:
                pres_feature = torch.from_numpy(
                    np.concatenate((
                    dns_scaler.transform(dns.reshape(-1, 1)),
                    vel_temp), axis=1).reshape(-1, 4)).to(device)

                pres_gcn_avg = case.get_adjacency(LAP_RADIUS, mode='gcn average', ids_type='nfs').to(device)

                pres_start = time.time()

                pred_pres_raw = pres_model.forward(pres_feature, pres_gcn_avg).detach().cpu().numpy()

                # ==============testing gcn and GraphSAGE==============================================
                # edge_idx_large, arr_size_large = case.get_adjacency(control_radius=1.6 * BASE_RADIUS,
                #                                                     mode='edge idx',
                #                                                     ids_type='nfs',
                #                                                     return_sparse_ele=True)
                #
                # edge_idx_small, arr_size_small = case.get_adjacency(control_radius=1.1 * BASE_RADIUS,
                #                                                     mode='edge idx',
                #                                                     ids_type='nfs',
                #                                                     return_sparse_ele=True)
                # edge_idx_large = torch.from_numpy(edge_idx_large.transpose())
                # edge_idx_small = torch.from_numpy(edge_idx_small.transpose())
                #
                # edge_idx_large = edge_idx_large.to(device)
                # edge_idx_small = edge_idx_small.to(device)
                #
                # gcn_adj = [edge_idx_large, edge_idx_small]
                # pred_pres_raw = general_pres_model.forward(pres_feature, gcn_adj).detach().cpu().numpy()

                # ======================================================================================
                pres_end = time.time()
                pres_inf_time = pres_end - pres_start
                print('pressure inference used time: %.6f' % pres_inf_time)
                inference_time.append(pres_inf_time)
                pred_pres = pres_scaler.inverse_transform(pred_pres_raw)
                case.correct(pred_pres)

            step_end = time.time()
            print('Current step used time: %.6f' % (step_end - step_start))
            case.write_info(output_path, t+1, write_status=True)
    


main()
