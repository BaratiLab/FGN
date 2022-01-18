import numpy as np
import os, sys
sys.path.append('..')
from train_utils import *
from Particles import sparse_tensor
from gnn_model import *
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import joblib
import pickle
import numba as nb


def read_dataset(fname, max_step=None, single_step=None):
    if not single_step is None:
        return np.load(fname + str(single_step) + '.npz', allow_pickle=True)
    data = []
    for step in range(1, max_step+1):
        npz_data = np.load(fname + str(step) + '.npz', allow_pickle=True)
        data += [npz_data]
    return data


def transform_prs_data(prs_data):
    adjs = []
    dns_feat = []
    vel_feat = []
    prs_gt = []
    for data in prs_data:
        idx, val, arr_size = data['gcn_idx'], data['gcn_val'], data['arr_size']
        # adjs += [sparse_tensor(idx, val, arr_size)]
        adjs += [(idx, val, arr_size)]
        dns_feat += [data['dns_feat']]
        vel_feat += [data['vel_feat']]
        prs_gt += [data['prs_gt']]
    return adjs, dns_feat, vel_feat, prs_gt


def transform_gcn_prs_data(prs_data):
    adjs = []
    dns_feat = []
    vel_feat = []
    prs_gt = []
    for data in prs_data:
        adj_large = data['adj_large']
        adj_small = data['adj_small']
        adjs += [(adj_large, adj_small)]
        dns_feat += [data['dns_feat']]
        vel_feat += [data['vel_feat']]
        prs_gt += [data['prs_gt']]
    return adjs, dns_feat, vel_feat, prs_gt


def transform_col_data(col_data):
    adj_info = []
    vel_prev = []
    vel_gt = []
    ids = []
    for data in col_data:
        edge_idx, edge_attr, edge_size = data['edge_idx'], data['edge_attr'], data['edge_size'].item()
        if edge_size == 0:
            continue
        adj_info += [(edge_attr, edge_idx, edge_size)]
        ids += [data['fld_ids']]
        vel_prev += [data['vel_prev']]
        vel_gt += [data['vel_gt']]
    return adj_info, vel_prev, vel_gt, ids


def transform_bf_data(bf_data):
    adjs = []
    vel_prev = []
    acc = []
    for data in bf_data:
        idx, val, arr_size = data['gcn_idx'], data['gcn_val'], data['arr_size']
        # adjs += [sparse_tensor(idx, val, arr_size)]
        adjs += [(idx, val, arr_size)]
        vel_prev += [data['vel_prev']]
        acc += [data['acc']]
    return adjs, vel_prev, acc


def load_data_for_epoch(path, data_type, data_num):
    data = []
    if data_type == 'bf':
        nus = []
    for _ in range(data_num):
        case_to_read = np.random.randint(low=11, high=16)
        step_to_read = np.random.randint(low=1, high=201)
        fname = path + '/case' + str(case_to_read) + '/' + data_type + '_data'
        if data_type == 'bf':
            visc_dat = np.load(path + '/case' + str(case_to_read) + '/viscosity_info.npz')
            nus += [visc_dat['visc_factor']*visc_dat['visc_magn']]

        data += [read_dataset(fname, single_step=step_to_read)]
    if data_type == 'prs':
        return transform_prs_data(data)
    elif data_type == 'col':
        return transform_col_data(data)
    elif data_type == 'bf':
        return transform_bf_data(data), nus
    elif data_type == 'gcn':
        return transform_gcn_prs_data(data)
    else:
        raise Exception('Unsupported data type')


def scale_all_density():
    from sklearn.preprocessing import RobustScaler as scaler
    import joblib
    dns = []
    for i in range(20):
        for t in range(1, 1001):
            pt = './data_set/case' + str(i) + '/prs_data' + str(t) + '.npz'
            data = np.load(pt, allow_pickle=True)
            dns.append(data['dns_feat'].reshape(-1, 1))
    dns_all = np.concatenate((dns), axis=0)
    dns_scaler = scaler()
    dns_scaler.fit(dns_all)
    joblib.dump(dns_scaler, 'dns_scaler.pt')


def scale_all_pressure():
    from sklearn.preprocessing import RobustScaler as scaler
    import joblib
    prs = []
    for i in range(20):
        for t in range(1, 1001):
            pt = './data_set/case' + str(i) + '/prs_data' + str(t) + '.npz'
            data = np.load(pt, allow_pickle=True)
            prs.append(data['prs_gt'].reshape(-1, 1))
    prs_all = np.concatenate(prs, axis=0)
    prs_scaler = scaler()
    prs_scaler.fit(prs_all)
    joblib.dump(prs_scaler, 'prs_scaler.pt')


def train_bf_model(break_point=None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    init_lr = 0.001
    start, epochs = 0, 200

    model_setting = [
        [5, 64],
        [64, 128],
        [128, 64],
        [64, 32],
        [32, 3]
    ]

    model = BFModel(model_setting, GCN_layer=2).to(device)
    if break_point is not None:
        state_pt = break_point['state']
        model.load_state_dict(torch.load(state_pt))
        init_lr = break_point['lr']
        start = break_point['start']

    batch_size = 48
    data_num = 250
    mini_epochs = int(10000 / data_num)
    optim = Adam(model.parameters(), lr=init_lr)
    scheduler = StepLR(optim, 20, gamma=0.5, last_epoch=-1)
    for t in range(start, epochs):
        batch_loss = 0.
        tot_loss = 0.
        for mini_epoch in range(mini_epochs):
            (adjs, vel_prev, acc_gt), nus = load_data_for_epoch('./visc_data', 'bf', data_num)
            data = BFDataLoader(adjs, vel_prev, acc_gt, viscosity=nus, batch_size=batch_size, train_test_split=1.00)
            data_size = data.get_train_size()
            iterate_all = False
            while not iterate_all:
                train_adjs, train_feat, train_gt, cursor, iterate_all = data.get_batch()
                current = mini_epoch * data_size + cursor

                acc_gt = train_gt.float().view(-1, 3).to(device)
                feature = train_feat.view(-1, 5).to(device)
                # v =  torch.from_numpy(train_feat[:, :3].reshape(-1, 3)).float().to(device)
                adj = train_adjs.to(device)

                pred_acc = model.forward(feature, adj).to(device)

                loss = nn.MSELoss()(pred_acc, acc_gt)
                # loss = nn.L1Loss(reduction='sum')(pred[ids_batch], gt[ids_batch])
                # loss = torch.sum((pred[ids_batch] - gt[ids_batch])**2)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_val = loss.item()
                tot_loss += loss_val
                batch_loss += loss_val
                if (current % (batch_size*10)) == 0:
                    print('Finished training %i samples in current epoch, average loss %.6f' %
                          (current, batch_loss / (batch_size*10)))
                    batch_loss = 0.
        print('=========================')
        print("Epoch: %i, loss: %.6f" % (t + 1, tot_loss / data_size * mini_epochs))
        scheduler.step()
        if (t + 1) % 20 == 0:
            torch.save(model.state_dict(), './visc_model/bf_model' + str(t + 1) + '.pt')
        print('=========================')


def train_col_model(break_point=None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    init_lr = 0.001
    start, epochs = 0, 100
    batch_size = 4
    data_num = 500
    mini_epochs = int(20000 / data_num)

    model = ColModel().to(device)
    if break_point is not None:
        state_pt = break_point['state']
        model.load_state_dict(torch.load(state_pt))
        init_lr = break_point['lr']
        start = break_point['start']

    optim = Adam(model.parameters(), lr=init_lr)
    scheduler = StepLR(optim, 20, gamma=0.5, last_epoch=-1)

    for t in range(start, epochs):
        batch_loss = 0.
        tot_loss = 0.
        for mini_epoch in range(mini_epochs):
            adjs, vel_prev, vel_gt, ids = load_data_for_epoch('./data_set', 'col', data_num)
            data = CollisionDataLoader(adjs, ids, vel_prev, vel_gt, batch_size, 1.00)
            data_size = data.get_train_size()
            iterate_all = False
            while (not iterate_all):
                train_adjs, train_feat, train_gt, cursor, iterate_all, ids_batch = data.get_batch(scale=False)
                current = mini_epoch * data_size + cursor

                v_gt = train_gt.view(-1, 3).to(device)
                dummy_feature = torch.ones((train_feat.shape[0], 1), dtype=torch.float32).to(device)
                edge_attr = train_adjs[0].to(device)
                edge_idx = train_adjs[1].to(device)
                edge_size = train_adjs[2]

                adj = edge_attr, edge_idx, edge_size
                v = train_feat.view(-1, 3).to(device)
                pred_v = model.forward(dummy_feature, adj, v).to(device)

                loss = nn.MSELoss()(pred_v[ids_batch], v_gt[ids_batch])
                # loss = nn.L1Loss(reduction='sum')(pred[ids_batch], gt[ids_batch])
                # loss = torch.sum((pred[ids_batch] - gt[ids_batch])**2)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_val = loss.item()
                tot_loss += loss_val
                batch_loss += loss_val
                if (current % (batch_size*100)) == 0:
                    print(
                        'Finished training %i samples in current epoch, average loss %.6f'
                        % (current, batch_loss / (batch_size*100)))
                    batch_loss = 0.
        print('=========================')
        print("Epoch: %i, loss: %.6f" % (t + 1, tot_loss / (data_num * mini_epochs)))
        scheduler.step()
        if (t + 1) % 20 == 0:
            torch.save(model.state_dict(), './bc_state/col_model' + str(t + 1) + '.pt')
        print('=========================')


def train_prs_model(break_point=None):
    dns_scaler = joblib.load('./scaler/dns_scaler.pt')
    prs_scaler = joblib.load('./scaler/prs_scaler.pt')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_setting = [
        [4, 64],
        [64, 128],
        [128, 64],
        [64, 32],
        [32, 1]
    ]
    init_lr = 0.001
    start, epochs = 0, 100
    batch_size = 32
    data_num = 200
    mini_epochs = int(20000 / data_num)

    model = PrsModel(model_setting, GCN_layer=2).to(device)
    if break_point is not None:
        state_pt = break_point['state']
        model.load_state_dict(torch.load(state_pt))
        init_lr = break_point['lr']
        start = break_point['start']

    optim = Adam(model.parameters(), init_lr)
    scheduler = StepLR(optim, 20, gamma=0.5, last_epoch=-1)
    print('Start training from epoch:%i'%start)
    for t in range(start, epochs):
        batch_loss = 0.
        tot_loss = 0.
        for mini_epoch in range(mini_epochs):
            adj, dns_feat, vel_feat, prs_gt = load_data_for_epoch('./data_set', 'prs', data_num)
            data = PressureDataLoader(adj, dns_feat, vel_feat, prs_gt, prs_scaler, dns_scaler, batch_size, 1.00)
            data_size = data.get_train_size()
            iterate_all = False
            while not iterate_all:
                train_adjs, train_feat, train_gt, cursor, iterate_all = data.get_batch()
                #train_feat, train_gt, cursor, iterate_all = data.get_batch(no_adj=True)

                current = mini_epoch * data_size + cursor
                gt = train_gt.to(device)

                feature = train_feat.to(device)
                adj = train_adjs.to(device)
                # adj = 0.
                pred = model.forward(feature, adj)

                loss = nn.MSELoss()(pred, gt)

                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_val = loss.item()
                tot_loss += loss_val
                batch_loss += loss_val
                if (current % (batch_size*100)) == 0:
                    print('Finished training %i samples in current epoch, average loss %.6f'
                          % (current, batch_loss / (batch_size*100)))
                    batch_loss = 0.
        print('=========================')
        print("Epoch: %i, loss: %.6f" % (t + 1, tot_loss / (data_size * mini_epochs)))
        scheduler.step()
        if (t + 1) % 20 == 0:
            torch.save(model.state_dict(), './bc_state/prs2_model' + str(t + 1) + '.pt')
        print('=========================')


def train_general_model(break_point=None):
    dns_scaler = joblib.load('./scaler/dns_scaler.pt')
    prs_scaler = joblib.load('./scaler/prs_scaler.pt')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_setting = [
        {'in_channels': 4, 'out_channels': 128},
        {'in_channels': 128, 'out_channels': 128},
        {'in_channels': 128, 'out_channels': 64},
        {'in_channels': 64, 'out_channels': 32},
        {'in_channels': 32, 'out_channels': 1}
    ]
    init_lr = 0.001
    start, epochs = 0, 100
    batch_size = 64
    data_num = 200
    mini_epochs = int(20000 / data_num)

    model = GeneralNet(model_setting, GCN_type='GCN', GCN_layer=2).to(device)
    if break_point is not None:
        state_pt = break_point['state']
        model.load_state_dict(torch.load(state_pt))
        init_lr = break_point['lr']
        start = break_point['start']

    optim = Adam(model.parameters(), init_lr)
    scheduler = StepLR(optim, 20, gamma=0.3, last_epoch=-1)
    print('Start training from epoch:%i' % start)
    for t in range(start, epochs):
        batch_loss = 0.
        tot_loss = 0.
        for mini_epoch in range(mini_epochs):
            adj, dns_feat, vel_feat, prs_gt = load_data_for_epoch('./gcn_dataset', 'gcn', data_num)
            data = GCNPressureDataLoader(adj, dns_feat, vel_feat, prs_gt, prs_scaler, dns_scaler, batch_size, 1.00)
            data_size = data.get_train_size()
            iterate_all = False
            while not iterate_all:
                train_adjs, train_feat, train_gt, cursor, iterate_all = data.get_batch()
                #train_feat, train_gt, cursor, iterate_all = data.get_batch(no_adj=True)

                current = mini_epoch * data_size + cursor
                gt = train_gt.to(device)

                feature = train_feat.to(device)
                adj_large = train_adjs[0].to(device)
                # adj_small = train_adjs[1].to(device)
                adj = [adj_large, adj_large]
                # adj = 0.
                pred = model.forward(feature, adj)

                loss = nn.MSELoss()(pred, gt)

                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_val = loss.item()
                tot_loss += loss_val
                batch_loss += loss_val
                if (current % (batch_size*100)) == 0:
                    print('Finished training %i samples in current epoch, average loss %.6f'
                          % (current, batch_loss / (batch_size*100)))
                    batch_loss = 0.
        print('=========================')
        print("Epoch: %i, loss: %.6f" % (t + 1, tot_loss / (data_size * mini_epochs)))
        scheduler.step()
        # if (t + 1) % 10 == 0:
        #     torch.save(model.state_dict(), './bc_state/gsage/gsage3_model' + str(t + 1) + '.pt')
        print('=========================')
        torch.save(model.state_dict(), './bc_state/gcn/gcn_model.pt')


def main():
    train_bf_model()


if __name__ == '__main__':
    main()