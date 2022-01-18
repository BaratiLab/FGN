import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataset import PrsData, AdvData, ColData
from torch.utils.data import DataLoader
import sys
import joblib
import pickle
from sklearn.preprocessing import StandardScaler as Scaler
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from nn_module_new import NodeNetwork, EdgeNetwork
from Constants import LAP_RADIUS, COL_RADIUS


def dump_break_point(output_path, model_state, optim_state, scheduler_state, scaler_dict, epoch):
    break_point_dict = {
        'model_state': model_state,
        'optim_state': optim_state,
        'schd_state': scheduler_state,
        'start': epoch,
    }
    break_point_dict.update(scaler_dict)
    with open(output_path, 'wb') as pickle_file:
        pickle.dump(break_point_dict, pickle_file)


def train_col_model(model: EdgeNetwork, dataset_path, output_path, break_point=None):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    init_lr = 0.001
    start, epochs = 0, 15

    model = model.to(device)
    model.train()
    optim = Adam(model.parameters(), init_lr)
    scheduler = StepLR(optim, 3, gamma=0.333, last_epoch=-1)

    # vel_norm_scaler = Scaler()
    # acc_scaler = Scaler()

    if break_point is not None:
        model_state = break_point['model_state']
        model.load_state_dict(model_state)
        optim_state = break_point['optim_state']
        optim.load_state_dict(optim_state)
        schd_state = break_point['schd_state']
        scheduler.load_state_dict(schd_state)
        start = break_point['start']

        # for scaler
        # vel_norm_scaler = break_point['vel_sacler']

    col_data = ColData(dataset_path, 1000, seed_num=20)
    data_num = len(col_data)
    data_loader = DataLoader(col_data, 1, shuffle=True, num_workers=4)
    os.makedirs(output_path, exist_ok=True)
    print('Start training from epoch:%i'%start)
    print(f'Using {data_num} data points in total')
    loss_history = []
    for t in range(start, epochs):
        tot_loss = 0.
        pbar = tqdm(enumerate(data_loader))
        for current, data in pbar:

            pos = data['pos'].squeeze(0).to(device)
            vel_prev = data['vel_prev'].squeeze(0).to(device)
            vel_after = data['vel_after'].squeeze(0).to(device)

            # # normalize vel
            # vel_norm = torch.norm(vel_prev, dim=1).view(-1, 1) + 1e-8
            # vel_prev /= vel_norm
            # vel_norm_scaler.partial_fit(vel_norm.cpu().numpy())
            # vel_norm_normalized = (vel_norm - vel_norm_scaler.mean_.item()) / np.sqrt(vel_norm_scaler.var_.item())
            # feat = torch.cat((vel_prev, vel_norm_normalized, ptype.view(-1, 1)), dim=1)
            pic2fld = data['pic2fluid'].squeeze(0)
            vel_norm = torch.norm(vel_prev[pic2fld], dim=1).view(-1, 1) + 1e-8

            pred, fl = model.forward(vel_prev, pos)
            if not fl:
                continue
            # normalize vel gt
            # vel_after /= vel_norm
            loss = (nn.L1Loss()(pred[pic2fld] / vel_norm, vel_after[pic2fld] / vel_norm))

            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_val = loss.item()
            tot_loss += loss_val
            loss_history += [loss_val]
            pbar.set_description(
                f'Running loss: {loss_val:.4f}' )

        print('=========================')
        print("Epoch: %i, loss: %.6f" % (t + 1, tot_loss / data_num))
        scheduler.step()
        if (t + 1) % 5 == 0:
            dump_break_point(os.path.join(output_path, f'ckpt_{t+1}.pkl'),
                              model.state_dict(), optim.state_dict(), scheduler.state_dict(),
                             {},
                             # {'vel_scaler': vel_norm_scaler,
                             #  'acc_scaler': acc_scaler},
                              t)
            np.save(os.path.join(output_path, f'loss_trend_{t+1}.npy'), np.array(loss_history))
        print('=========================')
    dump_break_point(os.path.join(output_path, 'ckpt_final.pkl'),
                     model.state_dict(), optim.state_dict(), scheduler.state_dict(),{},
                     # {'vel_scaler': vel_norm_scaler,
                     #  'acc_scaler': acc_scaler},
                     5)
    np.save(os.path.join(output_path, 'loss_trend.npy'), np.array(loss_history))


def train_adv_model(model: NodeNetwork, dataset_path, output_path, break_point=None):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    init_lr = 0.001
    start, epochs = 0, 10

    model = model.to(device)
    model.train()
    optim = Adam(model.parameters(), init_lr)
    scheduler = StepLR(optim, 3, gamma=0.333, last_epoch=-1)

    vel_norm_scaler = Scaler()
    acc_scaler = Scaler()

    if break_point is not None:
        model_state = break_point['model_state']
        model.load_state_dict(model_state)
        optim_state = break_point['optim_state']
        optim.load_state_dict(optim_state)
        schd_state = break_point['schd_state']
        scheduler.load_state_dict(schd_state)
        start = break_point['start']

        # for scaler
        vel_norm_scaler = break_point['vel_sacler']
        acc_scaler = break_point['acc_sacler']

    adv_data = AdvData(dataset_path, 1000, seed_num=20)
    data_num = len(adv_data)
    data_loader = DataLoader(adv_data, 1, shuffle=True, num_workers=4)
    os.makedirs(output_path, exist_ok=True)
    print('Start training from epoch:%i'%start)
    print(f'Using {data_num} data points in total')
    loss_history = []
    for t in range(start, epochs):
        tot_loss = 0.
        pbar = tqdm(enumerate(data_loader))
        for current, data in pbar:

            pos = data['pos'].squeeze(0).to(device)
            vel = data['vel'].squeeze(0).to(device)
            acc = data['acc'].squeeze(0)
            ptype = data['ptype'].squeeze(0).to(device)

            # normalize vel
            vel_norm = torch.norm(vel, dim=1).view(-1, 1) + 1e-8
            vel /= vel_norm
            vel_norm_scaler.partial_fit(vel_norm.cpu().numpy())
            vel_norm = (vel_norm - vel_norm_scaler.mean_.item()) / np.sqrt(vel_norm_scaler.var_.item())

            # normalize acc
            acc_scaler.partial_fit(acc.cpu().numpy())
            acc = (acc - torch.from_numpy(acc_scaler.mean_)) / (torch.from_numpy(np.sqrt(acc_scaler.var_)) + 1e-10)
            acc = acc.float().to(device)
            pic2fld = data['pic2fluid'].squeeze(0)

            feat = torch.cat((vel, vel_norm, ptype.view(-1, 1)), dim=1)

            pred = model.forward(feat, pos)

            loss = nn.MSELoss()(pred[pic2fld], acc)

            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_val = loss.item()
            tot_loss += loss_val
            loss_history += [loss_val]
            pbar.set_description(
                f'Running loss: {loss_val:.4f}' )

        print('=========================')
        print("Epoch: %i, loss: %.6f" % (t + 1, tot_loss / data_num))
        scheduler.step()
        if (t + 1) % 5 == 0:
            dump_break_point(os.path.join(output_path, f'ckpt_{t+1}.pkl'),
                              model.state_dict(), optim.state_dict(), scheduler.state_dict(),
                             {'vel_scaler': vel_norm_scaler,
                              'acc_scaler': acc_scaler},
                              t)
            np.save(os.path.join(output_path, f'loss_trend_{t+1}.npy'), np.array(loss_history))
        print('=========================')
    dump_break_point(os.path.join(output_path, 'ckpt_final.pkl'),
                     model.state_dict(), optim.state_dict(), scheduler.state_dict(),
                     {'vel_scaler': vel_norm_scaler,
                      'acc_scaler': acc_scaler},
                     5)
    np.save(os.path.join(output_path, 'loss_trend.npy'), np.array(loss_history))


def train_general_adv_model(model: NodeNetwork, dataset_path, output_path, break_point=None):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    init_lr = 0.001
    start, epochs = 0, 15

    model = model.to(device)
    model.train()
    optim = Adam(model.parameters(), init_lr)
    scheduler = StepLR(optim, 3, gamma=0.333, last_epoch=-1)

    vel_norm_scaler = Scaler()
    visc_scaler = Scaler()
    acc_scaler = Scaler()

    if break_point is not None:
        model_state = break_point['model_state']
        model.load_state_dict(model_state)
        optim_state = break_point['optim_state']
        optim.load_state_dict(optim_state)
        schd_state = break_point['schd_state']
        scheduler.load_state_dict(schd_state)
        start = break_point['start']

        # for scaler
        vel_norm_scaler = break_point['vel_scaler']
        visc_scaler = break_point['vel_scaler']
        acc_scaler = break_point['acc_scaler']

    adv_data = AdvData(dataset_path, 250, seed_num=100, use_param=True)
    data_num = len(adv_data)
    data_loader = DataLoader(adv_data, 1, shuffle=True, num_workers=4)
    os.makedirs(output_path, exist_ok=True)
    print('Start training from epoch:%i'%start)
    print(f'Using {data_num} data points in total')
    loss_history = []
    for t in range(start, epochs):
        tot_loss = 0.
        pbar = tqdm(enumerate(data_loader))
        for current, data in pbar:

            pos = data['pos'].squeeze(0).to(device)
            vel = data['vel'].squeeze(0).to(device)
            acc = data['acc'].squeeze(0)
            ptype = data['ptype'].squeeze(0).to(device)
            visc = data['visc_param'].squeeze(0)

            # normalize vel
            vel_norm = torch.norm(vel, dim=1).view(-1, 1) + 1e-8
            vel /= vel_norm
            vel_norm_scaler.partial_fit(vel_norm.cpu().numpy())
            vel_norm = (vel_norm - vel_norm_scaler.mean_.item()) / np.sqrt(vel_norm_scaler.var_.item())

            # normalize acc
            acc_scaler.partial_fit(acc.cpu().numpy())
            acc = (acc - torch.from_numpy(acc_scaler.mean_)) / (torch.from_numpy(np.sqrt(acc_scaler.var_)) + 1e-10)
            acc = acc.float().to(device)
            pic2fld = data['pic2fluid'].squeeze(0)

            # normalize viscosity
            visc = torch.log10(visc)
            visc_scaler.partial_fit(visc.cpu().numpy().reshape(-1 ,1))
            visc = (visc - torch.from_numpy(visc_scaler.mean_)) / (torch.from_numpy(np.sqrt(visc_scaler.var_)) + 1e-10)
            visc = visc.item() * torch.ones_like(ptype.view(-1, 1))

            feat = torch.cat((vel, vel_norm, ptype.view(-1, 1), visc), dim=1)

            pred = model.forward(feat, pos)

            loss = nn.MSELoss()(pred[pic2fld], acc)

            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_val = loss.item()
            tot_loss += loss_val
            loss_history += [loss_val]
            pbar.set_description(
                f'Running loss: {loss_val:.4f}' )

        print('=========================')
        print("Epoch: %i, loss: %.6f" % (t + 1, tot_loss / data_num))
        scheduler.step()
        if (t + 1) % 5 == 0:
            dump_break_point(os.path.join(output_path, f'ckpt_{t+1}.pkl'),
                              model.state_dict(), optim.state_dict(), scheduler.state_dict(),
                             {'vel_scaler': vel_norm_scaler,
                              'acc_scaler': acc_scaler,
                              'visc_scaler': visc_scaler},
                              t)
            np.save(os.path.join(output_path, f'loss_trend_{t+1}.npy'), np.array(loss_history))
        print('=========================')
    dump_break_point(os.path.join(output_path, 'ckpt_final.pkl'),
                     model.state_dict(), optim.state_dict(), scheduler.state_dict(),
                     {'vel_scaler': vel_norm_scaler,
                      'acc_scaler': acc_scaler,
                      'visc_scaler': visc_scaler},
                     5)
    np.save(os.path.join(output_path, 'loss_trend.npy'), np.array(loss_history))


def train_prs_model(model: NodeNetwork, dataset_path, output_path, break_point=None):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    init_lr = 0.001
    start, epochs = 0, 15

    model = model.to(device)
    model.train()
    optim = Adam(model.parameters(), init_lr)
    scheduler = StepLR(optim, 3, gamma=0.333, last_epoch=-1)

    vel_norm_scaler = Scaler()
    dns_scaler = Scaler()
    prs_scaler = Scaler()

    if break_point is not None:
        model_state = break_point['model_state']
        model.load_state_dict(model_state)
        optim_state = break_point['optim_state']
        optim.load_state_dict(optim_state)
        schd_state = break_point['schd_state']
        scheduler.load_state_dict(schd_state)
        start = break_point['start']

        # for scaler
        vel_norm_scaler = joblib.load(break_point['vel_sacler'])
        dns_scaler = joblib.load(break_point['dns_sacler'])
        prs_scaler = joblib.load(break_point['prs_sacler'])

    prs_data = PrsData(dataset_path, 1000, seed_num=20)
    data_num = len(prs_data)
    data_loader = DataLoader(prs_data, 1, shuffle=True, num_workers=4)
    os.makedirs(output_path, exist_ok=True)
    print('Start training from epoch:%i'%start)
    print(f'Using {data_num} data points in total')
    loss_history = []
    for t in range(start, epochs):
        tot_loss = 0.
        pbar = tqdm(enumerate(data_loader))
        for current, data in pbar:

            pos = data['pos'].squeeze(0).to(device)
            vel = data['vel'].squeeze(0).to(device)
            dns = data['dns'].squeeze(0)
            prs = data['prs'].squeeze(0)

            # normalize vel
            vel_norm = torch.norm(vel, dim=1).view(-1, 1) + 1e-8
            vel /= vel_norm
            vel_norm_scaler.partial_fit(vel_norm.cpu().numpy())
            vel_norm = (vel_norm - vel_norm_scaler.mean_.item()) / np.sqrt(vel_norm_scaler.var_.item())

            # normalize dns
            dns_scaler.partial_fit(dns.cpu().numpy())
            dns = (dns - dns_scaler.mean_.item()) / np.sqrt(dns_scaler.var_.item())
            dns = dns.to(device)

            # normalize prs
            prs_scaler.partial_fit(prs.cpu().numpy())
            prs = (prs - prs_scaler.mean_.item()) / np.sqrt(prs_scaler.var_.item())
            prs = prs.to(device)

            feat = torch.cat((vel, vel_norm, dns), dim=1)

            pred = model.forward(feat, pos)

            loss = nn.MSELoss()(pred, prs)

            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_val = loss.item()
            tot_loss += loss_val
            loss_history += [loss_val]
            pbar.set_description(
                f'Running loss: {loss_val:.4f}' )

        print('=========================')
        print("Epoch: %i, loss: %.6f" % (t + 1, tot_loss / data_num))
        scheduler.step()
        if (t + 1) % 5 == 0:
            dump_break_point(os.path.join(output_path, f'ckpt_{t+1}.pkl'),
                              model.state_dict(), optim.state_dict(), scheduler.state_dict(),
                             {'vel_scaler': vel_norm_scaler,
                              'dns_scaler': dns_scaler,
                              'prs_scaler': prs_scaler},
                              t)
            np.save(os.path.join(output_path, f'loss_trend_{t+1}.npy'), np.array(loss_history))
        print('=========================')
    dump_break_point(os.path.join(output_path, 'ckpt_final.pkl'),
                     model.state_dict(), optim.state_dict(), scheduler.state_dict(),
                     {'vel_scaler': vel_norm_scaler,
                      'dns_scaler': dns_scaler,
                      'prs_scaler': prs_scaler},
                     5)
    np.save(os.path.join(output_path, 'loss_trend.npy'), np.array(loss_history))


if __name__ == '__main__':
    prs_net = NodeNetwork(5, 1, 2, 0.55*LAP_RADIUS)
    total_params = sum(p.numel() for p in prs_net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    train_prs_model(prs_net, '../dataset/training', './prs_net_small_state')

    adv_net = NodeNetwork(5, 3, 2, 0.6*LAP_RADIUS)
    total_params = sum(p.numel() for p in adv_net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    train_adv_model(adv_net, '../dataset/training', './adv_net_state')

    col_net = EdgeNetwork(3, 3, 1, 1.0*COL_RADIUS)
    total_params = sum(p.numel() for p in col_net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    train_col_model(col_net, '../dataset/training', './col_net_state')

