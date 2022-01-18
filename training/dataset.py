import torch
import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader


def find_idx(to_find_arr, key):
    """
    Copied from
    https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    """
    assert len(to_find_arr) >= len(key)
    xsorted_idx = np.argsort(to_find_arr)
    ypos = np.searchsorted(to_find_arr[xsorted_idx], key)
    indices = xsorted_idx[ypos]
    return indices


class PrsData(Dataset):
    def __init__(self,
                 dataset_path,
                 sample_num,
                 case_prefix='data',
                 seed_num=20,
                 ):
        self.dataset_path = dataset_path
        self.sample_num = sample_num                
        self.seed_num = seed_num
        self.case_prefix = case_prefix

    def __len__(self):
        return self.sample_num * self.seed_num

    def __getitem__(self, idx):
        seed_to_read = idx // self.sample_num
        sample_to_read = idx % self.sample_num+1
        fname = f'seed{seed_to_read}_{self.case_prefix}{sample_to_read}'
        data_path = os.path.join(self.dataset_path, fname)

        data = {}
        with open(data_path + '.pkl', 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            prs_data = raw_data['prs']
            data['pos'] = prs_data['pos'].astype(np.float32)
            data['vel'] = prs_data['vel'].astype(np.float32)
            data['dns'] = prs_data['dns'].astype(np.float32).reshape(-1, 1)
            data['prs'] = prs_data['prs'].astype(np.float32).reshape(-1, 1)
        return data


class AdvData(Dataset):
    def __init__(self,
                 dataset_path,
                 sample_num,
                 case_prefix='data',
                 seed_num=20,
                 use_param=False,
                 ):
        self.dataset_path = dataset_path
        self.sample_num = sample_num                
        self.seed_num = seed_num
        self.case_prefix = case_prefix
        self.use_param = use_param

    def __len__(self):
        return self.sample_num * self.seed_num

    def __getitem__(self, idx):
        seed_to_read = idx // self.sample_num
        sample_to_read = (idx % self.sample_num+1) * 4
        fname = f'seed{seed_to_read}_{self.case_prefix}{sample_to_read}'
        data_path = os.path.join(self.dataset_path, fname)

        if self.use_param:
            param_path = os.path.join(self.dataset_path,
                                      f'seed{seed_to_read}_viscosity_info.npz')
            visc = np.load(param_path)['viscosity']

        data = {}
        with open(data_path + '.pkl', 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            fluid_idx = raw_data['fluid_idx']
            if self.use_param:
                adv_data = raw_data
            else:
                adv_data = raw_data['bf']
            data['pos'] = adv_data['pos'].astype(np.float32)
            data['vel'] = adv_data['vel'].astype(np.float32)
            data['acc'] = adv_data['acc'].astype(np.float32)
            pic_index = adv_data['in_cell_idx']
            data['pic2fluid'] = find_idx(pic_index, fluid_idx)
            ptype = np.zeros((data['pos'].shape[0],))
            ptype[data['pic2fluid']] = 1.
            data['ptype'] = ptype.astype(np.float32)

        if self.use_param:
            data['visc_param'] = visc.astype(np.float32)
        return data


class ColData(Dataset):
    def __init__(self,
                 dataset_path,
                 sample_num,
                 case_prefix='data',
                 seed_num=20,
                 ):
        self.dataset_path = dataset_path
        self.sample_num = sample_num               
        self.seed_num = seed_num
        self.case_prefix = case_prefix

    def __len__(self):
        return self.sample_num * self.seed_num

    def __getitem__(self, idx):
        seed_to_read = idx // self.sample_num
        sample_to_read = idx % self.sample_num+1
        fname = f'seed{seed_to_read}_{self.case_prefix}{sample_to_read}'
        data_path = os.path.join(self.dataset_path, fname)

        data = {}
        with open(data_path + '.pkl', 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            fluid_idx = raw_data['fluid_idx']
            col_data = raw_data['col']
            data['pos'] = col_data['pos'].astype(np.float32)
            data['vel_prev'] = col_data['vel_prev'].astype(np.float32)
            data['vel_after'] = col_data['vel_after'].astype(np.float32)
            pic_index = col_data['in_cell_idx']
            data['pic2fluid'] = find_idx(pic_index, fluid_idx)
            ptype = np.zeros((data['pos'].shape[0],))
            ptype[data['pic2fluid']] = 1.
            data['ptype'] = ptype.astype(np.float32)

        return data


class PerFrameData(Dataset):
    def __init__(self,
                 dataset_path,
                 sample_num,
                 case_prefix='data',
                 seed_num=20,
                 use_next_frame=False,
                 ):
        self.dataset_path = dataset_path
        self.sample_num = sample_num  # - 1 # we need next frame
        self.seed_num = seed_num
        self.case_prefix = case_prefix
        self.use_next_frame = use_next_frame
        if use_next_frame:
            self.sample_num -= 1

    def __len__(self):
        return self.sample_num * self.seed_num

    def __getitem__(self, idx):
        seed_to_read = idx // self.sample_num
        sample_to_read = idx % self.sample_num + 1
        fname = f'seed{seed_to_read}_{self.case_prefix}{sample_to_read}'
        data_path = os.path.join(self.dataset_path, fname)
        case_instance_path = os.path.join(self.dataset_path, f'seed{seed_to_read}_case_info.pkl')
        with open(case_instance_path, 'rb') as pickle_file:
            case_instance = pickle.load(pickle_file)

        data = {}
        with open(data_path + '.pkl', 'rb') as pickle_file:
            raw_data = pickle.load(pickle_file)
            data['pos_prev'] = raw_data['pos_prev'].astype(np.float32)
            data['vel_prev'] = raw_data['vel_prev'].astype(np.float32)
            data['pos_after'] = raw_data['pos_after'].astype(np.float32)
            data['vel_after'] = raw_data['vel_after'].astype(np.float32)

        if self.use_next_frame:
            next_data_path = os.path.join(self.dataset_path, f'seed{seed_to_read}_{self.case_prefix}{sample_to_read+1}')
            with open(next_data_path + '.pkl', 'rb') as pickle_file:
                raw_data = pickle.load(pickle_file)
                data['vel_after_after'] = raw_data['vel_after'].astype(np.float32)
                data['pos_after_after'] = raw_data['pos_after'].astype(np.float32)

        return data, case_instance


if __name__ == '__main__':
    # test pressure data
    # dataset = PrsData('../dataset/training/', 250, seed_num=1)
    # dat = dataset[10]
    # print(dat['prs'].shape)
    # print(dat['vel'].shape)
    # print(dat['dns'].shape)
    # print(dat['pos'].shape)

    # test advection data
    # dataset = AdvData('../dataset/training/', 250, seed_num=1, use_param=False)
    # dat = dataset[10]
    # print(dat['acc'].shape)
    # print(dat['vel'].shape)
    # print(dat['pos'].shape)
    # print(dat['pic2fluid'].shape)
    # print(dat['ptype'].shape)
    # print(np.argwhere(dat['ptype']==1).shape)

    # dataset = AdvData('../dataset/training_visc/', 250, seed_num=100, use_param=True)
    # for i in range(len(dataset)):
    #     print(i)
    #     dat = dataset[i]
    #     print(dat['acc'].shape)
    #     print(dat['vel'].shape)
    #     print(dat['pos'].shape)
    #     print(dat['pic2fluid'].shape)
    #     print(dat['ptype'].shape)
    #     print(np.argwhere(dat['ptype']==1).shape)
    #     print(dat['visc_param'])




