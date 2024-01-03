import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


N_ENVS = 2
N_CLASSES = 2


class WaterbirdsDataset(Dataset):
    def __init__(self, dpath, df):
        self.dpath = dpath
        self.df = df
        self.transforms = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.transforms(Image.open(os.path.join(self.dpath, self.df.img_filename.iloc[idx])).convert('RGB'))
        y = torch.tensor(self.df.y.iloc[idx], dtype=torch.long)
        e = torch.tensor(self.df.e.iloc[idx])
        if not torch.isnan(e).any():
            e = e.long()
        return x, y, e


def landbirds_on_land_idxs(df):
    return np.where((df.y == 0) & (df.place == 0))[0]


def waterbirds_on_land_idxs(df):
    return np.where((df.y == 1) & (df.place == 0))[0]


def landbirds_on_water_idxs(df):
    return np.where((df.y == 0) & (df.place == 1))[0]


def waterbirds_on_water_idxs(df):
    return np.where((df.y == 1) & (df.place == 1))[0]


def sample_groups(rng, df, n_landbirds_on_land, n_landbirds_on_water, n_waterbirds_on_land, n_waterbirds_on_water):
    full_landbirds_on_land_idxs = landbirds_on_land_idxs(df)
    full_landbirds_on_water_idxs = landbirds_on_water_idxs(df)
    full_waterbirds_on_land_idxs = waterbirds_on_land_idxs(df)
    full_waterbirds_on_water_idxs = waterbirds_on_water_idxs(df)
    sample_landbirds_on_land_idxs = rng.choice(full_landbirds_on_land_idxs, n_landbirds_on_land, replace=False)
    sample_landbirds_on_water_idxs = rng.choice(full_landbirds_on_water_idxs, n_landbirds_on_water, replace=False)
    sample_waterbirds_on_land_idxs = rng.choice(full_waterbirds_on_land_idxs, n_waterbirds_on_land, replace=False)
    sample_waterbirds_on_water_idxs = rng.choice(full_waterbirds_on_water_idxs, n_waterbirds_on_water, replace=False)
    idxs = np.concatenate((sample_landbirds_on_land_idxs, sample_landbirds_on_water_idxs, sample_waterbirds_on_land_idxs,
        sample_waterbirds_on_water_idxs))
    return idxs


def drop_idxs(df, idxs):
    remaining_idxs = np.setdiff1d(np.arange(len(df)), idxs)
    return df.iloc[remaining_idxs]


def make_data(train_ratio, batch_size, eval_batch_size, n_workers, n_test_examples):
    '''
    Landbirds / waterbirds by place:
    land:  6220 / 831
    water: 2905 / 1832
    '''

    rng = np.random.RandomState(0)
    dpath = os.path.join(os.environ['DATA_DPATH'], 'waterbird_complete95_forest2water2')

    df = pd.read_csv(os.path.join(dpath, 'metadata.csv'))
    df['e'] = np.nan

    test_idxs = sample_groups(rng, df, 250, 250, 250, 250)
    df_test = df.iloc[test_idxs]
    df = drop_idxs(df, test_idxs)

    env0_idxs = sample_groups(rng, df, 425, 25, 25, 425)
    df_env0 = df.iloc[env0_idxs]
    df_env0.e = 0
    df = drop_idxs(df, env0_idxs)

    env1_idxs = sample_groups(rng, df, 25, 425, 425, 25)
    df_env1 = df.iloc[env1_idxs]
    df_env1.e = 1

    df_trainval = pd.concat((df_env0, df_env1))

    train_idxs = rng.choice(len(df_trainval), int(train_ratio * len(df_trainval)), replace=False)
    val_idxs = np.setdiff1d(np.arange(len(df_trainval)), train_idxs)

    df_train = df_trainval.iloc[train_idxs]
    df_val = df_trainval.iloc[val_idxs]

    data_train = DataLoader(WaterbirdsDataset(dpath, df_train), shuffle=True, pin_memory=True, batch_size=batch_size,
        num_workers=n_workers)
    data_val = DataLoader(WaterbirdsDataset(dpath, df_val), pin_memory=True, batch_size=eval_batch_size, num_workers=n_workers)
    data_test = DataLoader(WaterbirdsDataset(dpath, df_test), pin_memory=True, batch_size=eval_batch_size, num_workers=n_workers)
    return data_train, data_val, data_test