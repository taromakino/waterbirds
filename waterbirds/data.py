import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.enums import Environment
from torch.utils.data import Dataset, DataLoader


N_ENVS = len(Environment) - 1 # Don't count test env
N_CLASSES = 2
N_TEST_LANDBIRDS = 700
N_TEST_WATERBIRDS = 200


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


def subsample(rng, df, n_eval_examples):
    if len(df) < n_eval_examples:
        return df
    else:
        idxs = rng.choice(len(df), n_eval_examples, replace=False)
        return df.iloc[idxs]


def make_data(train_ratio, batch_size, eval_batch_size, n_workers, n_test_examples):
    '''
    Landbirds / waterbirds by place:
    land:  6220 / 831
    water: 2905 / 1832

    Environment 0:
    land:  3089 / 436

    Environment 1:
    land:  3131 / 395
    water: 2205 / 1632

    Test environment:
    water: 700 / 200

    The spurious feature is constant for environment 0. On environment 1, the water background is positively correlated with
    waterbirds. On the test environment, the water background is negatively correlated with waterbirds.
    '''

    rng = np.random.RandomState(0)
    dpath = os.path.join(os.environ['DATA_DPATH'], 'waterbird_complete95_forest2water2')

    df = pd.read_csv(os.path.join(dpath, 'metadata.csv'))
    df['e'] = np.nan

    landbirds_on_water_idxs = np.where((df.y == 0) & (df.place == 1))[0]
    waterbirds_on_water_idxs = np.where((df.y == 1) & (df.place == 1))[0]

    test_landbirds_on_water_idxs = rng.choice(landbirds_on_water_idxs, N_TEST_LANDBIRDS, replace=False)
    test_waterbirds_on_water_idxs = rng.choice(waterbirds_on_water_idxs, N_TEST_WATERBIRDS, replace=False)

    test_idxs = np.concatenate((test_landbirds_on_water_idxs, test_waterbirds_on_water_idxs))
    trainval_idxs = np.setdiff1d(np.arange(len(df)), test_idxs)

    df_trainval = df.iloc[trainval_idxs]
    df_test = df.iloc[test_idxs]

    trainval_land_idxs = np.where(df_trainval.place == 0)[0]
    env0_idxs = rng.choice(trainval_land_idxs, len(trainval_land_idxs) // 2, replace=False)
    env1_idxs = np.setdiff1d(np.arange(len(df_trainval)), env0_idxs)
    df_trainval.e.iloc[env0_idxs] = 0
    df_trainval.e.iloc[env1_idxs] = 1

    train_idxs = rng.choice(len(df_trainval), int(train_ratio * len(df_trainval)), replace=False)
    val_idxs = np.setdiff1d(np.arange(len(df_trainval)), train_idxs)

    df_train = df_trainval.iloc[train_idxs]
    df_val = df_trainval.iloc[val_idxs]

    if n_test_examples is not None:
        df_test = subsample(rng, df_test, n_test_examples)

    data_train = DataLoader(WaterbirdsDataset(dpath, df_train), shuffle=True, pin_memory=True, batch_size=batch_size,
        num_workers=n_workers)
    data_val = DataLoader(WaterbirdsDataset(dpath, df_val), pin_memory=True, batch_size=eval_batch_size, num_workers=n_workers)
    data_test = DataLoader(WaterbirdsDataset(dpath, df_test), pin_memory=True, batch_size=eval_batch_size, num_workers=n_workers)
    return data_train, data_val, data_test