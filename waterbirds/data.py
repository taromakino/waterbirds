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


class WaterbirdsDataset(Dataset):
    def __init__(self, dpath, df):
        self.dpath = dpath
        self.df = df
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
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


def make_data(test_e_name, train_ratio, batch_size, eval_batch_size, n_workers, n_test_examples):
    rng = np.random.RandomState(0)
    dpath = os.path.join(os.environ['DATA_DPATH'], 'waterbird_complete95_forest2water2')

    df = pd.read_csv(os.path.join(dpath, 'metadata.csv'))
    df['e_name'] = df.place_filename.apply(lambda x: x.split('/')[2])

    df_test = df[df.e_name == test_e_name.value]
    landbird_idxs = np.where(df_test.y == 0)[0]
    waterbird_idxs = np.where(df_test.y == 1)[0]
    landbird_idxs = rng.choice(landbird_idxs, len(waterbird_idxs), replace=False)
    df_test = df_test.iloc[np.concatenate((landbird_idxs, waterbird_idxs))]
    df_test['e'] = float('nan')

    df_trainval = df[df.e_name != test_e_name.value]
    df_trainval['e'] = pd.factorize(df_trainval.e_name)[0]

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