import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.plot import *


N_ENVS = 3
N_CLASSES = 2
SUBPLACE_TO_INT = {
    'bamboo_forest': 0,
    'forest': 1,
    'lake': 2,
    'ocean': 3
}


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
        p = torch.tensor(self.df.subplace.map(SUBPLACE_TO_INT).values)
        c = y
        return x, y, e, p, c


def drop_idxs(df, idxs):
    remaining_idxs = np.setdiff1d(np.arange(len(df)), idxs)
    return df.iloc[remaining_idxs]


def make_dfs(train_ratio):
    '''
    Landbirds on bamboo forest:  3115
    Landbirds on forest:         3105
    Landbirds on lake:           1442
    Landbirds on ocean:          1463

    Waterbirds on bamboo forest: 419
    Waterbirds on forest:        412
    Waterbirds on lake:          932
    Waterbirds on ocean:         900
    '''

    rng = np.random.RandomState(0)
    dpath = os.path.join(os.environ['DATA_DPATH'], 'waterbird_complete95_forest2water2')

    df = pd.read_csv(os.path.join(dpath, 'metadata.csv'))
    df['e'] = np.nan
    df['subplace'] = df.place_filename.apply(lambda x: x.split('/')[2])

    test_landbirds_on_bamboo_forest_idxs = np.where((df.y == 0) & (df.subplace == 'bamboo_forest'))[0]
    test_landbirds_on_forest_idxs = np.where((df.y == 0) & (df.subplace == 'forest'))[0]
    test_landbirds_on_lake_idxs = np.where((df.y == 0) & (df.subplace == 'lake'))[0]
    test_landbirds_on_ocean_idxs = np.where((df.y == 0) & (df.subplace == 'ocean'))[0]

    test_waterbirds_on_bamboo_forest_idxs = np.where((df.y == 1) & (df.subplace == 'bamboo_forest'))[0]
    test_waterbirds_on_forest_idxs = np.where((df.y == 1) & (df.subplace == 'forest'))[0]
    test_waterbirds_on_lake_idxs = np.where((df.y == 1) & (df.subplace == 'lake'))[0]
    test_waterbirds_on_ocean_idxs = np.where((df.y == 1) & (df.subplace == 'ocean'))[0]

    test_landbirds_on_bamboo_forest_idxs = rng.choice(test_landbirds_on_bamboo_forest_idxs, 125, replace=False)
    test_landbirds_on_forest_idxs = rng.choice(test_landbirds_on_forest_idxs, 125, replace=False)
    test_landbirds_on_lake_idxs = rng.choice(test_landbirds_on_lake_idxs, 125, replace=False)
    test_landbirds_on_ocean_idxs = rng.choice(test_landbirds_on_ocean_idxs, 125, replace=False)

    test_waterbirds_on_bamboo_forest_idxs = rng.choice(test_waterbirds_on_bamboo_forest_idxs, 125, replace=False)
    test_waterbirds_on_forest_idxs = rng.choice(test_waterbirds_on_forest_idxs, 125, replace=False)
    test_waterbirds_on_lake_idxs = rng.choice(test_waterbirds_on_lake_idxs, 125, replace=False)
    test_waterbirds_on_ocean_idxs = rng.choice(test_waterbirds_on_ocean_idxs, 125, replace=False)

    test_idxs = np.concatenate((test_landbirds_on_bamboo_forest_idxs, test_landbirds_on_forest_idxs, test_landbirds_on_lake_idxs,
        test_landbirds_on_ocean_idxs, test_waterbirds_on_bamboo_forest_idxs, test_waterbirds_on_forest_idxs,
        test_waterbirds_on_lake_idxs, test_waterbirds_on_ocean_idxs))
    df_test = df.iloc[test_idxs]
    df = drop_idxs(df, test_idxs)

    env0_landbirds_on_bamboo_forest_idxs = np.where((df.y == 0) & (df.subplace == 'bamboo_forest'))[0]
    env0_landbirds_on_lake_idxs = np.where((df.y == 0) & (df.subplace == 'lake'))[0]

    env0_waterbirds_on_bamboo_forest_idxs = np.where((df.y == 1) & (df.subplace == 'bamboo_forest'))[0]
    env0_waterbirds_on_lake_idxs = np.where((df.y == 1) & (df.subplace == 'lake'))[0]

    env0_landbirds_on_bamboo_forest_idxs = rng.choice(env0_landbirds_on_bamboo_forest_idxs, 425, replace=False)
    env0_landbirds_on_lake_idxs = rng.choice(env0_landbirds_on_lake_idxs, 25, replace=False)

    env0_waterbirds_on_bamboo_forest_idxs = rng.choice(env0_waterbirds_on_bamboo_forest_idxs, 25, replace=False)
    env0_waterbirds_on_lake_idxs = rng.choice(env0_waterbirds_on_lake_idxs, 425, replace=False)

    env0_idxs = np.concatenate((env0_landbirds_on_bamboo_forest_idxs, env0_landbirds_on_lake_idxs,
        env0_waterbirds_on_bamboo_forest_idxs, env0_waterbirds_on_lake_idxs))
    df_env0 = df.iloc[env0_idxs]
    df_env0.e = 0
    df = drop_idxs(df, env0_idxs)

    env1_landbirds_on_forest_idxs = np.where((df.y == 0) & (df.subplace == 'forest'))[0]
    env1_landbirds_on_ocean_idxs = np.where((df.y == 0) & (df.subplace == 'ocean'))[0]

    env1_waterbirds_on_forest_idxs = np.where((df.y == 1) & (df.subplace == 'forest'))[0]
    env1_waterbirds_on_ocean_idxs = np.where((df.y == 1) & (df.subplace == 'ocean'))[0]

    env1_landbirds_on_forest_idxs = rng.choice(env1_landbirds_on_forest_idxs, 425, replace=False)
    env1_landbirds_on_ocean_idxs = rng.choice(env1_landbirds_on_ocean_idxs, 25, replace=False)

    env1_waterbirds_on_forest_idxs = rng.choice(env1_waterbirds_on_forest_idxs, 25, replace=False)
    env1_waterbirds_on_ocean_idxs = rng.choice(env1_waterbirds_on_ocean_idxs, 425, replace=False)

    env1_idxs = np.concatenate((env1_landbirds_on_forest_idxs, env1_landbirds_on_ocean_idxs, env1_waterbirds_on_forest_idxs,
         env1_waterbirds_on_ocean_idxs))
    df_env1 = df.iloc[env1_idxs]
    df_env1.e = 1
    df = drop_idxs(df, env1_idxs)

    env2_landbirds_on_lake_idxs = np.where((df.y == 0) & (df.subplace == 'lake'))[0]
    env2_waterbirds_on_lake_idxs = np.where((df.y == 1) & (df.subplace == 'lake'))[0]

    env2_landbirds_on_lake_idxs = rng.choice(env2_landbirds_on_lake_idxs, 382, replace=False)
    # env2_waterbirds_on_lake_idxs = rng.choice(env2_waterbirds_on_lake_idxs, 382, replace=False) # All remaining data

    env2_idxs = np.concatenate((env2_landbirds_on_lake_idxs, env2_waterbirds_on_lake_idxs))
    df_env2 = df.iloc[env2_idxs]
    df_env2.e = 2

    df_trainval = pd.concat((df_env0, df_env1, df_env2))

    train_idxs = rng.choice(len(df_trainval), int(train_ratio * len(df_trainval)), replace=False)
    val_idxs = np.setdiff1d(np.arange(len(df_trainval)), train_idxs)

    df_train = df_trainval.iloc[train_idxs]
    df_val = df_trainval.iloc[val_idxs]
    return df_train, df_val, df_test


def make_data(train_ratio, batch_size, eval_batch_size, n_workers):
    dpath = os.path.join(os.environ['DATA_DPATH'], 'waterbird_complete95_forest2water2')
    df_train, df_val, df_test = make_dfs(train_ratio)
    data_train = DataLoader(WaterbirdsDataset(dpath, df_train), shuffle=True, pin_memory=True, batch_size=batch_size,
        num_workers=n_workers)
    data_val = DataLoader(WaterbirdsDataset(dpath, df_val), pin_memory=True, batch_size=eval_batch_size, num_workers=n_workers)
    data_test = DataLoader(WaterbirdsDataset(dpath, df_test), pin_memory=True, batch_size=eval_batch_size, num_workers=n_workers)
    return data_train, data_val, data_test


def main():
    df_train, _, df_test = make_dfs(1)
    subplace_train = df_train.subplace.map(SUBPLACE_TO_INT)
    subplace_test = df_test.subplace.map(SUBPLACE_TO_INT)
    # Train e=0
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].hist(subplace_train[(df_train.y == 0) & (df_train.e == 0)], bins='auto', color='red')
    axes[1].hist(subplace_train[(df_train.y == 1) & (df_train.e == 0)], bins='auto', color='blue')
    axes[0].set_title('p(subplace | y=0, e=0)')
    axes[1].set_title('p(subplace | y=1, e=0)')
    for ax in axes:
        ax.set_xlim(0, len(SUBPLACE_TO_INT) - 1)
    fig.tight_layout()
    # Train e=1
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].hist(subplace_train[(df_train.y == 0) & (df_train.e == 1)], bins='auto', color='red')
    axes[1].hist(subplace_train[(df_train.y == 1) & (df_train.e == 1)], bins='auto', color='blue')
    axes[0].set_title('p(subplace | y=0, e=1)')
    axes[1].set_title('p(subplace | y=1, e=1)')
    for ax in axes:
        ax.set_xlim(0, len(SUBPLACE_TO_INT) - 1)
    fig.tight_layout()
    # Train e=2
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].hist(subplace_train[(df_train.y == 0) & (df_train.e == 2)], bins='auto', color='red')
    axes[1].hist(subplace_train[(df_train.y == 1) & (df_train.e == 2)], bins='auto', color='blue')
    axes[0].set_title('p(subplace | y=0, e=2)')
    axes[1].set_title('p(subplace | y=1, e=2)')
    for ax in axes:
        ax.set_xlim(0, len(SUBPLACE_TO_INT) - 1)
    fig.tight_layout()
    # Test
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].hist(subplace_test[df_test.y == 0], bins='auto', color='red')
    axes[1].hist(subplace_test[df_test.y == 1], bins='auto', color='blue')
    axes[0].set_title('p(subplace | y=0, e=test)')
    axes[1].set_title('p(subplace | y=1, e=test)')
    for ax in axes:
        ax.set_xlim(0, len(SUBPLACE_TO_INT) - 1)
    fig.tight_layout()
    # ERM
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].hist(subplace_train[df_train.y == 0], bins='auto', color='red')
    axes[1].hist(subplace_train[df_train.y == 1], bins='auto', color='blue')
    axes[0].set_title('p(subplace | y=0)')
    axes[1].set_title('p(subplace | y=1)')
    for ax in axes:
        ax.set_xlim(0, len(SUBPLACE_TO_INT) - 1)
    fig.tight_layout()
    plt.show(block=True)

if __name__ == '__main__':
    main()