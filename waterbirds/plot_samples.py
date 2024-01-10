import os
import pytorch_lightning as pl
import torch
from data import N_CLASSES, N_ENVS
from decoder_cnn import IMG_DECODE_SHAPE
from utils.enums import Task
from utils.plot import *
from vae import VAE


N_COLS = 10


def sample_prior(rng, model):
    y = torch.tensor(rng.choice(N_CLASSES), dtype=torch.long, device=model.device)[None]
    e = torch.tensor(rng.choice(N_ENVS), dtype=torch.long, device=model.device)[None]
    prior_parent, prior_child = model.prior(y, e)
    z_parent, z_child = prior_parent.sample(), prior_child.sample()
    return z_parent, z_child


def reconstruct_x(model, z_parent, z_child):
    batch_size = len(z_parent)
    z = torch.hstack((z_parent, z_child))
    x_pred = model.decoder.mlp(z).reshape(batch_size, *IMG_DECODE_SHAPE)
    x_pred = model.decoder.decoder_cnn(x_pred)
    return torch.sigmoid(x_pred)


def main(args):
    rng = np.random.RandomState(args.seed)
    task_dpath = os.path.join(args.dpath, Task.VAE.value)
    pl.seed_everything(args.seed)
    model = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    model.eval()
    fig, axes = plt.subplots(1, N_COLS, figsize=(N_COLS, 2))
    for ax in axes:
        remove_ticks(ax)
    for col_idx in range(N_COLS):
        z_parent, z_child = sample_prior(rng, model)
        x_pred = reconstruct_x(model, z_parent, z_child)
        plot_image(axes[col_idx], x_pred.squeeze().detach().cpu().numpy())
    fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig')
    os.makedirs(fig_dpath, exist_ok=True)
    plt.savefig(os.path.join(fig_dpath, 'plot_samples.png'))
    plt.close()