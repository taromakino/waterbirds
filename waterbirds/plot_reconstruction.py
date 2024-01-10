import os
import pytorch_lightning as pl
from data import make_data
from plot_samples import sample_prior, reconstruct_x
from utils.enums import Task
from utils.plot import *
from vae import VAE


N_EXAMPLES = 10
N_COLS = 10


def main(args):
    rng = np.random.RandomState(args.seed)
    task_dpath = os.path.join(args.dpath, Task.VAE.value)
    pl.seed_everything(args.seed)
    data_train, _, _ = make_data(args.train_ratio, args.batch_size, args.eval_batch_size, args.n_workers)
    model = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    model.eval()
    x_full, y_full, e_full, _, _ = data_train.dataset[:]
    for example_idx in range(N_EXAMPLES):
        fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'plot_reconstruction', str(example_idx))
        os.makedirs(fig_dpath, exist_ok=True)
        x, y, e = x_full[[example_idx]], y_full[[example_idx]], e_full[[example_idx]]
        x, y, e = x.to(model.device), y.to(model.device), e.to(model.device)
        posterior_parent, posterior_child = model.encoder(x, y, e)
        z_parent, z_child = posterior_parent.loc, posterior_child.loc
        # Original
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        remove_ticks(ax)
        plot_image(ax, x.squeeze().cpu().numpy())
        plt.savefig(os.path.join(fig_dpath, 'original.png'))
        plt.close()
        # Sample parent
        fig, axes = plt.subplots(1, N_COLS, figsize=(2 * N_COLS, 2))
        for ax in axes:
            remove_ticks(ax)
        for col_idx in range(N_COLS):
            z_parent_prior, _ = sample_prior(rng, model)
            x_pred = reconstruct_x(model, z_parent_prior, z_child)
            plot_image(axes[col_idx], x_pred.squeeze().detach().cpu().numpy())
        plt.savefig(os.path.join(fig_dpath, 'sample_parent.png'))
        plt.close()
        # Sample child
        fig, axes = plt.subplots(1, N_COLS, figsize=(2 * N_COLS, 2))
        for ax in axes:
            remove_ticks(ax)
        for col_idx in range(N_COLS):
            _, z_child_prior = sample_prior(rng, model)
            x_pred = reconstruct_x(model, z_parent, z_child_prior)
            plot_image(axes[col_idx], x_pred.squeeze().detach().cpu().numpy())
        plt.savefig(os.path.join(fig_dpath, 'sample_child.png'))
        plt.close()