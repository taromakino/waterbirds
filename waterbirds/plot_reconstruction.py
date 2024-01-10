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
    for example_idx in range(N_EXAMPLES):
        fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'plot_reconstruction', str(example_idx))
        os.makedirs(fig_dpath, exist_ok=True)
        x, y, e, p, c = data_train.dataset.__getitem__(example_idx)
        x, y, e = x.unsqueeze(0).to(model.device), y.unsqueeze(0).to(model.device), e.unsqueeze(0).to(model.device)
        posterior_parent, posterior_child = model.encoder(x, y, e)
        z_parent, z_child = posterior_parent.loc, posterior_child.loc
        fig, axes = plt.subplots(2, N_COLS, figsize=(2 * N_COLS, 2 * 2))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        plot_image(axes[0, 0], x.squeeze().cpu().numpy())
        plot_image(axes[1, 0], x.squeeze().cpu().numpy())
        x_pred = reconstruct_x(model, z_parent, z_child)
        plot_image(axes[0, 1], x_pred.squeeze().detach().cpu().numpy())
        plot_image(axes[1, 1], x_pred.squeeze().detach().cpu().numpy())
        for col_idx in range(2, N_COLS):
            z_parent_prior, z_child_prior = sample_prior(rng, model)
            x_pred_parent = reconstruct_x(model, z_parent_prior, z_child)
            x_pred_child = reconstruct_x(model, z_parent, z_child_prior)
            plot_image(axes[0, col_idx], x_pred_parent.squeeze().detach().cpu().numpy())
            plot_image(axes[1, col_idx], x_pred_child.squeeze().detach().cpu().numpy())
        fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'plot_reconstruction')
        os.makedirs(fig_dpath, exist_ok=True)
        plt.savefig(os.path.join(fig_dpath, f'{example_idx}.png'))
        plt.close()