import data
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from erm import ERM
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.enums import Task, EvalStage
from vae import VAE


def make_data(args):
    data_train, data_val, data_test = data.make_data(args.train_ratio, args.batch_size, args.eval_batch_size, args.n_workers)
    if args.eval_stage is None:
        data_eval = None
    elif args.eval_stage == EvalStage.TRAIN:
        data_eval = data_train
    elif args.eval_stage == EvalStage.VAL:
        data_eval = data_val
    else:
        assert args.eval_stage == EvalStage.TEST
        data_eval = data_test
    return data_train, data_val, data_test, data_eval


def ckpt_fpath(args, task):
    return os.path.join(args.dpath, task.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')


def make_model(args):
    is_train = args.eval_stage is None
    if args.task == Task.ERM:
        if is_train:
            return ERM(args.lr, args.weight_decay)
        else:
            return ERM.load_from_checkpoint(ckpt_fpath(args, args.task))
    elif args.task == Task.VAE:
        return VAE(args.task, args.z_size, args.h_sizes, args.y_mult, args.beta, args.prior_reg_mult, args.init_sd,
            args.lr, args.weight_decay)
    else:
        assert args.task == Task.CLASSIFY
        return VAE.load_from_checkpoint(ckpt_fpath(args, Task.VAE), task=args.task)


def main(args):
    pl.seed_everything(args.seed)
    data_train, data_val, data_test, data_eval = make_data(args)
    model = make_model(args)
    if args.task == Task.ERM:
        if args.eval_stage is None:
            trainer = pl.Trainer(
                logger=CSVLogger(os.path.join(args.dpath, args.task.value), name='', version=args.seed),
                callbacks=[
                    ModelCheckpoint(monitor='val_acc', mode='max', filename='best')],
                max_epochs=args.n_epochs,
                deterministic=True)
            trainer.fit(model, data_train, [data_val, data_test])
        else:
            trainer = pl.Trainer(
                logger=CSVLogger(os.path.join(args.dpath, args.task.value, args.eval_stage.value), name='', version=args.seed),
                max_epochs=1,
                deterministic=True)
            trainer.test(model, data_eval)
    elif args.task == Task.VAE:
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, args.task.value), name='', version=args.seed),
            callbacks=[
                ModelCheckpoint(monitor='val_loss', filename='best')],
            max_epochs=args.n_epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            num_sanity_val_steps=0,
            deterministic=True,
            inference_mode=False)
        trainer.fit(model, data_train, [data_val, data_test])
    else:
        assert args.task == Task.CLASSIFY
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, args.task.value, args.eval_stage.value), name='', version=args.seed),
            max_epochs=1,
            deterministic=True,
            inference_mode=False)
        trainer.test(model, data_eval)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=Task, choices=list(Task), required=True)
    parser.add_argument('--eval_stage', type=EvalStage, choices=list(EvalStage))
    parser.add_argument('--train_ratio', type=int, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--z_size', type=int, default=64)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--y_mult', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--prior_reg_mult', type=float, default=1e-5)
    parser.add_argument('--init_sd', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20)
    main(parser.parse_args())