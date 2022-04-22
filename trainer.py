import argparse
import json
import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl

import time

from callbacks.logParameters import LogParameters
from callbacks.lossCurveCallback import LossCurveCallback
from loggers.log import Log_and_print
from models.qAModel import VQAModelClassifier
from utils import collect_env_details

lstr_args = ['--max_epochs','3']

def cli_main(parser):

    print('MAIN START')
    ts_script = time.time()


    # ------------
    # args
    # ------------
    print('MAIN args')
    # program level args
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_dir', default='./output/results', type=str)
    parser.add_argument('--run_name', default='default_run', type=str)
    parser.add_argument('--log_parameters', default=0, type=int)
    parser.add_argument('--es_patience', default=20, type=int)
    parser.add_argument('--figsize_x', default=15, type=float)
    parser.add_argument('--figsize_y', default=10, type=float)
    # trainer level args
    parser = pl.Trainer.add_argparse_args(parser)
    # model level args
    parser = VQAModelClassifier.add_model_specific_args(parser)
    args = parser.parse_args(lstr_args)
    # always print full weights_summary
    args.weights_summary = 'full'
    # automatically use all available GPUs
    # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#select-gpu-devices
    args.gpus = -1


    # seed
    pl.seed_everything(args.seed)

    # logger
    # https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#tensorboard
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html
    # loggers need info from args, so have to run args first before loggers
    tb_logger = pl.loggers.TensorBoardLogger(save_dir = args.save_dir+'log/',
                                             name = args.run_name,
                                             version = 'fixed_version',
                                             log_graph = True)
    '''
    The tensorboard is creating a new version unless we fix it with a new version name.
    '''
    wandb_logger = pl.loggers.WandbLogger(save_dir = args.save_dir+'log/',
                                          offline = True, # cannot log model while offline
                                          name = args.run_name,
                                          version = 'fixed_version')
    lnp = Log_and_print(tb_logger, wandb_logger, args.run_name)
    lnp.lnp('Loggers start')
    lnp.lnp('ts_script: ' + str(ts_script))

    sys.path += [os.path.abspath(".."), os.path.abspath(".")]
    lnp.lnp(collect_env_details())

    strargs = ''
    for (k,v) in vars(args).items():
        strargs += str(k) + ': ' + str(v) + '\n'
    lnp.lnp('ARGUMENTS:\n' + strargs)

    # ------------
    # LightningModule
    # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    # ------------
    lnp.lnp('MAIN LightningModule')
    lm = VQAModelClassifier(**vars(args))
    for n,p in lm.named_parameters():
        lnp.lnp(n + ': ' + str(p.data.shape))

    # ------------
    # training
    # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
    # ------------

    # Callbacks
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html
    lnp.lnp('MAIN callbacks')
    l_callbacks = []

    # custom loss curve
    cbLossCurveCallback = LossCurveCallback(lnp, figsize=[args.figsize_x, args.figsize_y]) # custom
    l_callbacks.append(cbLossCurveCallback)

    # early stopping
    # https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html
    #cbEarlyStopping = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=args.es_patience)
    #l_callbacks.append(cbEarlyStopping)

    # model checkpoint
    # https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html#automatic-saving
    checkpoint_dirpath = args.save_dir + 'checkpoints/'
    checkpoint_filename = args.save_dir[:-1] + '_' + args.run_name
    lnp.lnp('checkpoint_dirpath: ' + checkpoint_dirpath)
    lnp.lnp('checkpoint_filename: ' + checkpoint_filename)
    cbModelCheckpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=checkpoint_dirpath, filename=checkpoint_filename)
    l_callbacks.append(cbModelCheckpoint)

    # log parameters
    if args.log_parameters:
        cbLogParameters = LogParameters()
        l_callbacks.append(cbLogParameters)

    lnp.lnp('MAIN trainer')
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=[tb_logger, wandb_logger],
                                            callbacks=l_callbacks,
                                            )
    trainer.val_percent_check = 0
    trainer.check_val_every_n_epoch = 3

    # LEARNING RATE FINDER
    # https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html#learning-rate-finder
    # MisconfigurationException: No `train_dataloader()` method defined. Lightning `Trainer` expects as minimum a `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined.
    """lr_finder = trainer.tuner.lr_find(lm)
    fig = lr_finder.plot(suggest=True)
    trainer.logger[0].experiment.add_figure('lr_finder', fig)
    trainer.logger[1].experiment.log({'lr_finder': fig})
    # TODO: log to wandb too https://docs.wandb.ai/guides/track/log/plots#matplotlib-and-plotly-plots
    new_lr = lr_finder.suggestion()
    if new_lr is None:
        lnp.lnp('new_lr was not found. Using default lr: ' + str(lm.hparams.lr))
    else:
        lnp.lnp('new_lr: ' + str(new_lr))
        lm.hparams.lr = new_lr"""

    # fit
    lnp.lnp('MAIN fit')

    """train_dataload = train_dataloader()
    val_dataload = val_dataloader()
    trainer.fit(lm, train_dataload, val_dataload)"""
    trainer.fit(lm)


    # ------------
    # testing
    # ------------
    lnp.lnp('MAIN test')
    ts_test = time.time()
    # test from the best checkpoint
    # https://pytorch-lightning.readthedocs.io/en/latest/common/test_set.html
    test_output = trainer.test(ckpt_path = 'best')
    tf_test = time.time()
    lnp.lnp('test_output: ' + str(test_output))
    lnp.lnp('ts_test: ' + str(ts_test))
    lnp.lnp('tf_test: ' + str(tf_test))
    dur_test = tf_test - ts_test
    lnp.lnp('Test duration: ' + str(dur_test))

    # JSON log
    dlog = {'lowest_val' : float(cbLossCurveCallback.a_val_loss.min()),
            'a_lowest_val' : int(cbLossCurveCallback.a_val_loss.argmin()),
            'current_epoch' : trainer.current_epoch,
            'dur_test' : dur_test,
            'tst_loss' : test_output[0]['tst_loss'],
            'new_lr' : lm.hparams.lr,
            'n_trainable_params' : sum(p.numel() for p in lm.parameters() if p.requires_grad)}
    dlog.update(vars(args))

    tf_script = time.time()
    lnp.lnp('tf_script: ' + str(tf_script))
    dur_script = tf_script - ts_script
    lnp.lnp('Script duration: ' + str(dur_script))

    dlog['dur_script'] = dur_script

    # Tear down
    lnp.lnp('MAIN END (only logging is left)')
    with open(args.save_dir + '/results' + '_' + args.run_name +'.json', "w") as outfile:
        json.dump(dlog, outfile)
    lnp.dump_to_tensorboard()
    lnp.dump_to_wandb()
    print('everything done')

    return trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full Pipeline Training')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='Shorter side transformation.')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='Shorter side transformation.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Shorter side transformation.')
    parser.add_argument('--n_val', type=int, default=5000,
                        help='Shorter side transformation.')
    parser.add_argument('--n_train', type=int, default=10000,
                        help='Shorter side transformation.')
    parser.add_argument('--n_test', type=int, default=5000,
                        help='Shorter side transformation.')
    parser.add_argument('--train_data_dir', type=str, default="./data",
                        help='Shorter side transformation.')
    parser.add_argument('--val_data_dir', type=str, default="./data",
                        help='Shorter side transformation.')
    parser.add_argument('--test_data_dir', type=str, default="./data",
                        help='Shorter side transformation.')
    parser.add_argument('--train_answersDataSubType', type=str, default="train2014",
                        help='Shorter side transformation.')
    parser.add_argument('--val_answersDataSubType', type=str, default="val2014",
                        help='Shorter side transformation.')
    parser.add_argument('--test_answersDataSubType', type=str, default="val2014",
                        help='Shorter side transformation.')
    parser.add_argument('--train_questionDataSubType', type=str, default="train2014",
                        help='Shorter side transformation.')
    parser.add_argument('--val_questionDataSubType', type=str, default="val2014",
                        help='Shorter side transformation.')
    parser.add_argument('--test_questionDataSubType', type=str, default="val2014",
                        help='Shorter side transformation.')
    parser.add_argument('--numCandidates', type=int, default=5,
                        help='Shorter side transformation.')
    parser.add_argument('--trainPklFilePath', type=str, default='./output/intermediate/trainNormalisedFeatures.pkl',
                        help='Shorter side transformation.')
    parser.add_argument('--valPklFilePath', type=str, default='./output/intermediate/normalisedFeatures.pkl',
                        help='Shorter side transformation.')
    parser.add_argument('--resultsTrain', type=str, default='resultsTrain',
                        help='Shorter side transformation.')
    parser.add_argument('--resultsVal', type=str, default='resultsVal',
                        help='Shorter side transformation.')
    trainer = cli_main(parser)

