import time

class Log_and_print():
    # need this to ensure that stuff are printed to STDOUT as well for backup
    '''
    https://stackoverflow.com/questions/45016458/tensorflow-tf-summary-text-and-linebreaks
    Tensorboard text uses the markdown format.
    That means you need to add 2 spaces before \n to produce a linebreak
    '''
    def __init__(self, tb_logger, wandb_logger, run_name):
        self.tb_logger = tb_logger
        self.wandb_logger = wandb_logger
        self.run_name = run_name
        self.str_log = ('PARTIAL COPY OF TEXT LOG TO TENSORBOARD TEXT  \n'
                        'class Log_and_print() by Arian Prabowo  \n'
                        'RUN NAME: ' + run_name + '  \n  \n')

    def lnp(self, tag):
        print(self.run_name, time.asctime(), tag)
        self.str_log += str(time.asctime()) + ' ' + str(tag) + '  \n'

    def dump_to_tensorboard(self):
        self.tb_logger.experiment.add_text('log', self.str_log)

    def dump_to_wandb(self):
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.WandbLogger.html#pytorch_lightning.loggers.WandbLogger.experiment
        # https://docs.wandb.ai/guides/track/log#summary-metrics
        self.wandb_logger.experiment.summary['log'] = self.str_log