import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt


class LossCurveCallback(pl.Callback):
    '''PyTorch Lightning metric callback.
    I think it is very clunky to make a callback class just to plot loss curve
    but I guess that's what you get if you want lightning to tbe minimalist
    https://forums.pytorchlightning.ai/t/how-to-access-the-logged-results-such-as-losses/155
    '''

    def __init__(self, lnp, figsize=[15, 10]):
        super().__init__()
        self.lnp = lnp
        self.figsize = figsize

    def on_init_end(self, trainer):
        self.a_trn_loss = np.ones(trainer.max_epochs) * np.inf
        self.a_val_loss = np.ones(trainer.max_epochs) * np.inf

    def on_validation_epoch_end(self, trainer, pl_module):
        self.lnp.lnp('EPOCH' +' '+ str(trainer.current_epoch) +' '+
                     'sanity_checking' +' '+ str(trainer.sanity_checking) +' '+
                     str(trainer.callback_metrics))
        if not trainer.sanity_checking: # WARN: sanity_check is turned on by default
            self.a_trn_loss[trainer.current_epoch] = trainer.callback_metrics['trn_loss']
            self.a_val_loss[trainer.current_epoch] = trainer.callback_metrics['val_loss']

    def on_train_end(self,trainer,pl_module):
        '''TODO: hp_metric won't update at on_fit_end
        If I use on_fit_end and:
        If I use self.log(), I will get MisconfigurationException
        If I use add_scalar(), It won't update, it will stay at -1
        '''
        '''Note: on_fit_end was destroyed at on_fit_end, but on_train_end
        pl_module.log('minval', self.a_val_loss.min())
        pl_module.log('hp_metric', self.a_val_loss.min())
        pl_module.log('aminval', self.a_val_loss.argmin())
        MisconfigurationException: on_fit_end function doesn't support logging using `self.log()`
        '''
        trainer.logger[0].experiment.add_scalar('aminval', self.a_val_loss.argmin())
        trainer.logger[0].experiment.add_scalar('minval', self.a_val_loss.min())
        trainer.logger[1].experiment.summary['aminval'] = self.a_val_loss.argmin()
        trainer.logger[1].experiment.summary['minval'] = self.a_val_loss.min()
        # trainer.logger.experiment.add_scalar('hp_metric', self.a_val_loss.min())
        self.lnp.lnp('aminval ' + str(self.a_val_loss.argmin()))
        self.lnp.lnp('minval ' + str(self.a_val_loss.min()))
        # self.lnp.lnp('hp_metric ' + str(self.a_val_loss.min()))
        ''' hp_metric
        This the metric that Tensorboard will use to compare between runs to pick the best hyperparameters.
        Ideally, it is a single scalar number per run.
        '''

        f,a = plt.subplots(figsize=self.figsize)
        a.set_title('Loss curve')
        a.plot(self.a_trn_loss, label='trn_loss')
        a.plot(self.a_val_loss, label='val_loss')
        # TODO: twinx and plot LR. Maybe we can see drop when the LR drop
        # TODO: twinx and plot LR. Maybe plot gradient l2 norm sum
        a.set_xlabel('Epoch')
        a.set_ylabel('Loss')
        a.vlines(x=self.a_val_loss.argmin(), ymin=self.a_val_loss.min(), ymax=self.a_val_loss[:trainer.current_epoch].max(),
                 label='lowest validation = '+str(self.a_val_loss.min())+' at '+str(self.a_val_loss.argmin()))
        a.legend()
        trainer.logger[0].experiment.add_figure('loss_curve', f)
        #trainer.logger[1].experiment.log({'loss_curve': f})

        f,a = plt.subplots(figsize=self.figsize)
        a.set_title('Loss curve')
        a.semilogy(self.a_trn_loss, label='trn_loss')
        a.semilogy(self.a_val_loss, label='val_loss')
        a.set_xlabel('Epoch')
        a.set_ylabel('Loss')
        a.vlines(x=self.a_val_loss.argmin(), ymin=self.a_val_loss.min(), ymax=self.a_val_loss[:trainer.current_epoch].max(),
                 label='lowest validation = '+str(self.a_val_loss.min())+' at '+str(self.a_val_loss.argmin()))
        a.legend()
        trainer.logger[0].experiment.add_figure('loss_curve_log', f)
        #trainer.logger[1].experiment.log({'loss_curve_log': f})