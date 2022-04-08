import numpy as np
import pytorch_lightning as pl

class LogParameters(pl.Callback):
    # weight and biases to tensorbard
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        self.d_parameters = {}
        for n,p in pl_module.named_parameters():
            self.d_parameters[n] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking: # WARN: sanity_check is turned on by default
            lp = []
            for n,p in pl_module.named_parameters():
                trainer.logger[0].experiment.add_histogram(n, p.data, trainer.current_epoch)
                # TODO add histogram to wandb too
                self.d_parameters[n].append(p.ravel().cpu().numpy())
                lp.append(p.ravel().cpu().numpy())
            p = np.concatenate(lp)
            trainer.logger[0].experiment.add_histogram('Parameters', p, trainer.current_epoch)
            # TODO add histogram to wandb too