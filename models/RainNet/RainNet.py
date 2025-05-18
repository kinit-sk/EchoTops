"""RainNet iterative model definition with definitions of custom loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from modelcomponents import RainNet as RN

class RainNet(pl.LightningModule):
    """Model for the RainNet iterative neural network."""

    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()

        self.input_shape = config.model.rainnet.input_shape
        self.personal_device = torch.device(config.train_params.device)
        self.network = RN(
            kernel_size=config.model.rainnet.kernel_size,
            mode=config.model.rainnet.mode,
            im_shape=self.input_shape[1:],  # x,y
            conv_shape=config.model.rainnet.conv_shape,
        )

        if config.model.loss.name == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss {config.model.loss.name} not implemented!")
        

        # on which leadtime to train the NN on?
        self.train_leadtimes_max = config.model.train_leadtimes
        self.train_leadtimes = self.train_leadtimes_max
        self.verif_leadtimes = config.train_params.verif_leadtimes
        # How many leadtimes to predict
        self.predict_leadtimes = config.prediction.predict_leadtimes

        # 1.0 corresponds to harmonic loss weight decrease,
        # 0.0 to no decrease at all,
        # less than 1.0 is sub-harmonic,
        # more is super-harmonic
        discount_rate = config.model.loss.discount_rate
        # equal weighting for each lt, sum to one.
        if discount_rate == 0:
            self.train_loss_weights = (
                np.ones(self.train_leadtimes) / self.train_leadtimes
            )
            self.verif_loss_weights = (
                np.ones(self.verif_leadtimes) / self.verif_leadtimes
            )
        # Diminishing weight by n_lt^( - discount_rate), sum to one.
        else:
            train_t = np.arange(1, self.train_leadtimes + 1)
            self.train_loss_weights = (
                train_t ** (-discount_rate) / (train_t ** (-discount_rate)).sum()
            )
            verif_t = np.arange(1, self.verif_leadtimes + 1)
            self.verif_loss_weights = (
                verif_t ** (-discount_rate) / (verif_t ** (-discount_rate)).sum()
            )

        # optimization parameters
        self.lr = float(config.model.lr)
        self.lr_sch_params = config.train_params.lr_scheduler
        self.automatic_optimization = False

        # leadtime scheduling
        self.lt_sch_fin = False

        self.eth_loss_factor = config.model.loss.eth_loss_factor
    
    def on_train_epoch_start(self):
        fib = [1,1,2,3,5,8,13,21]
        if not self.lt_sch_fin:
            self.train_leadtimes = min(fib[self.current_epoch], self.train_leadtimes_max)
            if fib[self.current_epoch] >= self.train_leadtimes_max:
                self.lt_sch_fin = True
        

        # equal weighting for each lt, sum to one.
        if self.discount_rate == 0:
            self.train_loss_weights = (
                np.ones(self.train_leadtimes) / self.train_leadtimes_max
            )
        # Diminishing weight by n_lt^( - discount_rate), sum to one.
        else:
            train_t = np.arange(1, self.train_leadtimes + 1)
            self.train_loss_weights = (
                train_t ** (-self.discount_rate) / (train_t ** (-self.discount_rate)).sum()
            )

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_sch_params.name is None:
            return optimizer
        elif self.lr_sch_params.name == "reduce_lr_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, **self.lr_sch_params.kwargs
            )
            return [optimizer], [lr_scheduler]
        else:
            raise NotImplementedError("Lr scheduler not defined.")

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        y_hat, loss = self._iterative_prediction(batch=batch, stage="train")
        opt.step()
        opt.zero_grad()
        self.log("train_loss", loss)
        return {"prediction": y_hat, "loss": loss}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, loss = self._iterative_prediction(batch=batch, stage="valid")
        self.log("val_loss", loss)
        self.log("leadtime", self.train_leadtimes)
        return {"prediction": y_hat, "loss": loss}

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, loss = self._iterative_prediction(batch=batch, stage="test")
        self.log("test_loss", loss)
        return {"prediction": y_hat, "loss": loss}
    
    def _iterative_prediction(self, batch, stage):

        if stage == "train":
            n_leadtimes = self.train_leadtimes
            calculate_loss = True
            loss_weights = self.train_loss_weights
        elif stage == "valid" or stage == "test":
            n_leadtimes = self.verif_leadtimes
            calculate_loss = True
            loss_weights = self.verif_loss_weights
        elif stage == "predict":
            n_leadtimes = self.predict_leadtimes
            calculate_loss = False
        else:
            raise ValueError(
                f"Stage {stage} is undefined. \n choices: 'train', 'valid', test', 'predict'"
            )

        x, y, _ = batch

        channels = x.shape[1]

        x_new = torch.zeros(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4], device=self.device)
        if channels == 2:
            x_new[:, ::2, :] = x[:,0].clone()
            x_new[:, 1::2, :] = x[:,1].clone()
        x = x_new

        y_new = torch.zeros(y.shape[0],y.shape[1]*y.shape[2],y.shape[3],y.shape[4], device=self.device)
        if channels == 2:
            y_new[:, ::2, :] = y[:,0].clone()
            y_new[:, 1::2, :] = y[:,1].clone()
        y = y_new

        y_seq = torch.empty(
            (x.shape[0], n_leadtimes, *self.input_shape[1:]), device=self.device
        )
        if calculate_loss:
            total_loss = 0

        for i in range(n_leadtimes):
            y_hat = self(x)
            if calculate_loss:
                y_i = y[:, None, channels * i, :, :].clone()
                y_eth_i = y[:, None, channels * i + 1, :, :].clone()
                loss = (self.criterion(y_hat[:, 0:1], y_i) + self.eth_loss_factor * self.criterion(y_hat[:, 1:2], y_eth_i)) * loss_weights[i]
                total_loss += loss.detach()
                if stage == "train":
                    self.manual_backward(loss)
                del y_i
            y_seq[:, i] = y_hat[:, 0].detach().squeeze()
            if i != n_leadtimes - 1:
                x = torch.roll(x, -channels, dims=1)
                x[:, -channels:, :, :] = y_hat.detach().squeeze()
            del y_hat
        if calculate_loss:
            return y_seq, total_loss
        else:
            return y_seq
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Get data
        x, y, idx = batch

        # Perform prediction with RainNet model
        y_seq = self._iterative_prediction(batch=(x, y, idx), stage="predict")

        #y_seq = y_seq[:,:,0] select only precip

        y_seq = self.trainer.datamodule.predict_dataset.from_transformed(
            y_seq
        )

        del x
        return y_seq
    