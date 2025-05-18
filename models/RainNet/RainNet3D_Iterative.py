"""RainNet iterative model definition with definitions of custom loss functions."""

import numpy as np
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

from modelcomponents import RainNet3D as RN

class RainNet3D(pl.LightningModule):
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
        self.discount_rate = config.model.loss.discount_rate
        # equal weighting for each lt, sum to one.
        if self.discount_rate == 0:
            self.verif_loss_weights = (
                np.ones(self.verif_leadtimes) / self.verif_leadtimes
            )
        # Diminishing weight by n_lt^( - discount_rate), sum to one.
        else:
            verif_t = np.arange(1, self.verif_leadtimes + 1)
            self.verif_loss_weights = (
                verif_t ** (-self.discount_rate) / (verif_t ** (-self.discount_rate)).sum()
            )

        # optimization parameters
        self.lr = float(config.model.lr)
        self.lr_sch_params = config.train_params.lr_scheduler
        self.automatic_optimization = False

        # leadtime scheduling
        self.lt_sch_fin = False

        self.eth_loss_factor = config.model.loss.eth_loss_factor


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


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        y_hat, loss = self._iterative_prediction(batch=batch, stage="train")
        opt.step()
        opt.zero_grad()
        self.log("train_loss", loss['total_loss'])
        return {"prediction": y_hat, "loss": loss['total_loss']}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, loss = self._iterative_prediction(batch=batch, stage="valid")
        self.log("val_loss", loss['total_loss'])
        self.log("val_loss_pcp_only", loss['total_loss_pcp_only'])
        self.log("leadtime", self.train_leadtimes)

        if batch_idx == 0:
            grid = torchvision.utils.make_grid(y_hat[:4,0:16:4, 0].flatten(0,1).unsqueeze(1), nrow=4, padding=8, normalize=True, value_range=(0,10))
            self.logger.log_image(key="nowcasts_valid", images=[grid])

        return {"prediction": y_hat, "loss": loss['total_loss']}

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, loss = self._iterative_prediction(batch=batch, stage="test")
        self.log("test_loss", loss['total_loss'])
        return {"prediction": y_hat, "loss": loss['total_loss']}
    
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
        x = x.float()
        y = y.float()
        y_seq = torch.empty(
            (x.shape[0], n_leadtimes, *self.input_shape[1:]), device=self.device
        )
        if calculate_loss:
            total_loss = 0
            total_loss_pcp_only = 0

        for i in range(n_leadtimes):
            y_hat = self(x)
            if calculate_loss:
                y_i = y[:, :, None, i, :, :].clone()
                if stage == "valid" or stage == "test":
                    loss_pcp_only = self.criterion(y_hat[:,0], y_i[:,0]) * loss_weights[i]
                    total_loss_pcp_only += loss_pcp_only.detach()
                loss = (self.criterion(y_hat[:,0], y_i[:,0]) + self.eth_loss_factor * self.criterion(y_hat[:,1], y_i[:,1])) * loss_weights[i]
                total_loss += loss.detach()
                if stage == "train":
                    self.manual_backward(loss)
                del y_i
            y_seq[:, i, :, :, :] = y_hat.detach().squeeze(dim=2)
            if i != n_leadtimes - 1:
                x = torch.roll(x, -1, dims=2)
                x[:, :, -1, :, :] = y_hat.detach().squeeze(dim=2)
            del y_hat
        if calculate_loss:
            if stage == "valid":
                return y_seq, {'total_loss': total_loss, 'total_loss_pcp_only': total_loss_pcp_only}
            else:
                return y_seq, {'total_loss': total_loss}
        else:
            return y_seq
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Get data
        x, y, idx = batch

        # Perform prediction with RainNet model
        y_seq = self._iterative_prediction(batch=(x, y, idx), stage="predict")

        y_seq = y_seq[:,:,0]

        y_seq = self.trainer.datamodule.predict_dataset.from_transformed(
            y_seq
        )

        del x
        return y_seq
    