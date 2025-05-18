"""Script for training RainNet using PyTorch Lightning API."""
from pathlib import Path
import argparse
import random
import numpy as np

from utils.config import load_config
from utils.logging import setup_logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from pytorch_lightning.loggers import WandbLogger

from datamodule import DutchDataModule

from models import RainNet3D_OneShot as RN

import datetime
import os
os.chdir("/home/ppavlik/repos/ppavlik-rainguru")


def main(config, run_name, checkpoint=None, seed=1):
    confpath = Path("config") / config
    dsconf = load_config(confpath / "datasets.yaml")
    outputconf = load_config(confpath / "output.yaml")
    modelconf = load_config(confpath / "model.yaml")

    if run_name is None:
       run_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    seed = int(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_float32_matmul_precision('high')

    setup_logging(outputconf.logging)

    datamodel = DutchDataModule(dsconf, modelconf.train_params)

    model = RN(modelconf)

    # Callbacks
    model_ckpt = ModelCheckpoint(
        dirpath=f"/scratch/p709-24-2/checkpoints/{modelconf.train_params.savefile}/{run_name}",
        save_top_k=1,
        monitor="val_loss",
        save_on_train_epoch_end=False,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(**modelconf.train_params.early_stopping)
    device_monitor = DeviceStatsMonitor()
    logger = WandbLogger(save_dir=f"/scratch/p709-24-2",
                         project=modelconf.train_params.savefile, name=run_name, log_model=False, config={**dsconf, **modelconf})

    trainer = pl.Trainer(
        logger=logger,
        val_check_interval=modelconf.train_params.val_check_interval,
        max_epochs=modelconf.train_params.max_epochs,
        max_time=modelconf.train_params.max_time,
        devices=modelconf.train_params.gpus,
        limit_val_batches=modelconf.train_params.val_batches,
        limit_train_batches=modelconf.train_params.train_batches,
        callbacks=[
            early_stopping,
            model_ckpt,
            lr_monitor,
            device_monitor,
        ],
        log_every_n_steps=1,
        fast_dev_run=False,
    )

    trainer.fit(model=model, datamodule=datamodel, ckpt_path=checkpoint)

    trainer.test(model=model, datamodule=datamodel)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config", type=str, help="Configuration folder")
    argparser.add_argument(
        "-n",
        "--run_name",
        type=str,
        default=None,
        help="Run name")
    argparser.add_argument(
        "-c",
        "--continue_training",
        type=str,
        default=None,
        help="Path to checkpoint for model that is continued.",
    )
    argparser.add_argument(
        "-s",
        "--random_seed",
        type=str,
        default=None,
        help="Random seed to use for training.",
    )
    args = argparser.parse_args()
    main(args.config, args.run_name, args.continue_training, args.random_seed)
