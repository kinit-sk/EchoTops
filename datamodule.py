"""A datamodule for working with Dutch dataset for training, validation and testing that has files in h5 format."""
import torch
import pytorch_lightning as pl
from torch.utils.data import Sampler, DataLoader
from torchvision.transforms.functional import crop, hflip, vflip, rotate

from dataset import DutchDataset
import random

class SetShuffleSampler(Sampler[int]):
    def __init__(self, length):
        self.list = list(range(0,length))
        random.shuffle(self.list)

    def __len__(self) -> int:
        return len(self.list)

    def __iter__(self):
        yield from self.list

class DutchDataModule(pl.LightningDataModule):
    def __init__(self, dsconfig, train_params, seed=1):
        super().__init__()
        self.dsconfig = dsconfig
        self.train_params = train_params
        self.seed = seed

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage):
        # called on every GPU
        if stage == "fit":
            self.train_dataset = DutchDataset(
                split="train", valid_part=self.seed, **self.dsconfig.SHMUDataset
            )
            self.valid_dataset = DutchDataset(
                split="valid", valid_part=self.seed, **self.dsconfig.SHMUDataset
            )
        if stage == "test":
            self.test_dataset = DutchDataset(
                split="test", **self.dsconfig.SHMUDataset
            )
        if stage == "predict":
            self.predict_dataset = DutchDataset(
                split="test", **self.dsconfig.SHMUDataset
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_params.train_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            sampler=SetShuffleSampler(len(self.valid_dataset)),
            batch_size=self.train_params.valid_batch_size,
            num_workers=self.train_params.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_params.test_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return test_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size=self.train_params.test_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return predict_loader
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y, idx = batch
        if self.trainer.training:
            x, y = self.apply_augments(x, y)
        return x, y, idx
    
    def apply_augments(self, x, y):
        if self.dsconfig.augmentations.horizontal_flip:
            if random.random() >= 0.5:
                x = hflip(x)
                y = hflip(y)

        if self.dsconfig.augmentations.vertical_flip:
            if random.random() >= 0.5:
                x = vflip(x)
                y = vflip(y)

        return x, y  

def _collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

