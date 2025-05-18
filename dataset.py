"""PyTorch Dataset for working with radar files that are in h5 format."""
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import pandas as pd
import os
from tqdm import tqdm
from functools import reduce


class DutchDataset(Dataset):
    """A dataset for working with Dutch radar files that are in h5 format."""

    def __init__(
        self,
        split="train",
        path=None,
        path_eth=None,
        metadata=None,
        input_block_length=4,
        prediction_block_length=18,
        bbox=None,
        image_size=None,
        transform_to_mmh=True,
        normalization_method='none',
        timestep=5,
        valid_part=0,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        split : {'train', 'test', 'valid'}
            The type of the dataset: training, testing or validation.
        path : str
            Path to the metadata file.
        path : str
            Format of the data path. May contain the tokens {year:*}, {month:*},
            {day:*}, {hour:*}, {minute:*}, {second:*} that are substituted when
            going through the dates.
        input_block_length : int
            The number of frames to be used as input to the models.
        prediction_block_length : int
            The number of frames that are predicted and tested against the
            observations.
        timestep : int
            Time step of the data (minutes).
        bbox : str
            Bounding box of the data in the format '[x1, x2, y1, x2]'.
        image_size : str
            Shape of the images after the bounding box in the format
            '[width, height]'.
        """
        assert path is not None, "No path to radar files provided!"
        assert metadata is not None, "No metadata file for radar files provided!"

        # Inherit from parent class
        super().__init__()

        # Load metadata
        self.metadata = pd.read_csv(metadata, index_col=0)

        self.split = split

        train_cutoff_date = '2016-10-01'
        test_cutoff_date = '2022-01-01'
        test_end_date = '2023-01-01'

        valid_split = 8

        if split == 'test':
            self.datelist = pd.to_datetime(self.metadata[(self.metadata.seq_start == True) &
                                                         (self.metadata.datetime < test_end_date) &
                                                         (self.metadata.datetime >= test_cutoff_date)].datetime)
            self.seq_start_indices = np.array(self.metadata[(self.metadata.seq_start == True) &
                                                            (self.metadata.datetime < test_end_date) &
                                                            (self.metadata.datetime >= test_cutoff_date)].index)
        elif split == 'train':
            self.datelist = pd.to_datetime(self.metadata[(self.metadata.seq_start == True) &
                                                         (self.metadata.datetime < test_cutoff_date) &
                                                         (self.metadata.datetime >= train_cutoff_date) &
                                                         (self.metadata.group % valid_split != valid_part)].datetime)
            self.seq_start_indices = np.array(self.metadata[(self.metadata.seq_start == True) &
                                                            (self.metadata.datetime < test_cutoff_date) &
                                                            (self.metadata.datetime >= train_cutoff_date) &
                                                            (self.metadata.group % valid_split != valid_part)].index)
        elif split == 'valid':
            self.datelist = pd.to_datetime(self.metadata[(self.metadata.seq_start == True) &
                                                         (self.metadata.datetime < test_cutoff_date) &
                                                         (self.metadata.datetime >= train_cutoff_date) &
                                                         (self.metadata.group % valid_split == valid_part)].datetime)
            self.seq_start_indices = np.array(self.metadata[(self.metadata.seq_start == True) &
                                                            (self.metadata.datetime < test_cutoff_date) &
                                                            (self.metadata.datetime >= train_cutoff_date) &
                                                            (self.metadata.group % valid_split == valid_part)].index)
        else:
            raise ValueError("split \"" + self.split  + "\" not available")
        
        indices = []
        for i in range(22):
            indices.append(self.seq_start_indices + i)
        self.full_indices = reduce(np.union1d, indices)
        self.seq_start_indices_remap = np.searchsorted(self.full_indices, self.seq_start_indices)

        self.path = path
        self.path_eth = path_eth

        self.image_size = image_size
        if bbox is None:
            self.use_bbox = False
        else:
            self.use_bbox = True
            self.bbox_x_slice = slice(bbox[0], bbox[1])
            self.bbox_y_slice = slice(bbox[2], bbox[3])

        self.num_frames_input = input_block_length
        self.num_frames_output = prediction_block_length
        self.num_frames = input_block_length + prediction_block_length

        self.transform_to_mmh = transform_to_mmh

        self.common_time_index = self.num_frames_input - 1

        if normalization_method not in ["log", "none", "max1"]:
            raise NotImplementedError(
                f"data normalization method {normalization_method} not implemented"
            )
        else:
            self.normalization = normalization_method
        
        self.timestep = timestep

        # Load raw data
        self.full_dataset = np.empty((len(self.full_indices), *self.image_size))
        for i, name in enumerate(tqdm(self.metadata.filename.loc[self.full_indices], total=len(self.full_indices))):
            fn = os.path.join(self.path, name)

            im = read_h5_composite(fn)
            if self.use_bbox:
                im = im[self.bbox_x_slice, self.bbox_y_slice]
            self.full_dataset[i, ...] = im
        
        # Load raw data - echotops
        if self.path_eth:
            self.full_dataset_eth = np.empty((len(self.full_indices), *self.image_size))
            for i, name in enumerate(tqdm(self.metadata.filename_eth.loc[self.full_indices], total=len(self.full_indices))):
                fn = os.path.join(self.path_eth, name)

                im = read_h5_echotops(fn)
                if self.use_bbox:
                    im = im[self.bbox_x_slice, self.bbox_y_slice]
                self.full_dataset_eth[i, ...] = im

    def __len__(self):
        """Mandatory property for Dataset."""
        return len(self.datelist)

    def __getitem__(self, idx):
        """Mandatory property for fetching data."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.full_dataset[self.seq_start_indices_remap[idx]:self.seq_start_indices_remap[idx]+self.num_frames]
        data = data[..., np.newaxis]

        if self.path_eth:
            data_eth = self.full_dataset_eth[self.seq_start_indices_remap[idx]:self.seq_start_indices_remap[idx]+self.num_frames]
            data_eth = data_eth[..., np.newaxis]
            data = np.concatenate([data, data_eth], axis=-1)
        
        inputs, outputs = self.postprocessing(data)

        return inputs, outputs, idx

    def from_transformed(self, data, scaled=True):
        if scaled:
            data = self.invScaler(data)  # to mm/h
        
        if self.transform_to_mmh:
            data = 200 * data ** (1.6)  # to z
            data = 10 * torch.log10(data + 1)  # to dBZ

        return data
    
    def postprocessing(self, data_in: np.ndarray):
        data = torch.Tensor(data_in)
        if self.transform_to_mmh:
            # dbZ to mm/h
            data[...,0] = 10 ** (data[...,0] * 0.1)
            data[...,0] = (data[...,0] / 200) ** (1 / 1.6)

        # to log-transformed
        data = self.scaler(data)

        # Divide to input & output
        if self.num_frames_output == 0:
            inputs = data
            outputs = torch.empty((0, data.shape[1], data.shape[2]))
        else:
            inputs = data[: -self.num_frames_output, ...].permute(3, 0, 1, 2).contiguous()
            outputs = data[-self.num_frames_output :, ...].permute(3, 0, 1, 2).contiguous()

        return inputs, outputs

    def get_common_time(self, index):
        return self.datelist.iloc[index]
    
    def scaler(self, data: torch.Tensor):
        if self.normalization == "log":
            return torch.log(data + 0.01)
        if self.normalization == "max1":
            data[..., 0] = data[..., 0] / 80
            if self.path_eth:
                data[..., 1] = data[..., 1] / 16
            data = np.clip(data, 0, 1)
            return data
        if self.normalization == "none":
            return data

    def invScaler(self, data: torch.Tensor):
        if self.normalization == "log":
            return torch.exp(data) - 0.01
        if self.normalization == "max1":
            return data * 80
        if self.normalization == "none":
            return data


def read_h5_composite(filename):
    """"Read h5 composite."""
    with h5py.File(filename, "r") as hf:
        data = np.array(hf['image1']['image_data'])
        data =  data * 0.5 - 31.5
    return data

def read_h5_echotops(filename):
    with h5py.File(filename, "r") as hf:
        PV = np.array(hf['image1']['image_data'])
        formula = hf['image1']['calibration'].attrs.get('calibration_formulas')
        if isinstance(formula, np.ndarray):
            formula = formula[0].decode('utf-8').split('=')[1].strip()
        else:
            formula = formula.decode('utf-8').split('=')[1].strip()
        PV = np.float16(PV)
        PV[PV == 255] = 0
        return eval(formula)
