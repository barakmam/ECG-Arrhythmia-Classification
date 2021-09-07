import io
import PIL.Image as Image
import ast
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pytorch_lightning as pl
import wandb
from google.cloud import storage
import pickle
from torchvision import transforms
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


# Setting credentials using the downloaded JSON file
path = 'model-azimuth-321409-241148a4b144.json'
if not os.path.isfile(path):
    raise ("Please provide the gcs key in the root directory")
# client = storage.Client.from_service_account_json(json_credentials_path=path)
client = storage.Client.from_service_account_json(json_credentials_path='model-azimuth-321409-241148a4b144.json')
bucket = client.get_bucket('ecg-arrhythmia-classification')


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, is_train,data_dir, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.state = 'train' if is_train else 'test'
        self.classes = super_classes = ["CD", "HYP", "MI", "NORM", "STTC"]
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'train' or stage is None:
            data_full = ImageFolder(self.data_dir,transforms=self.transform)
            self.train, self.val = random_split(data_full, [round(len(data_full) * 0.8),
                                                           len(data_full)- round(len(data_full) * 0.8)])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = ImageFolder(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        # return DataLoader(self.test, batch_size=self.batch_size)
        return DataLoader(self.val, batch_size=self.batch_size)



