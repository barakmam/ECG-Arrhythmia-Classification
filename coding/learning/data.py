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
from torch.utils.data import DataLoader, random_split, Dataset
import random


# Setting credentials using the downloaded JSON file
path = 'model-azimuth-321409-241148a4b144.json'
if not os.path.isfile(path):
    raise ("Please provide the gcs key in the root directory")
# client = storage.Client.from_service_account_json(json_credentials_path=path)
client = storage.Client.from_service_account_json(json_credentials_path='model-azimuth-321409-241148a4b144.json')
bucket = client.get_bucket('ecg-arrhythmia-classification')


class Pattern():
    def __init__(self, enc, percent):
        self.enc = enc
        self.percent = percent




class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, is_train,files_root, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.state = 'train' if is_train else 'test'
        self.classes = super_classes = ["CD", "HYP", "MI", "NORM", "STTC"]
        self.files_root = files_root
        self.number_of_files=sum([len(os.listdir(f'./STFT/STFT_{x}')) for x in self.classes])

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'train' or stage is None:
            self.train, self.val = random_split(self.files_root, [round(self.number_of_files * 0.8),
                                                           self.number_of_files - round(self.number_of_files * 0.8)])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = None
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        # return DataLoader(self.test, batch_size=self.batch_size)
        return DataLoader(self.val, batch_size=self.batch_size)



