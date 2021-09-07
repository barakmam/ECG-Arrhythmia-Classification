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
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils import check_integrity
import random


# Setting credentials using the downloaded JSON file
path = 'model-azimuth-321409-241148a4b144.json'
if not os.path.isfile(path):
    raise ("Please provide the gcs key in the root directory")
# client = storage.Client.from_service_account_json(json_credentials_path=path)
client = storage.Client.from_service_account_json(json_credentials_path='model-azimuth-321409-241148a4b144.json')
bucket = client.get_bucket('ecg-arrhythmia-classification')


class PtbData(Dataset):
    def __init__(self, data_dir, number_of_files,is_train):
        super().__init__(data_dir)

        self.is_train = is_train  # training set or test set
        self.number_of_files=number_of_files

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)




class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, is_train,data_dir, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.state = 'train' if is_train else 'test'
        self.classes = super_classes = ["CD", "HYP", "MI", "NORM", "STTC"]
        self.data_dir = data_dir
        self.number_of_files=sum([len(os.listdir(f'./STFT/STFT_{x}')) for x in self.classes])

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'train' or stage is None:
            data_full = PtbData(self.data_dir,number_of_files=self.number_of_files, train=True)
            self.train, self.val = random_split(data_full, [round(self.number_of_files * 0.8),
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



