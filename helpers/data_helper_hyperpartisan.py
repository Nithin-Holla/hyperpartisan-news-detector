import numpy as np

import torch
import torch.utils.data as data

from typing import Tuple


class DataHelperHyperpartisan():

    @classmethod
    def create_dataloaders(
            cls,
            train_dataset: data.Dataset = None,
            validation_dataset: data.Dataset = None,
            test_dataset: data.Dataset = None,
            batch_size: int = 32,
            shuffle: bool = True,
            drop_last: bool = True) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
        '''
        Creates DataLoader objects for the given datasets while 
        including padding and sorting
        '''

        train_loader = None
        if train_dataset:
            train_loader = data.DataLoader(
                dataset=train_dataset,
                batch_size=1,
                num_workers=1,
                shuffle=shuffle,
                drop_last=drop_last)

        validation_loader = None
        if validation_dataset:
            validation_loader = data.DataLoader(
                dataset=validation_dataset,
                batch_size=1,
                num_workers=1,
                shuffle=shuffle,
                drop_last=drop_last)

        test_loader = None
        if test_dataset:
            test_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=1,
                num_workers=1,
                shuffle=shuffle,
                drop_last=drop_last)

        return train_loader, validation_loader, test_loader