import torch.utils.data as data
from typing import Tuple

from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan
from helpers.data_helper import DataHelper


class DataHelperJoint():

    @classmethod
    def create_dataloaders(
            cls,
            train_dataset: data.Dataset,
            validation_dataset: data.Dataset,
            batch_size: int,
            shuffle: bool = True) -> Tuple[data.DataLoader, data.DataLoader]:
        '''
        Creates DataLoader objects for the given datasets while 
        including padding and sorting
        '''

        train_loader = None
        if train_dataset:
            train_loader = data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                num_workers=1,
                shuffle=shuffle,
                collate_fn=cls._pad_and_sort_batch)

        validation_loader = None
        if validation_dataset:
            validation_loader = data.DataLoader(
                dataset=validation_dataset,
                batch_size=batch_size,
                num_workers=1,
                shuffle=shuffle,
                collate_fn=cls._pad_and_sort_batch)

        return train_loader, validation_loader

    @classmethod
    def _pad_and_sort_batch(cls, DataLoaderBatch):
        """
        DataLoaderBatch for the Hyperpartisan should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest, 
        """

        batch_split = list(zip(*DataLoaderBatch))

        metaphor_batch, hyperpartisan_batch = batch_split[0], batch_split[1]

        metaphor_result = DataHelper._pad_and_sort_batch(metaphor_batch)
        hyperpartisan_result = DataHelperHyperpartisan._pad_and_sort_batch(hyperpartisan_batch)

        return metaphor_result, hyperpartisan_result