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
            batch_size: int = 64,
            shuffle: bool = True) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
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

        test_loader = None
        if test_dataset:
            test_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                num_workers=1,
                shuffle=shuffle,
                collate_fn=cls._pad_and_sort_batch)

        return train_loader, validation_loader, test_loader

    @classmethod
    def _pad_and_sort_batch(cls, DataLoaderBatch):
        """
        DataLoaderBatch for the Hyperpartisan should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest, 
        """        
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        list_of_sequences, targets, list_of_lengths, num_of_sent = batch_split[0], batch_split[1], batch_split[2], batch_split[3]

        concat_lengths = np.array([length for lengths in list_of_lengths for length in lengths])
        concat_sequences = np.array([seq for sequences in list_of_sequences for seq in sequences])

        # descending order
        sorted_idx = np.argsort(-concat_lengths) 
        sorted_concat_lengths = concat_lengths[sorted_idx]
        sorted_concat_sequences = concat_sequences[sorted_idx]
        max_length = sorted_concat_lengths[0]

        padded_sequences = np.ones((sum(num_of_sent), max_length), dtype=np.int64)

        for i, l in enumerate(sorted_concat_lengths):
            padded_sequences[i][0:l] = sorted_concat_sequences[i][0:l]

        recover_idx = np.argsort(sorted_idx)

        return torch.LongTensor(padded_sequences), torch.FloatTensor(targets), torch.LongTensor(recover_idx), torch.LongTensor(num_of_sent)