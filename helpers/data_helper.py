import numpy as np

import torch
import torch.utils.data as data

from typing import Tuple


class DataHelper():

    @classmethod
    def create_dataloaders(
            cls,
            train_dataset: data.Dataset = None,
            validation_dataset: data.Dataset = None,
            test_dataset: data.Dataset = None,
            batch_size: int = 32,
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
                shuffle=False,
                collate_fn=cls._pad_and_sort_batch)

        test_loader = None
        if test_dataset:
            test_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                num_workers=1,
                shuffle=False,
                collate_fn=cls._pad_and_sort_batch)

        return train_loader, validation_loader, test_loader

    @classmethod
    def _sort_batch(cls, batch, targets, lengths):
        """
        Sort a minibatch by the length of the sequences with the longest sequences first
        return the sorted batch targes and sequence lengths.
        This way the output can be used by pack_padded_sequences(...)
        """
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return seq_tensor, target_tensor, seq_lengths

    @classmethod
    def _pad_and_sort_batch(cls, DataLoaderBatch):
        """
        DataLoaderBatch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest, 
        """
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        sequences, targets, lengths = batch_split[0], batch_split[1], batch_split[2]
        
        max_length = max(lengths)

        embedding_dimension = sequences[0].shape[1]

        padded_sequences = np.ones((batch_size, max_length, embedding_dimension))
        padded_targets = np.zeros((batch_size, max_length), dtype=np.int64) - 1

        for i, l in enumerate(lengths):
            padded_sequences[i][0:l][:] = sequences[i][0:l][:]
            padded_targets[i][0:l] = targets[i][0:l]

        
        return cls._sort_batch(torch.from_numpy(padded_sequences), torch.from_numpy(padded_targets), torch.tensor(lengths))
