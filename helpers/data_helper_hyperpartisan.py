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

        # test_loader = None
        # if test_dataset:
        #     test_loader = data.DataLoader(
        #         dataset=test_dataset,
        #         batch_size=batch_size,
        #         num_workers=1,
        #         shuffle=shuffle,
        #         collate_fn=cls._pad_and_sort_batch)

        return train_loader, validation_loader

    @classmethod
    def _sort_batch(cls, batch, targets, num_sentences, lengths):
        """
        Sort a minibatch by the length of the sequences with the longest sequences first
        return the sorted batch targes and sequence lengths.
        This way the output can be used by pack_padded_sequences(...)
        """
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]

        _, recover_idx = perm_idx.sort(0)

        return seq_tensor, targets, recover_idx, num_sentences, seq_lengths

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
        concat_sequences = [seq for sequences in list_of_sequences for seq in sequences]
        max_length = max(concat_lengths)

        embedding_dimension = concat_sequences[0].shape[1]

        padded_sequences = np.ones((sum(num_of_sent) + batch_size, max_length, embedding_dimension))
        for i, l in enumerate(concat_lengths):
            padded_sequences[i][0:l][:] = concat_sequences[i][0:l][:]

        return cls._sort_batch(torch.Tensor(padded_sequences), torch.Tensor(targets), torch.LongTensor(num_of_sent), torch.LongTensor(concat_lengths))