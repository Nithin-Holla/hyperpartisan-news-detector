import torch
import numpy as np

from typing import List

class HyperpartisanBatch():
    def __init__(self, max_length: int = 15000):
        self._list_of_sequences = []
        self._targets = []
        self._list_of_lengths = []
        self._num_of_sent = []
        self._extra_feat = []
        self._max_length = max_length

    @property
    def list_of_sequences(self):
        return self._list_of_sequences

    @property
    def targets(self):
        return self._targets

    @property
    def list_of_lengths(self):
        return self._list_of_lengths

    @property
    def num_of_sent(self):
        return self._num_of_sent

    @property
    def extra_feat(self):
        return self._extra_feat

    def is_full(self):
        total_sum = sum([sum(lengths) for lengths in self._list_of_lengths])
        return total_sum > self._max_length

    def add_data(self, list_of_sequences, target: int, list_of_lengths, num_of_sent: int, extra_feat):
        self._list_of_sequences.append(list_of_sequences)
        self._targets.append(target)
        self._list_of_lengths.append(list_of_lengths)
        self._num_of_sent.append(num_of_sent)
        self._extra_feat.append(extra_feat)

    def pad_and_sort_batch(self):
        """
        DataLoaderBatch for the Hyperpartisan should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest, 
        """

        ids = []

        # concat_lengths - concatted lengths of each sentence - [ batch_size * n_sentences ]
        concat_lengths = np.array(
            [length for lengths in self._list_of_lengths for length in lengths])
        # concat_sequences - the embeddings for each batch for each sentence for each word - [ (batch_size * n_sentences) x n_words x embedding_dim ]
        concat_sequences = [
            seq[0] for sequences in self._list_of_sequences for seq in sequences]

        max_length = max(concat_lengths)
        
        # print(concat_lengths)
        # print([len(seq) for seq in concat_sequences])

        embedding_dimension = concat_sequences[0].shape[1]

        padded_sequences = np.ones(
            (sum(self._num_of_sent), max_length, embedding_dimension))
        for i, l in enumerate(concat_lengths):
            padded_sequences[i][0:l][:] = concat_sequences[i]

        return self._sort_batch(torch.Tensor(padded_sequences), torch.Tensor(self._targets), torch.LongTensor(self._num_of_sent), torch.LongTensor(concat_lengths), torch.Tensor(self._extra_feat), ids)

    def _sort_batch(self, batch, targets, num_sentences, lengths, extra_feat, ids):
        """
        Sort a minibatch by the length of the sequences with the longest sequences first
        return the sorted batch targes and sequence lengths.
        This way the output can be used by pack_padded_sequences(...)
        """
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]

        _, recover_idx = perm_idx.sort(0)

        # print(seq_tensor.shape, targets.shape, recover_idx.shape, num_sentences.shape, seq_lengths.shape, extra_feat.shape)

        return seq_tensor, targets, recover_idx, num_sentences, seq_lengths, extra_feat, ids
