import numpy as np

import torch
import torch.utils.data as data

from typing import Tuple


class DataHelperSnli():

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
				drop_last=True,
				collate_fn=cls._pad_and_sort_batch)

		validation_loader = None
		if validation_dataset:
			validation_loader = data.DataLoader(
				dataset=validation_dataset,
				batch_size=batch_size,
				num_workers=1,
				shuffle=shuffle,
				drop_last=True,
				collate_fn=cls._pad_and_sort_batch)

		test_loader = None
		if test_dataset:
			test_loader = data.DataLoader(
				dataset=test_dataset,
				batch_size=batch_size,
				num_workers=1,
				shuffle=shuffle,
				drop_last=True,
				collate_fn=cls._pad_and_sort_batch)

		return train_loader, validation_loader, test_loader

	@classmethod
	def _sort_batch(cls, batch1, batch2, targets, lengths1, lengths2):
		"""
		Sort a minibatch by the length of the sequences with the longest sequences first
		return the sorted batch targes and sequence lengths.
		This way the output can be used by pack_padded_sequences(...)
		"""
		seq_lengths1, perm_idx1 = lengths1.sort(0, descending=True)
		seq_lengths2, perm_idx2 = lengths2.sort(0, descending=True)

		seq_tensor1 = torch.index_select(batch1, dim = 0, index = perm_idx1)
		seq_tensor2 = torch.index_select(batch2, dim = 0, index = perm_idx2)
		
		recover_idx1 = torch.argsort(perm_idx1, dim = 0, descending = False)
		recover_idx2 = torch.argsort(perm_idx2, dim = 0, descending = False)

		return seq_tensor1, seq_lengths1, recover_idx1, seq_tensor2, seq_lengths2, recover_idx2, targets

	@classmethod
	def _pad_and_sort_batch(cls, DataLoaderBatch):
		"""
		DataLoaderBatch should be a list of (sequence, target, length) tuples...
		Returns a padded tensor of sequences sorted from longest to shortest, 
		"""
		batch_size = len(DataLoaderBatch)
		batch_split = list(zip(*DataLoaderBatch))

		sequences1, sequences2, targets, lengths1, lengths2 = batch_split[0], batch_split[1], batch_split[2], batch_split[3], batch_split[4]
		
		max_length1 = max(lengths1)
		max_length2 = max(lengths2)

		embedding_dimension = sequences1[0].shape[1]

		padded_sequences1 = np.ones((batch_size, max_length1, embedding_dimension))
		padded_sequences2 = np.ones((batch_size, max_length2, embedding_dimension))

		for i, l1, l2 in zip(range(batch_size), lengths1, lengths2):
			padded_sequences1[i][0:l1][:] = sequences1[i]
			padded_sequences2[i][0:l2][:] = sequences2[i]
		
		return cls._sort_batch(torch.from_numpy(padded_sequences1).float(), torch.from_numpy(padded_sequences2).float(),
		 torch.tensor(targets).long(), torch.tensor(lengths1).long(), torch.tensor(lengths2).long())