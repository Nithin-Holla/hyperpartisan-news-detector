import os

from torchtext.vocab import Vectors

from datasets.hyperpartisan_dataset import HyperpartisanDataset

from typing import Tuple


class HyperpartisanLoader():

	@staticmethod
	def get_hyperpartisan_datasets(
			hyperpartisan_dataset_folder: str,
			word_vector: Vectors) -> Tuple[HyperpartisanDataset, HyperpartisanDataset, HyperpartisanDataset]:
		'''
		Parses the hyperpartisan files and creates HyperpartisanDataset objects which
		include information about the vocabulary and the embedding of the sentences

		:param str hyperpartisan_dataset_folder: The folder where the hyperpartisan dataset files should be
		:param Vectors word_vector: The vector that will be used to embed the words in the hyperpartisan dataset. It could be GloVe for example
		'''

		assert os.path.isdir(hyperpartisan_dataset_folder), 'Hyperpartisan dataset folder is not valid'

		# Train
		train_filepath = os.path.join(hyperpartisan_dataset_folder, 'train_byart.txt')
		train_dataset = HyperpartisanDataset(train_filepath, word_vector)

		# Validation
		validation_filepath = os.path.join(hyperpartisan_dataset_folder, 'valid_byart.txt')
		validation_dataset = HyperpartisanDataset(validation_filepath, word_vector)

		# Test
		test_filepath = os.path.join(hyperpartisan_dataset_folder, 'test_byart.txt')
		test_dataset = HyperpartisanDataset(test_filepath, word_vector)

		return train_dataset, validation_dataset, test_dataset