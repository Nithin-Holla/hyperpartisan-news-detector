import os

from torchtext.vocab import Vectors

from datasets.hyperpartisan_dataset import HyperpartisanDataset

from typing import Tuple


class MetaphorLoader():

	@staticmethod
	def get_hyperpartisan_datasets(
			hyperpartisan_dataset_folder: str,
			word_vector: Vectors,
			use_data: list) -> Tuple[HyperpartisanDataset, HyperpartisanDataset, HyperpartisanDataset]:
		'''
		Parses the metaphor files and creates MetaphorDataset objects which
		include information about the vocabulary and the embedding of the sentences

		:param str metaphor_dataset_folder: The folder where the metaphor dataset files should be
		:param Vectors word_vector: The vector that will be used to embed the words in the metaphor dataset. It could be GloVe for example
		'''

		assert os.path.isdir(metaphor_dataset_folder), 'Hyperpartisan dataset folder is not valid'

		# Train
		train_filepath = os.path.join(hyperpartisan_dataset_folder, 'train.csv')
		train_dataset = HyperpartisanDataset(train_filepath, word_vector, use_data = use_data[0])

		# Validation
		validation_filepath = os.path.join(hyperpartisan_dataset_folder, 'valid.csv')
		validation_dataset = HyperpartisanMetaphorDataset(validation_filepath, word_vector, use_data = use_data[1])

		# Test
		test_filepath = os.path.join(hyperpartisan_dataset_folder, 'test.csv')
		test_dataset = HyperpartisanDataset(test_filepath, word_vector, use_data = use_data[2])

		return train_dataset, validation_dataset, test_dataset