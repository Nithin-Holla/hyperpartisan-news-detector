import os

from torchtext.vocab import Vectors

from datasets.hyperpartisan_dataset import HyperpartisanDataset

from typing import Tuple


class HyperpartisanLoader():

    @staticmethod
    def get_hyperpartisan_datasets(
            hyperpartisan_dataset_folder: str,
            glove_vectors: Vectors,
            lowercase_sentences: bool = False,
            articles_max_length: int = None) -> Tuple[HyperpartisanDataset, HyperpartisanDataset]:
        '''
        Parses the hyperpartisan files and creates HyperpartisanDataset objects which
        include information about the vocabulary and the embedding of the sentences

        :param str hyperpartisan_dataset_folder: The folder where the hyperpartisan dataset files should be
        :param Vectors glove_vectors: The vector that will be used to embed the words in the hyperpartisan dataset. It could be GloVe for example
        :param bool lowercase_sentences: Specify whether the sentences should be lowercased before embedding them
        '''

        assert os.path.isdir(
            hyperpartisan_dataset_folder), 'Hyperpartisan dataset folder is not valid'

        # Train
        train_filepath = os.path.join(
            hyperpartisan_dataset_folder, 'train_byart.txt')

        train_dataset = HyperpartisanDataset(
            filename=train_filepath,
            glove_vectors=glove_vectors,
            lowercase_sentences=lowercase_sentences,
            articles_max_length=articles_max_length)

        # Validation
        validation_filepath = os.path.join(
            hyperpartisan_dataset_folder, 'valid_byart.txt')

        validation_dataset = HyperpartisanDataset(
            filename=validation_filepath,
            glove_vectors=glove_vectors,
            lowercase_sentences=lowercase_sentences,
            articles_max_length=articles_max_length)

        # Test
        # test_filepath = os.path.join(hyperpartisan_dataset_folder, 'test_byart.txt')
		
        # test_dataset = HyperpartisanDataset(
        # 	filename=test_filepath,
        # 	glove_vectors=glove_vectors,
        # 	lowercase_sentences=lowercase_sentences,
        #   articles_max_length=articles_max_length)

        return train_dataset, validation_dataset
