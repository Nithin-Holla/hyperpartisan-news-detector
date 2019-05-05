import os

from torchtext.vocab import Vectors

from datasets.metaphor_dataset import MetaphorDataset

from typing import Tuple


class MetaphorLoader():

    @staticmethod
    def get_metaphor_datasets(
            metaphor_dataset_folder: str,
            embedding_vector: Vectors,
            embedding_dimension: int = 300) -> Tuple[MetaphorDataset, MetaphorDataset, MetaphorDataset]:
        '''
        Parses the metaphor files and creates MetaphorDataset objects which
        include information about the vocabulary and the embedding of the sentences

        :param str metaphor_dataset_folder: The folder where the metaphor dataset files should be
        :param Vectors embedding_vector: The vector that will be used to embed the words in the metaphor dataset. It could be 'GloVe' for example
        :param int embedding_dimension: the dimension of the embedding vector
        '''

        assert os.path.isdir(
            metaphor_dataset_folder), 'Metaphor dataset folder is not valid'

        # Train
        train_filepath = os.path.join(
            metaphor_dataset_folder, 'VUA_formatted_train.csv')

        train_dataset = MetaphorDataset(
            train_filepath, embedding_vector, embedding_dimension)

        # Validation
        validation_filepath = os.path.join(
            metaphor_dataset_folder, 'VUA_formatted_val.csv')

        validation_dataset = MetaphorDataset(
            validation_filepath, embedding_vector, embedding_dimension)

        # Test
        test_filepath = os.path.join(
            metaphor_dataset_folder, 'VUA_formatted_test.csv')

        test_dataset = MetaphorDataset(
            test_filepath, embedding_vector, embedding_dimension)

        return train_dataset, validation_dataset, test_dataset
