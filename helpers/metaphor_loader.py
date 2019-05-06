import os

from torchtext.vocab import Vectors

from datasets.metaphor_dataset import MetaphorDataset

from typing import Tuple


class MetaphorLoader():

    @staticmethod
    def get_metaphor_datasets(
            metaphor_dataset_folder: str,
            word_vector: Vectors) -> Tuple[MetaphorDataset, MetaphorDataset, MetaphorDataset]:
        '''
        Parses the metaphor files and creates MetaphorDataset objects which
        include information about the vocabulary and the embedding of the sentences

        :param str metaphor_dataset_folder: The folder where the metaphor dataset files should be
        :param Vectors word_vector: The vector that will be used to embed the words in the metaphor dataset. It could be GloVe for example
        '''

        assert os.path.isdir(
            metaphor_dataset_folder), 'Metaphor dataset folder is not valid'

        # Train
        train_filepath = os.path.join(
            metaphor_dataset_folder, 'VUA_seq_formatted_train.csv')

        train_dataset = MetaphorDataset(train_filepath, word_vector)

        # Validation
        validation_filepath = os.path.join(
            metaphor_dataset_folder, 'VUA_seq_formatted_val.csv')

        validation_dataset = MetaphorDataset(
            validation_filepath, word_vector)

        # Test
        test_filepath = os.path.join(
            metaphor_dataset_folder, 'VUA_seq_formatted_test.csv')

        test_dataset = MetaphorDataset(test_filepath, word_vector)

        return train_dataset, validation_dataset, test_dataset
