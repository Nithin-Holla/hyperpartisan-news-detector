import os
from torchtext.vocab import Vectors
from datasets.snli_dataset import SnliDataset
from typing import Tuple

class SnliLoader():

    @staticmethod
    def get_snli_datasets(snli_dataset_folder: str, glove_vectors: Vectors) -> Tuple[SnliDataset, SnliDataset, SnliDataset]:
        '''
        Parses the metaphor files and creates MetaphorDataset objects which
        include information about the vocabulary and the embedding of the sentences

        :param str metaphor_dataset_folder: The folder where the metaphor dataset files should be
        :param Vectors glove_vectors: The vector that will be used to embed the words in the metaphor dataset. It could be GloVe for example
        :param bool lowercase_sentences: Specify whether the sentences should be lowercased before embedding them
        :param bool tokenize_sentences: Specify whether the sentence words should be tokenized before embedding them
        :param bool only_news: Use only metaphors of genre 'news' when loading data
        '''

        assert os.path.isdir(snli_dataset_folder), 'Snli dataset folder is not valid'

        # Train
        train_filepath = os.path.join(snli_dataset_folder, 'train.csv')
        train_dataset = MetaphorDataset(filename=train_filepath, glove_vectors=glove_vectors)

        # Validation
        validation_filepath = os.path.join(snli_dataset_folder, 'val.csv')
        validation_dataset = MetaphorDataset(filename=validation_filepath, glove_vectors=glove_vectors)

        # Test
        test_filepath = os.path.join(snli_dataset_folder, 'test.csv')
        test_dataset = MetaphorDataset(filename=test_filepath, glove_vectors=glove_vectors)

        return train_dataset, validation_dataset, test_dataset