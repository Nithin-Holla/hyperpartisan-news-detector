import os

from torchtext.vocab import Vectors

from datasets.metaphor_dataset import MetaphorDataset

from typing import Tuple


class MetaphorLoader():

    @staticmethod
    def get_metaphor_datasets(
            metaphor_dataset_folder: str,
            glove_vectors: Vectors,
            lowercase_sentences: bool = False,
            tokenize_sentences: bool = True) -> Tuple[MetaphorDataset, MetaphorDataset, MetaphorDataset]:
        '''
        Parses the metaphor files and creates MetaphorDataset objects which
        include information about the vocabulary and the embedding of the sentences

        :param str metaphor_dataset_folder: The folder where the metaphor dataset files should be
        :param Vectors glove_vectors: The vector that will be used to embed the words in the metaphor dataset. It could be GloVe for example
        :param bool lowercase_sentences: Specify whether the sentences should be lowercased before embedding them
        :param bool tokenize_sentences: Specify whether the sentence words should be tokenized before embedding them
        '''

        assert os.path.isdir(
            metaphor_dataset_folder), 'Metaphor dataset folder is not valid'

        # Train
        train_filepath = os.path.join(
            metaphor_dataset_folder, 'VUA_seq_formatted_train.csv')

        train_dataset = MetaphorDataset(
            filename=train_filepath,
            glove_vectors=glove_vectors,
            lowercase_sentences=lowercase_sentences,
            tokenize_sentences=tokenize_sentences)

        # Validation
        validation_filepath = os.path.join(
            metaphor_dataset_folder, 'VUA_seq_formatted_val.csv')

        validation_dataset = MetaphorDataset(
            filename=validation_filepath,
            glove_vectors=glove_vectors,
            lowercase_sentences=lowercase_sentences,
            tokenize_sentences=tokenize_sentences)

        # Test
        test_filepath = os.path.join(
            metaphor_dataset_folder, 'VUA_seq_formatted_test.csv')

        test_dataset = MetaphorDataset(
            filename=test_filepath,
            glove_vectors=glove_vectors,
            lowercase_sentences=lowercase_sentences,
            tokenize_sentences=tokenize_sentences)

        return train_dataset, validation_dataset, test_dataset
