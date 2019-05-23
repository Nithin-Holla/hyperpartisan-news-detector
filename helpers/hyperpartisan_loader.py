import os

from torchtext.vocab import Vectors

from datasets.hyperpartisan_dataset import HyperpartisanDataset

from typing import Tuple

from enums.elmo_model import ELMoModel


class HyperpartisanLoader():

    @staticmethod
    def get_hyperpartisan_datasets(
            hyperpartisan_dataset_folder: str,
            concat_glove: bool,
            glove_vectors: Vectors,
            elmo_model: ELMoModel,
            lowercase_sentences: bool = False,
            articles_max_length: int = None,
            load_train: bool = True) -> Tuple[HyperpartisanDataset, HyperpartisanDataset]:
        '''
        Parses the hyperpartisan files and creates HyperpartisanDataset objects which
        include information about the vocabulary and the embedding of the sentences

        :param str hyperpartisan_dataset_folder: The folder where the hyperpartisan dataset files should be
        :param bool concat_glove: Whether GloVe vectors have to be concatenated with ELMo vectors for words
        :param Vectors glove_vectors: The vector that will be used to embed the words in the hyperpartisan dataset. It could be GloVe for example
        :param ELMoModel elmo_model: The ELMo from which vectors are used
        :param bool lowercase_sentences: Specify whether the sentences should be lowercased before embedding them
        '''

        assert os.path.isdir(
            hyperpartisan_dataset_folder), 'Hyperpartisan dataset folder is not valid'

        # Train
        if load_train:
            train_filepath = os.path.join(
                hyperpartisan_dataset_folder, 'train_byart.txt')

            train_dataset = HyperpartisanDataset(
                filename=train_filepath,
                concat_glove=concat_glove,
                glove_vectors=glove_vectors,
                elmo_model=elmo_model,
                lowercase_sentences=lowercase_sentences,
                articles_max_length=articles_max_length)
        else:
            train_dataset = None

        # Validation
        validation_filepath = os.path.join(
            hyperpartisan_dataset_folder, 'valid_byart.txt')

        validation_dataset = HyperpartisanDataset(
            filename=validation_filepath,
            concat_glove=concat_glove,
            glove_vectors=glove_vectors,
            elmo_model=elmo_model,
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
