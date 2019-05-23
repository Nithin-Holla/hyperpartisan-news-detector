import os
from torchtext.vocab import Vectors
from datasets.snli_dataset import SnliDataset
from typing import Tuple

class SnliLoader():

    @staticmethod
    def get_snli_datasets(snli_dataset_folder, glove_vectors, args) -> Tuple[SnliDataset, SnliDataset, SnliDataset]:

        assert os.path.isdir(snli_dataset_folder), 'Snli dataset folder is not valid'

        # Train
        train_filepath = os.path.join(snli_dataset_folder, 'train.csv')
        train_dataset = SnliDataset(filename=train_filepath, glove_vectors=glove_vectors, use_data=args.use_data_train)

        # Validation
        validation_filepath = os.path.join(snli_dataset_folder, 'val.csv')
        validation_dataset = SnliDataset(filename=validation_filepath, glove_vectors=glove_vectors, use_data=args.use_data_valid)

        # Test
        test_filepath = os.path.join(snli_dataset_folder, 'test.csv')
        test_dataset = SnliDataset(filename=test_filepath, glove_vectors=glove_vectors, use_data=args.use_data_test)

        return train_dataset, validation_dataset, test_dataset