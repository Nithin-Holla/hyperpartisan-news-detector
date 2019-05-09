import torchtext
import os
import numpy as np

from torchtext.vocab import Vectors

from torch.autograd import Variable
import torch.utils.data as data
import torch
import torch.nn as nn

import csv
import ast
from typing import List, Tuple, Dict, Set
from nltk.tokenize import WhitespaceTokenizer
import h5py


class MetaphorDataset(data.Dataset):
    def __init__(
            self,
            filename: str,
            word_vector: Vectors,
            elmo_vectors: int,
            elmo_dimensions: int = 1024):
        assert os.path.splitext(
            filename)[1] == '.csv', 'Metaphor dataset file should be of type CSV'

        self._sentences, self._labels, self._word_vectors = self._parse_csv_file(
            filename, word_vector)

        self.elmo_filename = self._assert_elmo_vectors_file(
            filename, self._sentences)

        self._data_size = len(self._sentences)
        self.word_vector = word_vector
        self.tokenizer = WhitespaceTokenizer()
        self.elmo_dimensions = elmo_dimensions
        self.elmo_vectors = elmo_vectors

    def __getitem__(self, idx):
        sentence = self._sentences[idx]

        words = self.tokenizer.tokenize(sentence.lower())
        sentence_length = len(words)

        indexed_sequence = torch.stack([self.word_vector[x] for x in words])
        elmo_embedding_file = h5py.File(self.elmo_filename, 'r')
        sentence_elmo_embeddings = elmo_embedding_file[str(idx)]
        elmo_embeddings = torch.cat([torch.Tensor(sentence_elmo_embeddings[i]) for i in range(self.elmo_vectors)], dim=1)

        # elmo: [ n_words x (1024*3) ]; [ n_words x 300 ] => [ n_words x 1324 ]
        assert list(elmo_embeddings.size()) == [sentence_length, self.elmo_dimensions * self.elmo_vectors]
        assert list(indexed_sequence.size()) == [sentence_length, self.word_vector.dim]

        combined_embeddings = torch.cat(
            [elmo_embeddings, indexed_sequence], dim=1)

        elmo_embedding_file.close()

        targets = self._labels[idx]

        assert sentence_length == len(
            targets), 'Length of sentence tokens is not the same as the length of the targets'

        return combined_embeddings, targets, sentence_length

    def __len__(self):
        return self._data_size

    def _parse_csv_file(self, filename: str, word_vector: Vectors) \
            -> Tuple[List[str], List[bool], Dict[str, torch.Tensor]]:
        '''
        Parses the metaphor CSV file and creates the necessary objects for the dataset

        :param str filename: the path to the metaphor CSV dataset file
        :param Vectors word_vector: the vector which will be used for word representation
        :return: list of all sentences, list of their labels, dictionary of all the word representations
        '''
        sentences = []
        labels = []
        word_vectors = {}

        with open(filename, 'r') as csv_file:
            next(csv_file)  # skip the first line - headers
            csv_reader = csv.reader(csv_file, delimiter=',')
            for counter, row in enumerate(csv_reader):
                sentence = row[2]
                sentence_labels_string = ast.literal_eval(row[3])
                sentence_labels = [int(n) for n in sentence_labels_string]
                sentences.append(sentence)
                labels.append(sentence_labels)

                words = sentence.split()
                for word in words:
                    if word not in word_vectors.keys():
                        word_vectors[word] = word_vector[word]

        return sentences, labels, word_vectors

    def _assert_elmo_vectors_file(self, csv_filename, sentences):
        dirname = os.path.dirname(csv_filename)
        filename_without_ext = os.path.splitext(
            os.path.basename(csv_filename))[0]
        elmo_filename = os.path.join(dirname, f'{filename_without_ext}.hdf5')
        if not os.path.isfile(elmo_filename):
            print("caching elmo vectors")
            sentences_filename = os.path.join(
                dirname, f'{filename_without_ext}.txt')
            with open(sentences_filename, 'w', encoding='utf-8') as sentences_file:
                for sentence in sentences:
                    sentences_file.write(f'{sentence}\n')

            raise Exception(
                'Please save the sentences file to elmo file using \'allennlp elmo\' command')

        return elmo_filename
