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


class MetaphorDataset(data.Dataset):
    def __init__(self, filename: str, word_vector: Vectors):
        assert os.path.splitext(
            filename)[1] == '.csv', 'Metaphor dataset file should be of type CSV'

        self._sentences, self._labels, self._word_vectors = self._parse_csv_file(
            filename, word_vector)

        self._data_size = len(self._sentences)
        self.word_vector = word_vector

    def __getitem__(self, idx):
        sentence = self._sentences[idx]

        words = sentence.split()
        indexed_sequence = [self.word_vector.stoi.get(x, -1) for x in words]
        targets = self._labels[idx]

        sentence_length = len(words)
        return indexed_sequence, targets, sentence_length

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
            for row in csv_reader:
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
