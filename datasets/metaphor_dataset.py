import torchtext
import os
import numpy as np

from torchtext.vocab import Vectors

from torch.autograd import Variable
import torch.utils.data as data
import torch
import torch.nn as nn

import csv
from typing import List, Tuple, Dict, Set


class MetaphorDataset(data.Dataset):
    _pad_token = '<PAD>'
    _unk_token = '<UNK>'

    def __init__(self, filename: str, embedding_vector: Vectors, embedding_dimension: int = 300):
        assert os.path.splitext(
            filename)[1] == '.csv', 'Metaphor dataset file should be of type CSV'

        self._sentences, self._labels, self._word_vectors, _vocabulary = self._parse_csv_file(
            filename, embedding_vector)

        self._word2id = {
            self._pad_token: 0,
            self._unk_token: 1
        }

        self._id2word = {
            0: self._pad_token,
            1: self._unk_token
        }

        for word in _vocabulary:
            assigned_index = len(self._word2id)
            self._word2id[word] = assigned_index
            self._id2word[assigned_index] = word

        self._data_size = len(self._sentences)
        self._token_size = len(self._word2id)
        self._embedding = self._create_embedding(embedding_dimension)

    def __getitem__(self, idx):
        sentence = self._sentences[idx]
        embedded_sentence = self._embed_sentence(sentence)
        target = torch.Tensor([self._labels[idx]])
        sentence_length = embedded_sentence.shape[0]
        return embedded_sentence, target, sentence_length

    def __len__(self):
        return self._data_size

    def _parse_csv_file(self, filename: str, embedding_vector: Vectors) \
            -> Tuple[List[str], List[bool], Dict[str, torch.Tensor], Set[str]]:
        '''
        Parses the metaphor CSV file and creates the necessary objects for the dataset

        :param str filename: the path to the metaphor CSV dataset file
        :param Vectors embedding_vector: the vector which will be used for word representation
        :return: list of all sentences, list of their labels, dictionary of all the word representations and vocabulary with all the unique words
        '''
        sentences = []
        labels = []
        word_vectors = {}

        with open(filename, 'r') as csv_file:
            next(csv_file)  # skip the first line - headers
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                sentence = row[3]
                is_metaphor = row[5] == '1'
                sentences.append(sentence)
                labels.append(is_metaphor)

                words = sentence.split()
                for word in words:
                    if word not in word_vectors.keys():
                        word_vectors[word] = embedding_vector[word]

        vocabulary = set(words)
        return sentences, labels, word_vectors, vocabulary

    def _create_embedding(self, embedding_dimension: int) -> nn.Embedding:
        '''
        Creates an embedding layer including all word vectors 

        :param str embedding_dimension: the dimensions of the original embedding vector
        :return: the created embedding layer
        '''
        all_embeddings = torch.stack(list(self._word_vectors.values()))
        embeddings_mean = float(all_embeddings.mean())
        embeddings_std = float(all_embeddings.std())

        # Initialize an embedding matrix of (vocab_size, embedding_dimension) shape with
        # random numbers and with a similar distribution as the pretrained embeddings for words in vocab.
        embedding_matrix = torch.FloatTensor(
            self._token_size, embedding_dimension).normal_(embeddings_mean, embeddings_std)

        # Go through the embedding matrix and replace the random vector with a
        # pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
        for i in range(2, self._token_size):
            word = self._id2word[i]
            if word in self._word_vectors:
                embedding_matrix[i] = torch.FloatTensor(
                    self._word_vectors[word])

        embeddings = nn.Embedding(
            self._token_size, embedding_dimension, padding_idx=0)

        embeddings.weight = nn.Parameter(embedding_matrix, requires_grad=False)
        return embeddings

    def _embed_sentence(self, sentence: str) -> torch.Tensor:
        '''
        Embeds the given sentence using the initial embedding layer and 

        :param str sentence: the sentence to be embedded
        :return: a tensor representation of the sentence
        '''
        words = sentence.split()
        indexed_sequence = [self._word2id.get(x, 1) for x in words]

        result = self._embedding(Variable(torch.LongTensor(indexed_sequence)))
        return result
