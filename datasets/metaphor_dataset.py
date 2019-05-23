import os
import numpy as np

from torchtext.vocab import Vectors

import torch.utils.data as data
import torch

import csv
import ast
from typing import List, Tuple, Dict
from nltk.tokenize import WhitespaceTokenizer
import h5py

from enums.elmo_model import ELMoModel


class MetaphorDataset(data.Dataset):
    def __init__(
            self,
            filename: str,
            glove_vectors: Vectors,
            elmo_model: ELMoModel,
            lowercase_sentences: bool = False,
            tokenize_sentences: bool = True,
            only_news: bool = False):

        assert os.path.splitext(
            filename)[1] == '.csv', 'Metaphor dataset file should be of type CSV'

        self.glove_vectors = glove_vectors
        self.tokenizer = WhitespaceTokenizer()
        self.lowercase_sentences = lowercase_sentences
        self.tokenize_sentences = tokenize_sentences
        self.only_news = only_news
        self.elmo_model = elmo_model

        self._sentences, self._labels = self._parse_csv_file(filename)

        self.elmo_filename = self._assert_elmo_vectors_file(
            filename, self._sentences)

        self._data_size = len(self._sentences)

    def __getitem__(self, idx):
        sentence = self._sentences[idx]

        if self.lowercase_sentences:
            sentence = sentence.lower()

        if self.tokenize_sentences:
            words = self.tokenizer.tokenize(sentence)
        else:
            words = sentence.split()

        sentence_length = len(words)

        # get the GloVe embedding. If it's missing for this word - try lowercasing it
        words_embeddings = [
            self.glove_vectors[x] if x in self.glove_vectors.stoi else self.glove_vectors[x.lower()] for x in words]

        glove_embeddings = torch.stack(words_embeddings)
        elmo_embedding_file = h5py.File(self.elmo_filename, 'r')
        elmo_embeddings = torch.Tensor(elmo_embedding_file[str(idx)])

        # elmo: [ n_words x (1024) ]; glove: [ n_words x 300 ] => combined: [ n_words x 1324 ]
        combined_embeddings = torch.cat(
            [elmo_embeddings, glove_embeddings], dim=1)

        elmo_embedding_file.close()

        targets = self._labels[idx]

        assert sentence_length == len(
            targets), 'Length of sentence tokens is not the same as the length of the targets'

        return combined_embeddings, targets, sentence_length

    def __len__(self):
        return self._data_size

    def _parse_csv_file(self, filename: str) \
            -> Tuple[List[str], List[bool], Dict[str, torch.Tensor]]:
        '''
        Parses the metaphor CSV file and creates the necessary objects for the dataset

        :param str filename: the path to the metaphor CSV dataset file
        :return: list of all sentences, list of their labels, dictionary of all the word representations
        '''
        
        sentences = []
        labels = []

        with open(filename, 'r') as csv_file:
            next(csv_file)  # skip the first line - headers
            csv_reader = csv.reader(csv_file, delimiter=',')
            for _, row in enumerate(csv_reader):
                if self.only_news:
                    genre = row[6]
                    if genre != 'news':
                        continue

                sentence = row[2]
                sentence_labels_string = ast.literal_eval(row[3])
                sentence_labels = [int(n) for n in sentence_labels_string]
                sentences.append(sentence)
                labels.append(sentence_labels)

        return sentences, labels

    def _assert_elmo_vectors_file(self, csv_filename, sentences):
        '''
        Saves the elmo sentences to a text file which will be used to create elmo embeddings after
        '''

        dirname = os.path.dirname(csv_filename)
        filename_without_ext = os.path.splitext(
            os.path.basename(csv_filename))[0]

        file_suffix = self._create_elmo_file_suffix()
        elmo_filename = os.path.join(
            dirname, f'{filename_without_ext}{file_suffix}.hdf5')
        if not os.path.isfile(elmo_filename):
            print("Saving sentences...")
            sentences_filename = os.path.join(
                dirname, f'{filename_without_ext}{file_suffix}.txt')

            with open(sentences_filename, 'w', encoding='utf-8') as sentences_file:
                for sentence in sentences:
                    if self.lowercase_sentences:
                        sentence = sentence.lower()

                    if self.tokenize_sentences:
                        sentence = ' '.join(self.tokenizer.tokenize(sentence))

                    sentences_file.write(f'{sentence}\n')

            raise Exception(
                f'Please save the sentences file to the file {filename_without_ext}{file_suffix}.hdf5 using \'allennlp elmo\' command')

        return elmo_filename

    def _create_elmo_file_suffix(self):
        '''
        Creates a file suffix which includes all current configuration options
        '''

        if self.elmo_model == ELMoModel.Original:
            file_suffix = '_elmo'
        elif self.elmo_model == ELMoModel.Small:
            file_suffix = '_elmo_small'

        if self.only_news:
            file_suffix = f'_only_news{file_suffix}'
        
        if self.lowercase_sentences:
            file_suffix = f'_lowercase{file_suffix}'

        if self.tokenize_sentences:
            file_suffix = f'_tokenize{file_suffix}'

        return file_suffix
