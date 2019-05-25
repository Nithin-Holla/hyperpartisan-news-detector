import torch
import torchtext
from sklearn import metrics

from typing import List


class UtilsHelper():
    def initialize_deterministic_mode(self, deterministic_seed):
        if not deterministic_seed:
            return

        print(
            f'Initializing deterministic mode with seed {deterministic_seed}')

        torch.manual_seed(deterministic_seed)
        torch.cuda.manual_seed(deterministic_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_glove_vectors(self, vector_file_name, vector_cache_dir, glove_size):

        print('Loading GloVe vectors...\r', end='')

        if glove_size:
            glove_size = int(glove_size)

        glove_vectors = torchtext.vocab.Vectors(name=vector_file_name,
                                                cache=vector_cache_dir,
                                                max_vectors=glove_size)
        glove_vectors.stoi = {k: v+2 for (k, v) in glove_vectors.stoi.items()}
        glove_vectors.itos = ['<unk>', '<pad>'] + glove_vectors.itos
        glove_vectors.stoi['<unk>'] = 0
        glove_vectors.stoi['<pad>'] = 1
        unk_vector = torch.zeros((1, glove_vectors.dim))
        pad_vector = torch.mean(glove_vectors.vectors, dim=0, keepdim=True)
        glove_vectors.vectors = torch.cat(
            (unk_vector, pad_vector, glove_vectors.vectors), dim=0)

        print('Loading GloVe vectors...Done')

        return glove_vectors

    def calculate_accuracy(self, prediction_scores, targets):
        """
        Calculate the accuracy
        :param prediction_scores: Scores obtained by the model
        :param targets: Ground truth targets
        :return: Accuracy
        """
        binary = len(prediction_scores.shape) == 1

        if binary:
            prediction = prediction_scores > 0.5
            accuracy = torch.sum(prediction == targets.byte()
                                 ).float() / prediction.shape[0]
        else:
            prediction = torch.argmax(prediction_scores, dim=1)
            accuracy = torch.mean((prediction == targets).float())

        return accuracy

    def calculate_metrics(
            self,
            targets: List,
            predictions: List,
            average: str = "binary"):

        if sum(predictions) == 0:
            return 0, 0, 0

        precision = metrics.precision_score(
            targets, predictions, average=average)
        recall = metrics.recall_score(targets, predictions, average=average)
        f1 = metrics.f1_score(targets, predictions, average=average)

        return f1, precision, recall
