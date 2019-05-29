import torch
import torchtext
from sklearn import metrics

from typing import List

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sns.set()

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

        print('Loading GloVe vectors of size {} ... Done'.format(glove_size))

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

    def plot_grad_flow(self, named_parameters, gradient_save_path, epoch, mode):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        Taken from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n) and (p.grad is not None):
                if torch.max(p.grad).item() != .0:
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean())
                    max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation=90, fontsize = 5)
        plt.yticks(fontsize = 6)
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers", fontsize = 6)
        plt.ylabel("average gradient", fontsize = 6)
        plt.title("Gradient flow", fontsize = 6)
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.tight_layout()

        plt.savefig("gradients/" + gradient_save_path + mode + "_epoch" + str(epoch) + ".png", dpi = 400)
