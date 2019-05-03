import torch
import argparse
from model.HierarchicalAttentionNetwork import HierarchicalAttentionNetwork


def train_model():
    model = HierarchicalAttentionNetwork(vocab_size, embedding_dim, hidden_dim, num_classes, pretrained_vectors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add arguments here
    config = parser.parse_args()
