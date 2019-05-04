import torch
from torch import nn, optim
import torchtext
import argparse
import os
from model.HierarchicalAttentionNetwork import HierarchicalAttentionNetwork


def train_model(config):
    """
    Train the multi-task classifier model
    :param config: Dictionary specifying the model configuration
    :return:
    """
    # Flags for deterministic runs
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load GloVe vectors
    glove_vectors = torchtext.vocab.Vectors(name=config.vector_file_name,
                                            cache=config.vector_cache_dir,
                                            max_vectors=config.glove_size)
    vocab_size = len(glove_vectors.vectors)

    # Define the model, the optimizer and the loss module
    model = HierarchicalAttentionNetwork(vocab_size=vocab_size,
                                         embedding_dim=glove_vectors.dim,
                                         hidden_dim=config.hidden_dim,
                                         num_classes=2,
                                         pretrained_vectors=glove_vectors.vectors).to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=config.learning_rate, weight_decay=config.weight_decay)
    cross_entropy_loss = nn.CrossEntropyLoss()

    # Load the checkpoint if found
    if os.path.isfile(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Resuming training from epoch %d with loaded model and optimizer..." % start_epoch)
    else:
        start_epoch = 1
        print("Training the model from scratch...")

    # Initialize dataloaders
    hyperpartisan_dataloader = None
    metaphor_dataloader = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to save/load the checkpoint data')
    parser.add_argument('--data_path', type=str,
                        help='Path where data is saved')
    parser.add_argument('--vector_file_name', type=str, required=True,
                        help='File in which vectors are saved')
    parser.add_argument('--vector_cache_dir', type=str, default='.vector_cache',
                        help='Directory where vectors would be cached')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='Maximum number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training the model')
    parser.add_argument('--hidden_dim', type=int, default=50,
                        help='Hidden dimension of the recurrent network')
    parser.add_argument('--glove_size', type=int,
                        help='Number of GloVe vectors to load initially')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for the optimizer')
    config = parser.parse_args()

    train_model(config)
