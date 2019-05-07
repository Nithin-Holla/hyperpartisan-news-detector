import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torchtext
import argparse
import os

from datasets.hyperpartisan_dataset import HyperpartisanDataset
from helpers.hyperpartisan_loader import HyperpartisanLoader
from datasets.metaphor_dataset import MetaphorDataset
from helpers.metaphor_loader import MetaphorLoader
from helpers.data_helper import DataHelper
from model.JointModel import JointModel


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
    model = JointModel(vocab_size=vocab_size,
                       embedding_dim=glove_vectors.dim,
                       hidden_dim=config.hidden_dim,
                       hyp_n_classes=2,
                       pretrained_vectors=glove_vectors.vectors).to(device)
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=config.learning_rate, weight_decay=config.weight_decay)
    metaphor_criterion = nn.BCELoss()

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

    # Load hyperpartisan data
    hyperpartisan_train_dataset, hyperpartisan_validation_dataset, hyperpartisan_test_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
        hyperpartisan_dataset_folder=config.hyperpartisan_dataset_folder,
        word_vector=glove_vectors)

    hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader, hyperpartisan_test_dataloader = DataHelper.create_dataloaders(
        train_dataset=hyperpartisan_train_dataset,
        validation_dataset=hyperpartisan_validation_dataset,
        test_dataset=hyperpartisan_test_dataset,
        batch_size=config.batch_size,
        shuffle=True)

    # Load metaphor data
    metaphor_train_dataset, metaphor_validation_dataset, metaphor_test_dataset = MetaphorLoader.get_metaphor_datasets(
        metaphor_dataset_folder=config.metaphor_dataset_folder,
        word_vector=glove_vectors)

    metaphor_train_dataloader, metaphor_validation_dataloader, metaphor_test_dataloader = DataHelper.create_dataloaders(
        train_dataset=metaphor_train_dataset,
        validation_dataset=metaphor_validation_dataset,
        test_dataset=metaphor_test_dataset,
        batch_size=config.batch_size,
        shuffle=True)

    for epoch in range(start_epoch, config.max_epochs + 1):
        print("Epoch %d" % epoch)
        for step, (m_batch_inputs, m_batch_targets, m_batch_lengths) in enumerate(metaphor_train_dataloader):
            m_batch_inputs = m_batch_inputs.to(device)
            m_batch_targets = m_batch_targets.to(device).view(-1).float()
            m_batch_lengths = m_batch_lengths.to(device)
            optimizer.zero_grad()
            pred = model(m_batch_inputs, m_batch_lengths, task='metaphor')
            unpad_targets = m_batch_targets[m_batch_targets != -1]
            unpad_pred = pred.view(-1)[m_batch_targets != -1]
            loss = metaphor_criterion(unpad_pred, unpad_targets)
            loss.backward()
            optimizer.step()
            print("Loss = ", loss.item())


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
    parser.add_argument('--embedding_dimension', type=int, default=300,
                        help='Dimensions of the vector embeddings')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
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
    parser.add_argument('--metaphor_dataset_folder', type=str,
                        help='Path to the metaphor dataset')
    parser.add_argument('--hyperpartisan_dataset_folder', type=str,
                        help='Path to the hyperpartisan dataset')

    config = parser.parse_args()

    train_model(config)
