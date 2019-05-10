import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torchtext
import argparse
import os
import numpy as np

from datasets.hyperpartisan_dataset import HyperpartisanDataset
from helpers.hyperpartisan_loader import HyperpartisanLoader
from datasets.metaphor_dataset import MetaphorDataset
from helpers.metaphor_loader import MetaphorLoader
from helpers.data_helper import DataHelper
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan
from model.JointModel import JointModel

from enums.training_mode import TrainingMode

from sklearn import metrics
from datetime import datetime
import time

elmo_vectors_size = 1024

def initialize_deterministic_mode():
    print('Initializing deterministic mode')
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_accuracy(prediction_scores, targets):
    """
    Calculate the accuracy
    :param prediction_scores: Scores obtained by the model
    :param targets: Ground truth targets
    :return: Accuracy
    """
    binary = len(prediction_scores.shape) == 1

    if binary:
        prediction = prediction_scores > 0.5
        accuracy = torch.sum(prediction == targets.byte()).float() / prediction.shape[0]
    else:
        prediction = torch.argmax(prediction_scores, dim=1)
        accuracy = torch.mean((prediction == targets).float())

    return accuracy


def load_glove_vectors(config):
    print('Loading GloVe vectors...\r', end='')

    glove_vectors = torchtext.vocab.Vectors(name=config.vector_file_name,
                                            cache=config.vector_cache_dir,
                                            max_vectors=config.glove_size)
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

def initialize_model(config, device, glove_vectors_dim):
    print('Loading model state...\r', end='')

    total_embedding_dim = elmo_vectors_size + glove_vectors_dim

    joint_model = JointModel(embedding_dim=total_embedding_dim, hidden_dim=config.hidden_dim, device=device).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, joint_model.parameters()),
                           lr=config.learning_rate, weight_decay=config.weight_decay)

    # Load the checkpoint if found
    if os.path.isfile(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path)
        joint_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print('Found previous model state')
    else:
        start_epoch = 1
        print('Loading model state...Done')

    print("Starting training in '%s' mode from epoch %d..." % (config.mode, start_epoch))

    return joint_model, optimizer, start_epoch

def create_hyperpartisan_loaders(config, glove_vectors):
    hyperpartisan_train_dataset, hyperpartisan_validation_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
        hyperpartisan_dataset_folder=config.hyperpartisan_dataset_folder,
        glove_vectors=glove_vectors,
        lowercase_sentences=config.lowercase)

    hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader = DataHelperHyperpartisan.create_dataloaders(
        train_dataset=hyperpartisan_train_dataset,
        validation_dataset=hyperpartisan_validation_dataset,
        test_dataset=None,
        batch_size=config.batch_size,
        shuffle=True)

    return hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader

def create_metaphor_loaders(config, glove_vectors):
    metaphor_train_dataset, metaphor_validation_dataset, metaphor_test_dataset = MetaphorLoader.get_metaphor_datasets(
        metaphor_dataset_folder=config.metaphor_dataset_folder,
        glove_vectors=glove_vectors,
        lowercase_sentences=config.lowercase,
        tokenize_sentences=not config.not_tokenize,
        only_news=config.only_news)

    metaphor_train_dataloader, metaphor_validation_dataloader, _ = DataHelper.create_dataloaders(
        train_dataset=metaphor_train_dataset,
        validation_dataset=metaphor_validation_dataset,
        test_dataset=metaphor_test_dataset,
        batch_size=config.batch_size,
        shuffle=True)

    return metaphor_train_dataloader, metaphor_validation_dataloader

def calculate_metrics(targets, predictions):
    precision = metrics.precision_score(targets, predictions, average = "binary")
    recall = metrics.recall_score(targets, predictions, average = "binary")
    f1 = metrics.f1_score(targets, predictions, average="binary")

    return f1, precision, recall

def iterate_hyperpartisan(
    joint_model,
    optimizer,
    criterion,
    batch_inputs,
    batch_targets,
    batch_recover_idx,
    batch_num_sent,
    batch_sent_lengths,
    device,
    train: bool = False):

    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)
    batch_recover_idx = batch_recover_idx.to(device)
    batch_num_sent = batch_num_sent.to(device)
    batch_sent_lengths = batch_sent_lengths.to(device)

    if train:
        optimizer.zero_grad()
        
    predictions = joint_model.forward(batch_inputs, (batch_recover_idx,
                                    batch_num_sent, batch_sent_lengths), task=TrainingMode.Hyperpartisan)

    loss = criterion.forward(predictions, batch_targets)
    
    if train:
        loss.backward()
        optimizer.step()

    accuracy = get_accuracy(predictions, batch_targets)

    return loss.item(), accuracy.item(), batch_targets.long().tolist(), predictions.round().long().tolist()

def forward_full_hyperpartisan(
    joint_model,
    optimizer,
    criterion,
    dataloader,
    device,
    train: bool = False):

    all_targets = []
    all_predictions = []

    for step, (batch_inputs, batch_targets, batch_recover_idx, batch_num_sent, batch_sent_lengths) in enumerate(dataloader):

        loss, accuracy, batch_targets, batch_predictions = iterate_hyperpartisan(
            joint_model=joint_model,
            optimizer=optimizer,
            criterion=criterion,
            batch_inputs=batch_inputs,
            batch_targets=batch_targets,
            batch_recover_idx=batch_recover_idx,
            batch_num_sent=batch_num_sent,
            batch_sent_lengths=batch_sent_lengths,
            device=device,
            train=train)

        running_loss += loss
        running_accuracy += accuracy
        all_targets += batch_targets
        all_predictions += batch_predictions

    final_loss = running_loss / (step + 1)
    final_accuracy = running_accuracy / (step + 1)

    return final_loss, final_accuracy, all_targets, all_predictions

def iterate_metaphor(
    joint_model,
    optimizer,
    criterion,
    batch_inputs,
    batch_targets,
    batch_lengths,
    device,
    train: bool = False):
    batch_inputs = batch_inputs.to(device).float()
    batch_targets = batch_targets.to(device).view(-1).float()
    batch_lengths = batch_lengths.to(device)

    if train:
        optimizer.zero_grad()

    predictions = joint_model.forward(batch_inputs, batch_lengths, task=TrainingMode.Metaphor)
    
    unpadded_targets = batch_targets[batch_targets != -1]
    unpadded_predictions = predictions.view(-1)[batch_targets != -1]
    
    loss = criterion.forward(unpadded_predictions, unpadded_targets)
    
    if train:
        loss.backward()
        optimizer.step()

    return unpadded_targets.long().tolist(), unpadded_predictions.round().long().tolist()

def forward_full_metaphor(
    joint_model,
    optimizer,
    criterion,
    dataloader,
    device,
    train: bool = False):

    all_targets = []
    all_predictions = []

    for _, (batch_inputs, batch_targets, batch_lengths) in enumerate(dataloader):

        batch_targets, batch_predictions = iterate_metaphor(
            joint_model=joint_model,
            optimizer=optimizer,
            criterion=criterion,
            batch_inputs=batch_inputs,
            batch_targets=batch_targets,
            batch_lengths=batch_lengths,
            device=device,
            train=train)

        all_targets.extend(batch_targets)
        all_predictions.extend(batch_predictions)

    return all_targets, all_predictions

def train_model(config):
    """
    Train the multi-task classifier model
    :param config: Dictionary specifying the model configuration
    :return: None
    """
    # Flags for deterministic runs
    if config.deterministic:
        initialize_deterministic_mode()

    # Set device
    device = torch.device("cuda:0")  # if torch.cuda.is_available() else "cpu")

    # Load GloVe vectors
    glove_vectors = load_glove_vectors(config)

    # Define the model, the optimizer and the loss module
    joint_model, optimizer, start_epoch = initialize_model(config, device, glove_vectors.dim)

    metaphor_criterion = nn.BCELoss()
    hyperpartisan_criterion = nn.BCELoss()

    # Load hyperpartisan data
    if config.mode == TrainingMode.Hyperpartisan or config.mode == TrainingMode.Joint:
        hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader = create_hyperpartisan_loaders(config, glove_vectors)

    # Load metaphor data
    if config.mode == TrainingMode.Metaphor or config.mode == TrainingMode.Joint:
        metaphor_train_dataloader, metaphor_validation_dataloader = create_metaphor_loaders(config, glove_vectors)
    
    f1_validation_scores = []

    tic = time.process_time()

    for epoch in range(start_epoch, config.max_epochs + 1):
        if config.mode == TrainingMode.Metaphor or config.mode == TrainingMode.Joint:

            joint_model.train()
            
            forward_full_metaphor(
                joint_model=joint_model,
                optimizer=optimizer,
                criterion=metaphor_criterion,
                dataloader=metaphor_train_dataloader,
                device=device,
                train=True)

            joint_model.eval()
            
            val_targets, val_predictions = forward_full_metaphor(
                joint_model=joint_model,
                optimizer=None,
                criterion=metaphor_criterion,
                dataloader=metaphor_validation_dataloader,
                device=device)

            current_f1_score = metrics.f1_score(val_targets, val_predictions, average="binary")
            f1_validation_scores.append(current_f1_score)
            
            print("[{}] epoch {} || F1: valid = {:.4f}".format(
                datetime.now().time().replace(microsecond=0), epoch, current_f1_score))

        if config.mode == TrainingMode.Hyperpartisan or config.mode == TrainingMode.Joint:
            joint_model.train()

            loss_train, accuracy_train, _, _ = forward_full_hyperpartisan(
                joint_model=joint_model,
                optimizer=optimizer,
                criterion=hyperpartisan_criterion,
                dataloader=hyperpartisan_train_dataloader,
                device=device,
                train=True)

            joint_model.eval()

            loss_valid, accuracy_valid, valid_targets, valid_predictions = forward_full_hyperpartisan(
                joint_model=joint_model,
                optimizer=None,
                criterion=hyperpartisan_criterion,
                dataloader=hyperpartisan_validation_dataloader,
                device=device)

            f1, precision, recall = calculate_metrics(valid_targets, valid_predictions)
            f1_validation_scores.append(f1)

            print("[{}] epoch {} || LOSS: train = {:.4f}, valid = {:.4f} || ACCURACY: train = {:.4f}, valid = {:.4f}".format(
                datetime.now().time().replace(microsecond=0), epoch, loss_train, loss_valid, accuracy_train, accuracy_valid))
            print("     (valid): precision_score = {:.4f}, recall_score = {:.4f}, f1 = {:.4f}".format(
                precision, recall, f1))

    print("[{}] Training completed in {:.2f} minutes".format(datetime.now().time().replace(microsecond=0),
                                                             (time.process_time() - tic) / 60))

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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training the model')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='Hidden dimension of the recurrent network')
    parser.add_argument('--glove_size', type=int,
                        help='Number of GloVe vectors to load initially')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for the optimizer')
    parser.add_argument('--metaphor_dataset_folder', type=str, required=True,
                        help='Path to the metaphor dataset')
    parser.add_argument('--hyperpartisan_dataset_folder', type=str,
                        help='Path to the hyperpartisan dataset')
    parser.add_argument('--mode', type=TrainingMode, choices=list(TrainingMode), required=True,
                        help='The mode in which to train the model')
    parser.add_argument('--lowercase', action='store_true',
                        help='Lowercase the sentences before training')
    parser.add_argument('--not_tokenize', action='store_true',
                        help='Do not tokenize the sentences before training')
    parser.add_argument('--only_news', action='store_true',
                        help='Use only metaphors which have News as genre')
    parser.add_argument('--deterministic', action='store_true',
                        help='Make sure the training is done deterministically')

    config = parser.parse_args()

    train_model(config)
