import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from torchtext.vocab import Vectors

import torchtext
import os
import numpy as np

from typing import List

from datasets.hyperpartisan_dataset import HyperpartisanDataset
from datasets.metaphor_dataset import MetaphorDataset
from datasets.joint_dataset import JointDataset

from helpers.hyperpartisan_loader import HyperpartisanLoader
from helpers.metaphor_loader import MetaphorLoader

from helpers.data_helper import DataHelper
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan
from helpers.data_helper_joint import DataHelperJoint

from helpers.argument_parser_helper import ArgumentParserHelper
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
        accuracy = torch.sum(prediction == targets.byte()
                             ).float() / prediction.shape[0]
    else:
        prediction = torch.argmax(prediction_scores, dim=1)
        accuracy = torch.mean((prediction == targets).float())

    return accuracy


def load_glove_vectors(vector_file_name, vector_cache_dir, glove_size):

    print('Loading GloVe vectors...\r', end='')

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


def initialize_model(
        argument_parser: ArgumentParserHelper,
        device: torch.device,
        glove_vectors_dim: int):

    print('Loading model state...\r', end='')

    total_embedding_dim = elmo_vectors_size + glove_vectors_dim

    joint_model = JointModel(embedding_dim=total_embedding_dim,
                             hidden_dim=argument_parser.hidden_dim,
                             sent_encoder_dropout_rate=argument_parser.sent_encoder_dropout_rate,
                             doc_encoder_dropout_rate=argument_parser.doc_encoder_dropout_rate,
                             output_dropout_rate=argument_parser.output_dropout_rate,
                             device=device).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, joint_model.parameters()),
                           lr=argument_parser.learning_rate, weight_decay=argument_parser.weight_decay)

    # Load the checkpoint if found
    if argument_parser.load_model and os.path.isfile(argument_parser.model_checkpoint):
        checkpoint = torch.load(argument_parser.model_checkpoint)
        joint_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print('Found previous model state')
    else:
        start_epoch = 1
        print('Loading model state...Done')

    print("Starting training in '%s' mode from epoch %d..." %
          (argument_parser.mode, start_epoch))

    return joint_model, optimizer, start_epoch


def create_hyperpartisan_loaders(
        argument_parser: ArgumentParserHelper,
        glove_vectors: Vectors):

    hyperpartisan_train_dataset, hyperpartisan_validation_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
        hyperpartisan_dataset_folder=argument_parser.hyperpartisan_dataset_folder,
        glove_vectors=glove_vectors,
        lowercase_sentences=argument_parser.lowercase)

    hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader = DataHelperHyperpartisan.create_dataloaders(
        train_dataset=hyperpartisan_train_dataset,
        validation_dataset=hyperpartisan_validation_dataset,
        test_dataset=None,
        batch_size=argument_parser.batch_size,
        shuffle=True)

    return hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader


def create_metaphor_loaders(
        argument_parser: ArgumentParserHelper,
        glove_vectors: Vectors):

    metaphor_train_dataset, metaphor_validation_dataset, metaphor_test_dataset = MetaphorLoader.get_metaphor_datasets(
        metaphor_dataset_folder=argument_parser.metaphor_dataset_folder,
        glove_vectors=glove_vectors,
        lowercase_sentences=argument_parser.lowercase,
        tokenize_sentences=argument_parser.tokenize,
        only_news=argument_parser.only_news)

    metaphor_train_dataloader, metaphor_validation_dataloader, _ = DataHelper.create_dataloaders(
        train_dataset=metaphor_train_dataset,
        validation_dataset=metaphor_validation_dataset,
        test_dataset=metaphor_test_dataset,
        batch_size=argument_parser.batch_size,
        shuffle=True)

    return metaphor_train_dataloader, metaphor_validation_dataloader


def create_joint_loaders(
        argument_parser: ArgumentParserHelper,
        glove_vectors: Vectors):

    hyperpartisan_train_dataset, hyperpartisan_validation_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
        hyperpartisan_dataset_folder=argument_parser.hyperpartisan_dataset_folder,
        glove_vectors=glove_vectors,
        lowercase_sentences=argument_parser.lowercase)

    metaphor_train_dataset, metaphor_validation_dataset, _ = MetaphorLoader.get_metaphor_datasets(
        metaphor_dataset_folder=argument_parser.metaphor_dataset_folder,
        glove_vectors=glove_vectors,
        lowercase_sentences=argument_parser.lowercase,
        tokenize_sentences=argument_parser.tokenize,
        only_news=argument_parser.only_news)

    joint_train_dataset = JointDataset(
        metaphor_train_dataset, hyperpartisan_train_dataset)
    joint_validation_dataset = JointDataset(
        metaphor_validation_dataset, hyperpartisan_validation_dataset)

    joint_train_dataloader, joint_validation_dataloader = DataHelperJoint.create_dataloaders(
        train_dataset=joint_train_dataset,
        validation_dataset=joint_validation_dataset,
        batch_size=argument_parser.batch_size,
        shuffle=True)

    return joint_train_dataloader, joint_validation_dataloader


def calculate_metrics(
        targets: List,
        predictions: List):

    precision = metrics.precision_score(targets, predictions, average="binary")
    recall = metrics.recall_score(targets, predictions, average="binary")
    f1 = metrics.f1_score(targets, predictions, average="binary")

    return f1, precision, recall


def iterate_hyperpartisan(
        joint_model: JointModel,
        optimizer: Optimizer,
        criterion: Module,
        batch_inputs,
        batch_targets,
        batch_recover_idx,
        batch_num_sent,
        batch_sent_lengths,
        device: torch.device,
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
        joint_model: JointModel,
        optimizer: Optimizer,
        criterion: Module,
        dataloader: DataLoader,
        device: torch.device,
        train: bool = False):

    all_targets = []
    all_predictions = []

    running_loss = 0
    running_accuracy = 0

    for step, (batch_inputs, batch_targets, batch_recover_idx, batch_num_sent, batch_sent_lengths) in enumerate(dataloader):
        print(
            f'Step {step+1}/{dataloader.__len__()}                  \r', end='')

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
        joint_model: JointModel,
        optimizer: Optimizer,
        criterion: Module,
        batch_inputs,
        batch_targets,
        batch_lengths,
        device: torch.device,
        train: bool = False):
    batch_inputs = batch_inputs.to(device).float()
    batch_targets = batch_targets.to(device).view(-1).float()
    batch_lengths = batch_lengths.to(device)

    if train:
        optimizer.zero_grad()

    predictions = joint_model.forward(
        batch_inputs, batch_lengths, task=TrainingMode.Metaphor)

    unpadded_targets = batch_targets[batch_targets != -1]
    unpadded_predictions = predictions.view(-1)[batch_targets != -1]

    loss = criterion.forward(unpadded_predictions, unpadded_targets)

    if train:
        loss.backward()
        optimizer.step()

    return unpadded_targets.long().tolist(), unpadded_predictions.round().long().tolist()


def forward_full_metaphor(
        joint_model: JointModel,
        optimizer: Optimizer,
        criterion: Module,
        dataloader: DataLoader,
        device: torch.device,
        train: bool = False):

    all_targets = []
    all_predictions = []

    for step, (batch_inputs, batch_targets, batch_lengths) in enumerate(dataloader):
        print(
            f'Step {step+1}/{dataloader.__len__()}                  \r', end='')

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


def forward_full_joint_batches(
        joint_model: JointModel,
        optimizer: Optimizer,
        metaphor_criterion: Module,
        hyperpartisan_criterion: Module,
        dataloader: DataLoader,
        device: torch.device,
        joint_metaphors_first: bool,
        eval_func = None,
        eval_every: int = 50,
        train: bool = False):

    all_hyperpartisan_targets = []
    all_hyperpartisan_predictions = []

    running_hyperpartisan_loss = 0
    running_hyperpartisan_accuracy = 0

    for step, (metaphor_batch, hyperpartisan_batch) in enumerate(dataloader):
        print(
            f'Step {step+1}/{dataloader.__len__()}                  \r', end='')

        if joint_metaphors_first:
            _, _ = iterate_metaphor(
                joint_model=joint_model,
                optimizer=optimizer,
                criterion=metaphor_criterion,
                batch_inputs=metaphor_batch[0],
                batch_targets=metaphor_batch[1],
                batch_lengths=metaphor_batch[2],
                device=device,
                train=train)

        loss, accuracy, batch_targets, batch_predictions = iterate_hyperpartisan(
            joint_model=joint_model,
            optimizer=optimizer,
            criterion=hyperpartisan_criterion,
            batch_inputs=hyperpartisan_batch[0],
            batch_targets=hyperpartisan_batch[1],
            batch_recover_idx=hyperpartisan_batch[2],
            batch_num_sent=hyperpartisan_batch[3],
            batch_sent_lengths=hyperpartisan_batch[4],
            device=device,
            train=train)

        if not joint_metaphors_first:
            _, _ = iterate_metaphor(
                joint_model=joint_model,
                optimizer=optimizer,
                criterion=metaphor_criterion,
                batch_inputs=metaphor_batch[0],
                batch_targets=metaphor_batch[1],
                batch_lengths=metaphor_batch[2],
                device=device,
                train=train)

        running_hyperpartisan_loss += loss
        running_hyperpartisan_accuracy += accuracy

        all_hyperpartisan_targets.extend(batch_targets)
        all_hyperpartisan_predictions.extend(batch_predictions)

        if step > 0 and step % eval_every == 0:
            eval_func()

    final_loss = running_hyperpartisan_loss / (step + 1)
    final_accuracy = running_hyperpartisan_accuracy / (step + 1)

    return final_loss, final_accuracy, all_hyperpartisan_targets, all_hyperpartisan_predictions


def train_and_eval_hyperpartisan(
        joint_model: JointModel,
        optimizer: Optimizer,
        hyperpartisan_criterion: Module,
        hyperpartisan_train_dataloader: DataLoader,
        hyperpartisan_validation_dataloader: DataLoader,
        device: torch.device,
        epoch: int):

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

    print("[{}] HYPERPARTISAN -> epoch {} || LOSS: train = {:.4f}, valid = {:.4f} || "
          "ACCURACY: train = {:.4f}, valid = {:.4f} || PRECISION: valid = {:.4f}, RECALL: valid = {:.4f} || "
          "F1 SCORE: valid = {:.4f}".format(
        datetime.now().time().replace(microsecond=0), epoch, loss_train, loss_valid, accuracy_train, accuracy_valid,
        precision, recall, f1))

    return f1


def train_and_eval_metaphor(
        joint_model: JointModel,
        optimizer: Optimizer,
        metaphor_criterion: Module,
        metaphor_train_dataloader: DataLoader,
        metaphor_validation_dataloader: DataLoader,
        device: torch.device,
        epoch: int):

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

    f1, _, _ = calculate_metrics(val_targets, val_predictions)

    print("[{}] METAPHOR -> epoch {} || F1 SCORE: valid = {:.4f}".format(
        datetime.now().time().replace(microsecond=0), epoch, f1))

    return f1


def train_and_eval_joint(
        joint_model: JointModel,
        optimizer: Optimizer,
        hyperpartisan_criterion: Module,
        metaphor_criterion: Module,
        joint_train_dataloader: DataLoader,
        metaphor_validation_dataloader: DataLoader,
        hyperpartisan_validation_dataloader: DataLoader,
        device: torch.device,
        eval_every: int,
        joint_metaphors_first: bool,
        epoch: int):

    joint_model.train()

    train_loss, train_accuracy, _, _ = forward_full_joint_batches(
        joint_model=joint_model,
        optimizer=optimizer,
        metaphor_criterion=metaphor_criterion,
        hyperpartisan_criterion=hyperpartisan_criterion,
        dataloader=joint_train_dataloader,
        device=device,
        eval_func=lambda: evaluate_joint_batches(joint_model, hyperpartisan_criterion, hyperpartisan_validation_dataloader, device),
        eval_every=eval_every,
        joint_metaphors_first=joint_metaphors_first,
        train=True)

    joint_model.eval()

    val_targets, val_predictions = forward_full_metaphor(
        joint_model=joint_model,
        optimizer=None,
        criterion=metaphor_criterion,
        dataloader=metaphor_validation_dataloader,
        device=device)

    f1, precision, recall = calculate_metrics(val_targets, val_predictions)

    print("[{}] METAPHOR -> epoch {} || F1 SCORE: valid = {:.4f}".format(
        datetime.now().time().replace(microsecond=0), epoch, f1))

    valid_loss, valid_accuracy, valid_targets, valid_predictions = forward_full_hyperpartisan(
        joint_model=joint_model,
        optimizer=None,
        criterion=hyperpartisan_criterion,
        dataloader=hyperpartisan_validation_dataloader,
        device=device)

    f1, precision, recall = calculate_metrics(valid_targets, valid_predictions)

    print("[{}] HYPERPARTISAN -> epoch {} || LOSS: train = {:.4f}, valid = {:.4f} || ACCURACY: train = {:.4f}, "
          "valid = {:.4f} || PRECISION: valid = {:.4f} || RECALL: valid = {:.4f} || F1 SCORE = {:.4f}".format(
        datetime.now().time().replace(microsecond=0), epoch, train_loss, valid_loss, train_accuracy, valid_accuracy,
        precision, recall, f1))

    return f1


def evaluate_joint_batches(
    joint_model: JointModel,
    hyperpartisan_criterion: Module,
    hyperpartisan_validation_dataloader: DataLoader,
    device: torch.device):
    
    joint_model.eval()

    loss_valid, accuracy_valid, valid_targets, valid_predictions = forward_full_hyperpartisan(
        joint_model=joint_model,
        optimizer=None,
        criterion=hyperpartisan_criterion,
        dataloader=hyperpartisan_validation_dataloader,
        device=device)

    f1, precision, recall = calculate_metrics(valid_targets, valid_predictions)

    print("[{}] HYPERPARTISAN -> LOSS: valid = {:.4f} || ACCURACY: valid = {:.4f} || "
          "PRECISION: valid = {:.4f} || RECALL: valid = {:.4f} || F1 SCORE = {:.4f}".format(
        datetime.now().time().replace(microsecond=0), loss_valid, accuracy_valid, precision, recall, f1))

    joint_model.train()


def train_model(argument_parser: ArgumentParserHelper):
    """
    Train the multi-task classifier model
    :param argument_parser: Dictionary specifying the model configuration
    :return: None
    """
    # Flags for deterministic runs
    if argument_parser.deterministic:
        initialize_deterministic_mode()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load GloVe vectors
    glove_vectors = load_glove_vectors(argument_parser.vector_file_name, argument_parser.vector_cache_dir, argument_parser.glove_size)

    # Define the model, the optimizer and the loss module
    joint_model, optimizer, start_epoch = initialize_model(
        argument_parser, device, glove_vectors.dim)

    metaphor_criterion = nn.BCELoss()
    hyperpartisan_criterion = nn.BCELoss()

    # Load hyperpartisan data
    if TrainingMode.contains_hyperpartisan(argument_parser.mode):
        hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader = create_hyperpartisan_loaders(
            argument_parser, glove_vectors)

    # Load metaphor data
    if TrainingMode.contains_metaphor(argument_parser.mode):
        metaphor_train_dataloader, metaphor_validation_dataloader = create_metaphor_loaders(
            argument_parser, glove_vectors)

    # Load Joint batches data
    if argument_parser.mode == TrainingMode.JointBatches:
        joint_train_dataloader, joint_validation_dataloader = create_joint_loaders(
            argument_parser, glove_vectors)

    tic = time.process_time()

    best_f1 = .0

    for epoch in range(start_epoch, argument_parser.max_epochs + 1):
        # Joint mode by batches
        if argument_parser.mode == TrainingMode.JointBatches:
            f1 = train_and_eval_joint(
                joint_model=joint_model,
                optimizer=optimizer,
                hyperpartisan_criterion=hyperpartisan_criterion,
                metaphor_criterion=metaphor_criterion,
                joint_train_dataloader=joint_train_dataloader,
                metaphor_validation_dataloader=metaphor_validation_dataloader,
                hyperpartisan_validation_dataloader=hyperpartisan_validation_dataloader,
                device=device,
                eval_every=argument_parser.joint_eval_every,
                joint_metaphors_first=argument_parser.joint_metaphors_first,
                epoch=epoch)

        else:
            # Joint mode by epochs or single training
            if TrainingMode.contains_metaphor(argument_parser.mode) and argument_parser.joint_metaphors_first:
                # Complete one epoch of metaphors BEFORE the hyperpartisan
                f1 = train_and_eval_metaphor(
                    joint_model=joint_model,
                    optimizer=optimizer,
                    metaphor_criterion=metaphor_criterion,
                    metaphor_train_dataloader=metaphor_train_dataloader,
                    metaphor_validation_dataloader=metaphor_validation_dataloader,
                    device=device,
                    epoch=epoch)

            if TrainingMode.contains_hyperpartisan(argument_parser.mode):
                # Complete one epoch of hyperpartisan
                f1 = train_and_eval_hyperpartisan(
                    joint_model=joint_model,
                    optimizer=optimizer,
                    hyperpartisan_criterion=hyperpartisan_criterion,
                    hyperpartisan_train_dataloader=hyperpartisan_train_dataloader,
                    hyperpartisan_validation_dataloader=hyperpartisan_validation_dataloader,
                    device=device,
                    epoch=epoch)
            
            if TrainingMode.contains_metaphor(argument_parser.mode) and not argument_parser.joint_metaphors_first:
                # Complete one epoch of metaphors AFTER the hyperpartisan
                f1 = train_and_eval_metaphor(
                    joint_model=joint_model,
                    optimizer=optimizer,
                    metaphor_criterion=metaphor_criterion,
                    metaphor_train_dataloader=metaphor_train_dataloader,
                    metaphor_validation_dataloader=metaphor_validation_dataloader,
                    device=device,
                    epoch=epoch)

        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model_state_dict': joint_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch},
                       argument_parser.model_checkpoint)

    print("[{}] Training completed in {:.2f} minutes".format(datetime.now().time().replace(microsecond=0),
                                                             (time.process_time() - tic) / 60))


if __name__ == '__main__':
    argument_parser = ArgumentParserHelper()
    argument_parser.parse_arguments()
    argument_parser.print_unique_arguments()

    train_model(argument_parser)
