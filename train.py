import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from torchtext.vocab import Vectors

import os

from datasets.hyperpartisan_dataset import HyperpartisanDataset
from datasets.metaphor_dataset import MetaphorDataset

from helpers.hyperpartisan_loader import HyperpartisanLoader
from helpers.metaphor_loader import MetaphorLoader

from helpers.data_helper import DataHelper
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan

from helpers.argument_parser_helper import ArgumentParserHelper
from helpers.utils_helper import UtilsHelper
from model.JointModel import JointModel

from enums.training_mode import TrainingMode

from datetime import datetime
import time
import itertools

from constants import Constants
from tensorboardX import SummaryWriter

utils_helper = UtilsHelper()

def initialize_model(
        argument_parser: ArgumentParserHelper,
        device: torch.device,
        glove_vectors_dim: int):

    print('Loading model state...\r', end='')

    total_embedding_dim = Constants.DEFAULT_ELMO_EMBEDDING_DIMENSION + glove_vectors_dim

    joint_model = JointModel(embedding_dim=total_embedding_dim,
                             hidden_dim=argument_parser.hidden_dim,
                             num_layers=argument_parser.num_layers,
                             sent_encoder_dropout_rate=argument_parser.sent_encoder_dropout_rate,
                             doc_encoder_dropout_rate=argument_parser.doc_encoder_dropout_rate,
                             output_dropout_rate=argument_parser.output_dropout_rate,
                             device=device).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, joint_model.parameters()),
                           lr=argument_parser.learning_rate, weight_decay=argument_parser.weight_decay)

    # Load the checkpoint if found
    start_epoch = 1

    if argument_parser.load_model and os.path.isfile(argument_parser.model_checkpoint):
        checkpoint = torch.load(argument_parser.model_checkpoint)
        joint_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['epoch']:
            start_epoch = checkpoint['epoch'] + 1

        print('Found previous model state')
    else:
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
        lowercase_sentences=argument_parser.lowercase,
        articles_max_length=argument_parser.hyperpartisan_max_length)

    hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
        train_dataset=hyperpartisan_train_dataset,
        validation_dataset=hyperpartisan_validation_dataset,
        test_dataset=None,
        batch_size=argument_parser.hyperpartisan_batch_size,
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
        batch_size=argument_parser.metaphor_batch_size,
        shuffle=True)

    return metaphor_train_dataloader, metaphor_validation_dataloader

def iterate_hyperpartisan(
        joint_model: JointModel,
        optimizer: Optimizer,
        criterion: Module,
        hyperpartisan_data,
        device: torch.device,
        train: bool = False,
        loss_suppress_factor=1):

    batch_inputs = hyperpartisan_data[0].to(device)
    batch_targets = hyperpartisan_data[1].to(device)
    batch_recover_idx = hyperpartisan_data[2].to(device)
    batch_num_sent = hyperpartisan_data[3].to(device)
    batch_sent_lengths = hyperpartisan_data[4].to(device)
    batch_feat = hyperpartisan_data[5].to(device)

    if train:
        optimizer.zero_grad()

    predictions = joint_model.forward(batch_inputs, (batch_recover_idx,
                                                     batch_num_sent, batch_sent_lengths, batch_feat), task=TrainingMode.Hyperpartisan)

    loss = loss_suppress_factor * criterion.forward(predictions, batch_targets)

    if train:
        loss.backward()
        optimizer.step()

    accuracy = utils_helper.calculate_accuracy(predictions, batch_targets)

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

    total_length = len(dataloader)
    for step, hyperpartisan_data in enumerate(dataloader):
        print(f'Step {step+1}/{total_length}                  \r', end='')

        loss, accuracy, batch_targets, batch_predictions = iterate_hyperpartisan(
            joint_model=joint_model,
            optimizer=optimizer,
            criterion=criterion,
            hyperpartisan_data=hyperpartisan_data,
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
        metaphor_data,
        device: torch.device,
        train: bool = False):
    batch_inputs = metaphor_data[0].to(device).float()
    batch_targets = metaphor_data[1].to(device).view(-1).float()
    batch_lengths = metaphor_data[2].to(device)

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

    total_length = len(dataloader)
    for step, metaphor_data in enumerate(dataloader):
        print(f'Step {step+1}/{total_length}                  \r', end='')

        batch_targets, batch_predictions = iterate_metaphor(
            joint_model=joint_model,
            optimizer=optimizer,
            criterion=criterion,
            metaphor_data=metaphor_data,
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
        hyperpartisan_dataloader: DataLoader,
        metaphor_dataloader: DataLoader,
        device: torch.device,
        joint_metaphors_first: bool,
        loss_suppress_factor: float,
        train: bool = False):

    all_hyperpartisan_targets = []
    all_hyperpartisan_predictions = []

    running_hyperpartisan_loss = 0
    running_hyperpartisan_accuracy = 0

    total_length = max(len(hyperpartisan_dataloader), len(metaphor_dataloader))

    hyperpartisan_iterator = hyperpartisan_dataloader
    metaphor_iterator = metaphor_dataloader
    if len(hyperpartisan_dataloader) < len(metaphor_dataloader):
        hyperpartisan_iterator = itertools.cycle(hyperpartisan_iterator)
    else:
        metaphor_iterator = itertools.cycle(metaphor_iterator)

    for step, (hyperpartisan_batch, metaphor_batch) in enumerate(zip(hyperpartisan_iterator, metaphor_iterator)):
        print(f'Step {step+1}/{total_length}                  \r', end='')

        assert hyperpartisan_batch != None
        assert metaphor_batch != None

        if joint_metaphors_first and metaphor_batch != None:
            _, _ = iterate_metaphor(
                joint_model=joint_model,
                optimizer=optimizer,
                criterion=metaphor_criterion,
                metaphor_data=metaphor_batch,
                device=device,
                train=train)

        if hyperpartisan_batch != None:
            loss, accuracy, batch_targets, batch_predictions = iterate_hyperpartisan(
                joint_model=joint_model,
                optimizer=optimizer,
                criterion=hyperpartisan_criterion,
                hyperpartisan_data=hyperpartisan_batch,
                device=device,
                train=train,
                loss_suppress_factor=loss_suppress_factor)

            running_hyperpartisan_loss += loss
            running_hyperpartisan_accuracy += accuracy
            
            all_hyperpartisan_targets.extend(batch_targets)
            all_hyperpartisan_predictions.extend(batch_predictions)

        if not joint_metaphors_first and metaphor_batch != None:
            _, _ = iterate_metaphor(
                joint_model=joint_model,
                optimizer=optimizer,
                criterion=metaphor_criterion,
                metaphor_data=metaphor_batch,
                device=device,
                train=train)

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
        best_f1_score: int,
        summary_writer: SummaryWriter,
        epoch: int):

    joint_model.train()

    loss_train, accuracy_train, _, _ = forward_full_hyperpartisan(joint_model=joint_model,
                                                                  optimizer=optimizer,
                                                                  criterion=hyperpartisan_criterion,
                                                                  dataloader=hyperpartisan_train_dataloader,
                                                                  device=device, train=True)

    joint_model.eval()

    loss_valid, accuracy_valid, valid_targets, valid_predictions = forward_full_hyperpartisan(joint_model=joint_model,
                                                                                              optimizer=None,
                                                                                              criterion=hyperpartisan_criterion,
                                                                                              dataloader=hyperpartisan_validation_dataloader,
                                                                                              device=device)

    f1, precision, recall = utils_helper.calculate_metrics(valid_targets, valid_predictions)

    log_metrics(
        summary_writer,
        epoch,
        loss_train,
        accuracy_train,
        loss_valid,
        accuracy_valid,
        precision,
        recall,
        f1)

    print_hyperpartisan_stats(
        train_loss=loss_train,
        valid_loss=loss_valid,
        train_accuracy=accuracy_train,
        valid_accuracy=accuracy_valid,
        valid_precision=precision,
        valid_recall=recall,
        valid_f1=f1,
        new_best_score=(best_f1_score < f1),
        epoch=epoch)

    return f1


def train_and_eval_metaphor(
        joint_model: JointModel,
        optimizer: Optimizer,
        metaphor_criterion: Module,
        metaphor_train_dataloader: DataLoader,
        metaphor_validation_dataloader: DataLoader,
        device: torch.device,
        best_f1_score: int,
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

    f1, _, _ = utils_helper.calculate_metrics(val_targets, val_predictions)

    print_metaphor_stats(f1, epoch, (best_f1_score < f1))

    return f1

def train_and_eval_joint(
        joint_model: JointModel,
        optimizer: Optimizer,
        hyperpartisan_criterion: Module,
        metaphor_criterion: Module,
        hyperpartisan_train_dataloader: DataLoader,
        metaphor_train_dataloader: DataLoader,
        metaphor_validation_dataloader: DataLoader,
        hyperpartisan_validation_dataloader: DataLoader,
        device: torch.device,
        joint_metaphors_first: bool,
        epoch: int,
        loss_suppress_factor: float,
        summary_writer: SummaryWriter,
        best_hyperpartisan_f1_score: bool):

    # Train
    joint_model.train()

    train_loss, train_accuracy, _, _ = forward_full_joint_batches(
        joint_model=joint_model,
        optimizer=optimizer,
        metaphor_criterion=metaphor_criterion,
        hyperpartisan_criterion=hyperpartisan_criterion,
        hyperpartisan_dataloader=hyperpartisan_train_dataloader,
        metaphor_dataloader=metaphor_train_dataloader,
        device=device,
        loss_suppress_factor=loss_suppress_factor,
        joint_metaphors_first=joint_metaphors_first,
        train=True)

    # Evaluate

    joint_model.eval()

    # Metaphor
    
    val_targets, val_predictions = forward_full_metaphor(
        joint_model=joint_model,
        optimizer=None,
        criterion=metaphor_criterion,
        dataloader=metaphor_validation_dataloader,
        device=device)

    metaphor_f1, _, _ = utils_helper.calculate_metrics(val_targets, val_predictions)

    print_metaphor_stats(metaphor_f1, epoch, False)

    # Hyperpartisan

    valid_loss, valid_accuracy, valid_targets, valid_predictions = forward_full_hyperpartisan(
        joint_model=joint_model,
        optimizer=None,
        criterion=hyperpartisan_criterion,
        dataloader=hyperpartisan_validation_dataloader,
        device=device)

    hyperpartisan_f1, precision, recall = utils_helper.calculate_metrics(valid_targets, valid_predictions)

    # Log results

    log_metrics(
        summary_writer=summary_writer,
        global_step=epoch,
        loss_train=train_loss,
        accuracy_train=train_accuracy,
        valid_loss=valid_loss,
        valid_accuracy=valid_accuracy,
        valid_precision=precision,
        valid_recall=recall,
        valid_f1=hyperpartisan_f1)

    print_hyperpartisan_stats(
        train_loss=train_loss,
        valid_loss=valid_loss,
        train_accuracy=train_accuracy,
        valid_accuracy=valid_accuracy,
        valid_precision=precision,
        valid_recall=recall,
        valid_f1=hyperpartisan_f1,
        epoch=epoch,
        new_best_score=(best_hyperpartisan_f1_score < hyperpartisan_f1))

    return hyperpartisan_f1

def print_hyperpartisan_stats(
        train_loss,
        valid_loss,
        train_accuracy,
        valid_accuracy,
        valid_precision,
        valid_recall,
        valid_f1,
        epoch=None,
        new_best_score: bool = False):

    epoch_str = str(epoch) if epoch is not None else '_'

    new_best_str = '<- new best result' if new_best_score else ''

    print("[{}] HYPERPARTISAN -> epoch {} || LOSS: train = {:.4f}, valid = {:.4f} || ACCURACY: train = {:.4f}, "
          "valid = {:.4f} || PRECISION: valid = {:.4f} || RECALL: valid = {:.4f} || F1 SCORE = {:.4f} {}".format(
              datetime.now().time().replace(microsecond=0), epoch_str, train_loss, valid_loss, train_accuracy, valid_accuracy, valid_precision, valid_recall, valid_f1, new_best_str))


def print_metaphor_stats(f1, epoch, new_best_score):
    new_best_str = '<- new best result' if new_best_score else ''

    print("[{}] METAPHOR -> epoch {} || F1 SCORE: valid = {:.4f} {}".format(
        datetime.now().time().replace(microsecond=0), epoch, f1, new_best_str))


def cache_model(joint_model, optimizer, epoch=None):
    torch_state = {'model_state_dict': joint_model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict()}

    if epoch is not None:
        torch_state['epoch'] = epoch

    torch.save(torch_state, argument_parser.model_checkpoint)

def log_metrics(
        summary_writer: SummaryWriter,
        global_step,
        loss_train,
        accuracy_train,
        valid_loss,
        valid_accuracy,
        valid_precision,
        valid_recall,
        valid_f1):

    summary_writer.add_scalar(
        'train_loss', loss_train, global_step=global_step)
    summary_writer.add_scalar(
        'train_accuracy', accuracy_train, global_step=global_step)
    summary_writer.add_scalar(
        'valid_loss', valid_loss, global_step=global_step)
    summary_writer.add_scalar(
        'valid_accuracy', valid_accuracy, global_step=global_step)
    summary_writer.add_scalar(
        'valid_precision', valid_precision, global_step=global_step)
    summary_writer.add_scalar(
        'valid_recall', valid_recall, global_step=global_step)
    summary_writer.add_scalar(
        'valid_f1', valid_f1, global_step=global_step)

def save_best_result(arg_parser: ArgumentParserHelper, best_f1: float):
    titles = ['time', 'f1']
    values = [str(datetime.now().time().replace(microsecond=0)), str(best_f1)]
    for key, value in arg_parser.__dict__.items():
        titles.append(str(key))
        values.append(str(value))

    results_filepath = 'results.csv'
    with open(results_filepath, mode='a') as results_file:
        if os.stat(results_filepath).st_size == 0:
            results_file.write(', '.join(titles))
            results_file.write('\n')

        results_file.write(', '.join(values))
        results_file.write('\n')

def train_model(argument_parser: ArgumentParserHelper):
    """
    Train the multi-task classifier model
    :param argument_parser: Dictionary specifying the model configuration
    :return: None
    """

    # Flags for deterministic runs
    if argument_parser.deterministic:
        utils_helper.initialize_deterministic_mode(argument_parser.deterministic)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load GloVe vectors
    glove_vectors = utils_helper.load_glove_vectors(
        argument_parser.vector_file_name, argument_parser.vector_cache_dir, argument_parser.glove_size)

    # Define the model, the optimizer and the loss module
    joint_model, optimizer, start_epoch = initialize_model(
        argument_parser, device, glove_vectors.dim)

    summary_writer = SummaryWriter(
        f'runs/exp-{argument_parser.mode}-odr_{argument_parser.output_dropout_rate}-lr_{argument_parser.learning_rate}-wd_{argument_parser.weight_decay}-lsf_{argument_parser.loss_suppress_factor}')
    
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
                hyperpartisan_train_dataloader=hyperpartisan_train_dataloader,
                metaphor_train_dataloader=metaphor_train_dataloader,
                metaphor_validation_dataloader=metaphor_validation_dataloader,
                hyperpartisan_validation_dataloader=hyperpartisan_validation_dataloader,
                device=device,
                joint_metaphors_first=argument_parser.joint_metaphors_first,
                epoch=epoch,
                loss_suppress_factor=argument_parser.loss_suppress_factor,
                summary_writer=summary_writer,
                best_hyperpartisan_f1_score=best_f1)

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
                    best_f1_score=best_f1,
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
                    best_f1_score=best_f1,
                    epoch=epoch,
                    summary_writer=summary_writer)

            if TrainingMode.contains_metaphor(argument_parser.mode) and not argument_parser.joint_metaphors_first:
                # Complete one epoch of metaphors AFTER the hyperpartisan
                f1 = train_and_eval_metaphor(
                    joint_model=joint_model,
                    optimizer=optimizer,
                    metaphor_criterion=metaphor_criterion,
                    metaphor_train_dataloader=metaphor_train_dataloader,
                    metaphor_validation_dataloader=metaphor_validation_dataloader,
                    device=device,
                    best_f1_score=best_f1,
                    epoch=epoch)

        if f1 > best_f1:
            best_f1 = f1
            cache_model(joint_model=joint_model,
                        optimizer=optimizer,
                        epoch=epoch)

    print("[{}] Training completed in {:.2f} minutes".format(datetime.now().time().replace(microsecond=0),
                                                             (time.process_time() - tic) / 60))

    save_best_result(argument_parser, best_f1)

    summary_writer.close()


if __name__ == '__main__':
    argument_parser = ArgumentParserHelper()
    argument_parser.parse_arguments()
    argument_parser.print_unique_arguments()

    train_model(argument_parser)
