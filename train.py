import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from torchtext.vocab import Vectors

import numpy as np
import os

from batches.hyperpartisan_batch import HyperpartisanBatch

from enums.elmo_model import ELMoModel
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
        device: torch.device):

    print('Loading model state...\r', end='')

    if argument_parser.elmo_model == ELMoModel.Original:
        total_embedding_dim = Constants.ORIGINAL_ELMO_EMBEDDING_DIMENSION
    elif argument_parser.elmo_model == ELMoModel.Small:
        total_embedding_dim = Constants.SMALL_ELMO_EMBEDDING_DIMENSION

    if argument_parser.concat_glove:
        total_embedding_dim += Constants.GLOVE_EMBEDDING_DIMENSION

    joint_model = JointModel(embedding_dim=total_embedding_dim,
                             sent_encoder_hidden_dim=argument_parser.sent_encoder_hidden_dim,
                             doc_encoder_hidden_dim=argument_parser.doc_encoder_hidden_dim,
                             num_layers=argument_parser.num_layers,
                             sent_encoder_dropout_rate=argument_parser.sent_encoder_dropout_rate,
                             doc_encoder_dropout_rate=argument_parser.doc_encoder_dropout_rate,
                             output_dropout_rate=argument_parser.output_dropout_rate,
                             device=device,
                             skip_connection=argument_parser.skip_connection,
                             include_article_features=argument_parser.include_article_features,
                             doc_encoder_model=argument_parser.document_encoder_model,
                             pre_attn_layer=argument_parser.pre_attention_layer
                             ).to(device)

    # Load the checkpoint if found
    start_epoch = 1

    if argument_parser.load_model and os.path.isfile(argument_parser.model_checkpoint):
        checkpoint = torch.load(argument_parser.model_checkpoint)
        joint_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['epoch']:
            start_epoch = checkpoint['epoch'] + 1

        print('Found previous model state')
    elif argument_parser.load_pretrained and os.path.isfile(argument_parser.pretrained_path):
        snli_checkpoint = torch.load(argument_parser.pretrained_path)
        joint_model.load_my_state_dict(snli_checkpoint['model_state_dict'])

        if argument_parser.freeze_sentence_encoder:
            for name, p in joint_model.named_parameters():
                if "sentence_encoder" in name:
                    p.requires_grad = False

        print("Loaded pre-trained sentence encoder")
    else:
        print('Loading model state...Done')


    if argument_parser.per_layer_config:

        sentence_enc_weights = [p for n, p in joint_model.sentence_encoder.encoder.named_parameters() if (p.requires_grad and "weight" in n)]
        document_enc_weights = [p for n, p in joint_model.document_encoder.encoder.named_parameters() if (p.requires_grad and "weight" in n)]
        classifer_weights = [p for n, p in joint_model.named_parameters() if (p.requires_grad and ("weight" in n) and ("fc" in n))]
        context_and_biases = [p for n, p in joint_model.named_parameters() if (p.requires_grad and (("bias" in n) or ("context" in n)))]

        assert len(sentence_enc_weights) + len(document_enc_weights) + len(classifer_weights) + len(context_and_biases) == len([p for p in joint_model.parameters() if p.requires_grad])

        optimizer = optim.Adam([
                        {"params": sentence_enc_weights, "lr": argument_parser.learning_rate / 10, "weight_decay": argument_parser.weight_decay * 5},
                        {"params": document_enc_weights, "lr": argument_parser.learning_rate * 10, "weight_decay": 0},
                        {"params": classifer_weights, "lr": argument_parser.learning_rate, "weight_decay": argument_parser.weight_decay},
                        {"params": context_and_biases, "lr": argument_parser.learning_rate, "weight_decay": .0}
                        ])

    else:

        optimizer = optim.Adam([{"params": [p for p in joint_model.parameters() if p.requires_grad]}],
                                lr=argument_parser.learning_rate, weight_decay=argument_parser.weight_decay)
        
    print("Starting training in '%s' mode from epoch %d..." %
          (argument_parser.mode, start_epoch))

    return joint_model, optimizer, start_epoch


def create_hyperpartisan_loaders(
        argument_parser: ArgumentParserHelper,
        glove_vectors: Vectors):

    hyperpartisan_train_dataset, hyperpartisan_validation_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
        hyperpartisan_dataset_folder=argument_parser.hyperpartisan_dataset_folder,
        concat_glove=argument_parser.concat_glove,
        glove_vectors=glove_vectors,
        elmo_model=argument_parser.elmo_model,
        lowercase_sentences=argument_parser.lowercase,
        articles_max_length=argument_parser.hyperpartisan_max_length)

    hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
        train_dataset=hyperpartisan_train_dataset,
        validation_dataset=hyperpartisan_validation_dataset,
        test_dataset=None,
        batch_size=argument_parser.hyperpartisan_batch_size,
        shuffle=True)

    pos_weight = hyperpartisan_train_dataset.pos_weight

    return hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader, pos_weight

def create_metaphor_loaders(
        argument_parser: ArgumentParserHelper,
        glove_vectors: Vectors):

    metaphor_train_dataset, metaphor_validation_dataset, metaphor_test_dataset = MetaphorLoader.get_metaphor_datasets(
        metaphor_dataset_folder=argument_parser.metaphor_dataset_folder,
        concat_glove=argument_parser.concat_glove,
        glove_vectors=glove_vectors,
        elmo_model=argument_parser.elmo_model,
        lowercase_sentences=argument_parser.lowercase,
        tokenize_sentences=argument_parser.tokenize,
        only_news=argument_parser.only_news)

    pos_weight = metaphor_train_dataset.pos_weight

    metaphor_train_dataloader, metaphor_validation_dataloader, _ = DataHelper.create_dataloaders(
        train_dataset=metaphor_train_dataset,
        validation_dataset=metaphor_validation_dataset,
        test_dataset=metaphor_test_dataset,
        batch_size=argument_parser.metaphor_batch_size,
        shuffle=True)

    return metaphor_train_dataloader, metaphor_validation_dataloader, pos_weight

def iterate_hyperpartisan(
        joint_model: JointModel,
        optimizer: Optimizer,
        criterion: Module,
        hyperpartisan_data,
        device: torch.device,
        step: int,
        epoch: int,
        gradient_save_path: str,
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

    loss = loss_suppress_factor * criterion(predictions, batch_targets)

    if train:
        loss.backward()

        if step == 0:
           utils_helper.plot_grad_flow(joint_model.named_parameters(), gradient_save_path, epoch, "hyperpartisan")

        optimizer.step()

    accuracy = utils_helper.calculate_accuracy(predictions, batch_targets)

    return loss.item(), accuracy.item(), batch_targets.long().tolist(), predictions.round().long().tolist()


def forward_full_hyperpartisan(
        joint_model: JointModel,
        optimizer: Optimizer,
        criterion: Module,
        dataloader: DataLoader,
        device: torch.device,
        hyperpartisan_batch_max_size: int,
        epoch: int,
        gradient_save_path: str,
        train: bool = False):

    all_targets = []
    all_predictions = []

    running_loss = 0
    running_accuracy = 0

    total_length = len(dataloader)
    hyperpartisan_iterator = enumerate(dataloader)
    batches_counter = 0

    while True:
        try:
            step, hyperpartisan_data = next(hyperpartisan_iterator)
        except StopIteration:
            break

        batches_counter += 1
        print(f'Step {step+1}/{total_length}                  \r', end='')
        
        hyperpartisan_batch = HyperpartisanBatch(hyperpartisan_batch_max_size)
        hyperpartisan_batch.add_data(hyperpartisan_data[0], hyperpartisan_data[1].item(), hyperpartisan_data[2], hyperpartisan_data[3].item(), hyperpartisan_data[4])
        
        # get bigger than 1 batches only when training to limit memory errors
        if train:
            while not hyperpartisan_batch.is_full():
                try:
                    _, hyperpartisan_data = next(hyperpartisan_iterator)
                except StopIteration:
                    break
            
                hyperpartisan_batch.add_data(hyperpartisan_data[0], hyperpartisan_data[1].item(), hyperpartisan_data[2], hyperpartisan_data[3].item(), hyperpartisan_data[4])

        hyperpartisan_data = hyperpartisan_batch.pad_and_sort_batch()

        loss, accuracy, batch_targets, batch_predictions = iterate_hyperpartisan(
            joint_model=joint_model,
            optimizer=optimizer,
            criterion=criterion,
            hyperpartisan_data=hyperpartisan_data,
            device=device,
            step=step,
            epoch=epoch,
            gradient_save_path=gradient_save_path,
            train=train)

        running_loss += loss
        running_accuracy += accuracy
        all_targets += batch_targets
        all_predictions += batch_predictions

    final_loss = running_loss / batches_counter
    final_accuracy = running_accuracy / batches_counter

    return final_loss, final_accuracy, all_targets, all_predictions


def iterate_metaphor(
        joint_model: JointModel,
        optimizer: Optimizer,
        criterion: Module,
        metaphor_data,
        device: torch.device,
        epoch: int,
        step: int,
        gradient_save_path: str,
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

    loss = criterion(unpadded_predictions, unpadded_targets)

    if train:
        loss.backward()

        if step == 0:
           utils_helper.plot_grad_flow(joint_model.named_parameters(), gradient_save_path, epoch, "metaphor")

        optimizer.step()

    return unpadded_targets.long().tolist(), unpadded_predictions.round().long().tolist()


def forward_full_metaphor(
        joint_model: JointModel,
        optimizer: Optimizer,
        criterion: Module,
        dataloader: DataLoader,
        device: torch.device,
        epoch: int,
        gradient_save_path: str,
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
            epoch=epoch,
            step=step,
            gradient_save_path=gradient_save_path,
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
        hyperpartisan_batch_max_size: int,
        epoch: int,
        gradient_save_path: str,
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

    hyperpartisan_iterator = enumerate(hyperpartisan_iterator)
    metaphor_iterator = enumerate(metaphor_iterator)
    batch_counter = 0
    while True:
        try:
            step, hyperpartisan_data = next(hyperpartisan_iterator)
            _, metaphor_batch = next(metaphor_iterator)
        except StopIteration:
            break

        batch_counter += 1

        assert hyperpartisan_data != None
        assert metaphor_batch != None

        print(f'Step {step+1}/{total_length}                  \r', end='')

        if joint_metaphors_first:
            _, _ = iterate_metaphor(
                joint_model=joint_model,
                optimizer=optimizer,
                criterion=metaphor_criterion,
                metaphor_data=metaphor_batch,
                device=device,
                epoch=epoch,
                step=step,
                gradient_save_path=gradient_save_path,
                train=train)


        hyperpartisan_batch = HyperpartisanBatch(hyperpartisan_batch_max_size)
        hyperpartisan_batch.add_data(hyperpartisan_data[0], hyperpartisan_data[1].item(), hyperpartisan_data[2], hyperpartisan_data[3].item(), hyperpartisan_data[4])
        
        # get bigger than 1 batches only when training to limit memory errors
        if train:
            while not hyperpartisan_batch.is_full():
                try:
                    _, hyperpartisan_data = next(hyperpartisan_iterator)
                except StopIteration:
                    break

                hyperpartisan_batch.add_data(hyperpartisan_data[0], hyperpartisan_data[1].item(), hyperpartisan_data[2], hyperpartisan_data[3].item(), hyperpartisan_data[4])

        hyperpartisan_data = hyperpartisan_batch.pad_and_sort_batch()
        
        loss, accuracy, batch_targets, batch_predictions = iterate_hyperpartisan(
            joint_model=joint_model,
            optimizer=optimizer,
            criterion=hyperpartisan_criterion,
            hyperpartisan_data=hyperpartisan_data,
            device=device,
            step=step,
            epoch=epoch,
            gradient_save_path=gradient_save_path,
            train=train,
            loss_suppress_factor=loss_suppress_factor)

        running_hyperpartisan_loss += loss
        running_hyperpartisan_accuracy += accuracy
        
        all_hyperpartisan_targets.extend(batch_targets)
        all_hyperpartisan_predictions.extend(batch_predictions)

        if not joint_metaphors_first:
            _, _ = iterate_metaphor(
                joint_model=joint_model,
                optimizer=optimizer,
                criterion=metaphor_criterion,
                metaphor_data=metaphor_batch,
                device=device,
                epoch=epoch,
                step=step,
                gradient_save_path=gradient_save_path,
                train=train)

    final_loss = running_hyperpartisan_loss / batch_counter
    final_accuracy = running_hyperpartisan_accuracy / batch_counter

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
        epoch: int,
        gradient_save_path: str,
        hyperpartisan_batch_max_size: int):

    joint_model.train()

    loss_train, accuracy_train, _, _ = forward_full_hyperpartisan(joint_model=joint_model,
                                                                  optimizer=optimizer,
                                                                  criterion=hyperpartisan_criterion,
                                                                  dataloader=hyperpartisan_train_dataloader,
                                                                  device=device,
                                                                  hyperpartisan_batch_max_size=hyperpartisan_batch_max_size,
                                                                  epoch=epoch,
                                                                  gradient_save_path=gradient_save_path,
                                                                  train=True)

    joint_model.eval()

    loss_valid, accuracy_valid, valid_targets, valid_predictions = forward_full_hyperpartisan(joint_model=joint_model,
                                                                                              optimizer=None,
                                                                                              criterion=hyperpartisan_criterion,
                                                                                              dataloader=hyperpartisan_validation_dataloader,
                                                                                              hyperpartisan_batch_max_size=hyperpartisan_batch_max_size,
                                                                                              epoch=epoch,
                                                                                              gradient_save_path=gradient_save_path,
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

    return f1, accuracy_valid, precision, recall


def train_and_eval_metaphor(
        joint_model: JointModel,
        optimizer: Optimizer,
        metaphor_criterion: Module,
        metaphor_train_dataloader: DataLoader,
        metaphor_validation_dataloader: DataLoader,
        device: torch.device,
        best_f1_score: int,
        epoch: int,
        gradient_save_path: str):

    joint_model.train()
    forward_full_metaphor(
        joint_model=joint_model,
        optimizer=optimizer,
        criterion=metaphor_criterion,
        dataloader=metaphor_train_dataloader,
        device=device,
        epoch=epoch,
        gradient_save_path=gradient_save_path,
        train=True)

    joint_model.eval()
    val_targets, val_predictions = forward_full_metaphor(
        joint_model=joint_model,
        optimizer=None,
        criterion=metaphor_criterion,
        dataloader=metaphor_validation_dataloader,
        epoch=epoch,
        gradient_save_path=gradient_save_path,
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
        best_hyperpartisan_f1_score: bool,
        hyperpartisan_batch_max_size: int,
        gradient_save_path: str):

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
        hyperpartisan_batch_max_size=hyperpartisan_batch_max_size,
        epoch=epoch,
        gradient_save_path=gradient_save_path,
        train=True)

    # Evaluate

    joint_model.eval()

    # Metaphor
    
    val_targets, val_predictions = forward_full_metaphor(
        joint_model=joint_model,
        optimizer=None,
        criterion=metaphor_criterion,
        dataloader=metaphor_validation_dataloader,
        epoch=epoch,
        gradient_save_path=gradient_save_path,
        device=device)

    metaphor_f1, _, _ = utils_helper.calculate_metrics(val_targets, val_predictions)

    print_metaphor_stats(metaphor_f1, epoch, False)

    # Hyperpartisan

    valid_loss, valid_accuracy, valid_targets, valid_predictions = forward_full_hyperpartisan(
        joint_model=joint_model,
        optimizer=None,
        criterion=hyperpartisan_criterion,
        dataloader=hyperpartisan_validation_dataloader,
        hyperpartisan_batch_max_size=hyperpartisan_batch_max_size,
        epoch=epoch,
        gradient_save_path=gradient_save_path,
        device=device)

    hyperpartisan_f1, hyperpartisan_precision, hyperpartisan_recall = utils_helper.calculate_metrics(valid_targets, valid_predictions)

    # Log results

    log_metrics(
        summary_writer=summary_writer,
        global_step=epoch,
        loss_train=train_loss,
        accuracy_train=train_accuracy,
        valid_loss=valid_loss,
        valid_accuracy=valid_accuracy,
        valid_precision=hyperpartisan_precision,
        valid_recall=hyperpartisan_recall,
        valid_f1=hyperpartisan_f1)

    print_hyperpartisan_stats(
        train_loss=train_loss,
        valid_loss=valid_loss,
        train_accuracy=train_accuracy,
        valid_accuracy=valid_accuracy,
        valid_precision=hyperpartisan_precision,
        valid_recall=hyperpartisan_recall,
        valid_f1=hyperpartisan_f1,
        epoch=epoch,
        new_best_score=(best_hyperpartisan_f1_score < hyperpartisan_f1))

    return hyperpartisan_f1, valid_accuracy, hyperpartisan_precision, hyperpartisan_recall, metaphor_f1

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

def save_best_result(arg_parser: ArgumentParserHelper, metrics: dict):
    titles = ['time']
    values = [str(datetime.now().time().replace(microsecond=0))]

    for key, value in metrics.items():
        titles.append(str(key))
        values.append(str(value))

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
    if argument_parser.concat_glove:
        glove_vectors = utils_helper.load_glove_vectors(
            argument_parser.vector_file_name, argument_parser.vector_cache_dir, argument_parser.glove_size)
    else:
        glove_vectors = None

    # Define the model, the optimizer and the loss module
    joint_model, optimizer, start_epoch = initialize_model(argument_parser, device)

    summary_writer = SummaryWriter(
        f'runs/exp-{argument_parser.mode}-odr_{argument_parser.output_dropout_rate}-lr_{argument_parser.learning_rate}-wd_{argument_parser.weight_decay}-lsf_{argument_parser.loss_suppress_factor}')
    
    # Load hyperpartisan data
    if TrainingMode.contains_hyperpartisan(argument_parser.mode):
        hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader, hyperpartisan_pos_weight = create_hyperpartisan_loaders(
            argument_parser, glove_vectors)

    # Load metaphor data
    if TrainingMode.contains_metaphor(argument_parser.mode):
        metaphor_train_dataloader, metaphor_validation_dataloader, metaphor_pos_weight = create_metaphor_loaders(
            argument_parser, glove_vectors)

    if argument_parser.class_weights:
        if TrainingMode.contains_metaphor(argument_parser.mode):
            print("Setting up metaphor loss with positive weight {}".format(metaphor_pos_weight))
            metaphor_criterion = nn.BCEWithLogitsLoss(pos_weight = torch.ones([1]) * metaphor_pos_weight).to(device)
        if TrainingMode.contains_hyperpartisan(argument_parser.mode):
            print("Setting up hyperpartisan loss with positive weight {}".format(hyperpartisan_pos_weight))
            hyperpartisan_criterion = nn.BCEWithLogitsLoss(pos_weight = torch.ones([1]) * hyperpartisan_pos_weight).to(device)
    else:
        metaphor_criterion = nn.BCEWithLogitsLoss()
        hyperpartisan_criterion = nn.BCEWithLogitsLoss()

    tic = time.process_time()

    best_f1 = .0
    best_metrics = {}
    stop_counter = 0

    # folder for storing gradient flow graphs
    if "gradients" not in os.listdir():
        os.mkdir("gradients")
    gradient_save_path = "odr_{}-lr_{}-wd_{}-size_{}/".format(argument_parser.output_dropout_rate, argument_parser.learning_rate, argument_parser.weight_decay, argument_parser.doc_encoder_hidden_dim)
    try:
        os.mkdir("gradients/" + gradient_save_path)
    except:
        pass

    for epoch in range(start_epoch, argument_parser.max_epochs + 1):
        # Joint mode by batches
        if argument_parser.mode == TrainingMode.JointBatches:
            f1, hyp_accuracy, hyp_precision, hyp_recall, metaphor_f1 = train_and_eval_joint(
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
                best_hyperpartisan_f1_score=best_f1,
                hyperpartisan_batch_max_size=argument_parser.hyperpartisan_batch_max_size,
                gradient_save_path=gradient_save_path)

            metrics = {'hyp_f1': f1,
                       'hyp_accuracy': hyp_accuracy,
                       'hyp_precision': hyp_precision,
                       'hyp_recall': hyp_recall,
                       'metaphor_f1': metaphor_f1}

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
                    epoch=epoch,
                    gradient_save_path=gradient_save_path)
                metrics = {'hyp_f1': 'NA',
                           'hyp_accuracy': 'NA',
                           'hyp_precision': 'NA',
                           'hyp_recall': 'NA',
                           'metaphor_f1': f1}

            if TrainingMode.contains_hyperpartisan(argument_parser.mode):
                # Complete one epoch of hyperpartisan
                f1, hyp_accuracy, hyp_precision, hyp_recall = train_and_eval_hyperpartisan(
                    joint_model=joint_model,
                    optimizer=optimizer,
                    hyperpartisan_criterion=hyperpartisan_criterion,
                    hyperpartisan_train_dataloader=hyperpartisan_train_dataloader,
                    hyperpartisan_validation_dataloader=hyperpartisan_validation_dataloader,
                    device=device,
                    best_f1_score=best_f1,
                    epoch=epoch,
                    summary_writer=summary_writer,
                    hyperpartisan_batch_max_size=argument_parser.hyperpartisan_batch_max_size,
                    gradient_save_path=gradient_save_path)
                metrics = {'hyp_f1': f1,
                           'hyp_accuracy': hyp_accuracy,
                           'hyp_precision': hyp_precision,
                           'hyp_recall': hyp_recall,
                           'metaphor_f1': 'NA'}

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
                    epoch=epoch,
                    gradient_save_path=gradient_save_path)
                metrics = {'hyp_f1': 'NA',
                           'hyp_accuracy': 'NA',
                           'hyp_precision': 'NA',
                           'hyp_recall': 'NA',
                           'metaphor_f1': f1}

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = metrics
            cache_model(joint_model=joint_model,
                        optimizer=optimizer,
                        epoch=epoch)

    print("[{}] Training completed in {:.2f} minutes".format(datetime.now().time().replace(microsecond=0),
                                                             (time.process_time() - tic) / 60))

    save_best_result(argument_parser, best_metrics)

    summary_writer.close()


if __name__ == '__main__':
    argument_parser = ArgumentParserHelper()
    argument_parser.parse_arguments()
    argument_parser.print_unique_arguments()

    train_model(argument_parser)
