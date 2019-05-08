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

from sklearn import metrics
from datetime import datetime
import time

def get_accuracy(pred_scores, targets):
    """
    Calculate the accuracy
    :param pred_scores: Scores obtained by the model
    :param targets: Ground truth targets
    :return: Accuracy
    """
    binary = len(pred_scores.shape) == 1

    if binary:
        pred = pred_scores > 0.5
        accuracy = torch.sum(pred == targets.byte()).float() / pred.shape[0]
    else:
        pred = torch.argmax(pred_scores, dim = 1)
        accuracy = torch.mean((pred == targets).float())
    
    return accuracy


def train_model(config):
    """
    Train the multi-task classifier model
    :param config: Dictionary specifying the model configuration
    :return: None
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
    hyperpartisan_criterion = nn.CrossEntropyLoss()

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
    if config.train_hyperpartisan:

        hyperpartisan_train_dataset, hyperpartisan_validation_dataset, hyperpartisan_test_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
            hyperpartisan_dataset_folder=config.hyperpartisan_dataset_folder,
            word_vector=glove_vectors)

        hyperpartisan_train_dataloader, hyperpartisan_validation_dataloader, hyperpartisan_test_dataloader = DataHelperHyperpartisan.create_dataloaders(
            train_dataset=hyperpartisan_train_dataset,
            validation_dataset=hyperpartisan_validation_dataset,
            test_dataset=hyperpartisan_test_dataset,
            batch_size=config.batch_size,
            shuffle=True)

    # Load metaphor data
    if config.train_metaphor:

        metaphor_train_dataset, metaphor_validation_dataset, metaphor_test_dataset = MetaphorLoader.get_metaphor_datasets(
            metaphor_dataset_folder=config.metaphor_dataset_folder,
            word_vector=glove_vectors)

        metaphor_train_dataloader, metaphor_validation_dataloader, metaphor_test_dataloader = DataHelper.create_dataloaders(
            train_dataset=metaphor_train_dataset,
            validation_dataset=metaphor_validation_dataset,
            test_dataset=metaphor_test_dataset,
            batch_size=config.batch_size,
            shuffle=True)

        f1_validation_scores = []

    tic = time.clock()

    for epoch in range(start_epoch, config.max_epochs + 1):

        if config.train_metaphor:

            model.train()
            for _, (m_batch_inputs, m_batch_targets, m_batch_lengths) in enumerate(metaphor_train_dataloader):
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

            model.eval()

            val_targets = []
            val_predictions = []
            for _, (m_val_batch_inputs, m_val_batch_targets, m_val_batch_lengths) in enumerate(metaphor_validation_dataloader):
                m_val_batch_inputs = m_val_batch_inputs.to(device)
                m_val_batch_targets = m_val_batch_targets.to(device).view(-1).float()
                m_val_batch_lengths = m_val_batch_lengths.to(device)

                pred = model(m_val_batch_inputs, m_val_batch_lengths, task='metaphor')
                unpad_targets = m_val_batch_targets[m_val_batch_targets != -1]
                unpad_pred = pred.view(-1)[m_val_batch_targets != -1]
            
                val_targets.extend(unpad_targets.tolist())
                val_predictions.extend(unpad_pred.round().tolist())
        
            current_f1_score = metrics.f1_score(val_targets, val_predictions, average="binary")
            f1_validation_scores.append(current_f1_score)
            print(f'f1 score: {current_f1_score}')

        if config.train_hyperpartisan:

            running_loss, running_accu = 0, 0
            model.train()

            for step, (h_batch_inputs, h_batch_targets, h_batch_recover_idx, h_batch_num_sent) in enumerate(hyperpartisan_train_dataloader):
                h_batch_inputs = h_batch_inputs.to(device)
                h_batch_targets = h_batch_targets.to(device)
                h_batch_recover_idx = h_batch_recover_idx.to(device)
                h_batch_num_sent = h_batch_num_sent.to(device)

                optimizer.zero_grad()
                logits = model(h_batch_inputs, (h_batch_recover_idx, h_batch_num_sent), task='hyperpartisan')

                loss = hyperpartisan_criterion(logits, h_batch_targets)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

                accuracy = get_accuracy(logits, h_batch_targets)
                running_accu += accuracy.item()

            loss_train = running_loss / (step + 1)
            accu_train = running_accu / (step + 1)

            running_loss, running_accu = 0, 0
            model.eval()

            for step, (h_batch_inputs, h_batch_targets, h_batch_recover_idx, h_batch_num_sent) in enumerate(hyperpartisan_validation_dataloader):
                h_batch_inputs = h_batch_inputs.to(device)
                h_batch_targets = h_batch_targets.to(device)
                h_batch_recover_idx = h_batch_recover_idx.to(device)
                h_batch_num_sent = h_batch_num_sent.to(device)

                with torch.no_grad():

                    logits = model(h_batch_inputs, (h_batch_recover_idx, h_batch_num_sent), task='hyperpartisan')

                    loss = hyperpartisan_criterion(logits, h_batch_targets)
                    accu = get_accuracy(logits, h_batch_targets)

                    running_loss += loss.item()
                    running_accu += accu.item()

            loss_valid = running_loss / (step + 1)
            accu_valid = running_accu / (step + 1)

            print("[{}] epoch {} || LOSS: train = {:.4f}, valid = {:.4f} || ACCURACY: train = {:.4f}, valid = {:.4f}".format(
                datetime.now().time().replace(microsecond = 0), epoch, loss_train, loss_valid, accu_train, accu_valid))

    print("[{}] Training completed in {:.2f} minutes".format(datetime.now().time().replace(microsecond = 0), (time.clock() - tic)/60))   

    running_loss, running_accu = 0, 0
    model.eval()

    for step, (h_batch_inputs, h_batch_targets, h_batch_recover_idx, h_batch_num_sent) in enumerate(hyperpartisan_test_dataloader):
        h_batch_inputs = h_batch_inputs.to(device)
        h_batch_targets = h_batch_targets.to(device)
        h_batch_recover_idx = h_batch_recover_idx.to(device)
        h_batch_num_sent = h_batch_num_sent.to(device)

        with torch.no_grad():

            logits = model(h_batch_inputs, (h_batch_recover_idx, h_batch_num_sent), task='hyperpartisan')

            loss = hyperpartisan_criterion(logits, h_batch_targets)
            accu = get_accuracy(logits, h_batch_targets)

            running_loss += loss.item()
            running_accu += accu.item()

    loss_test = running_loss / (step + 1)
    accu_test = running_accu / (step + 1)

    print("[{}] Performance on test set: Loss = {:.4f} Accuracy = {:.4f}".format(datetime.now().time().replace(microsecond = 0), loss_test, accu_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to save/load the checkpoint data')
    parser.add_argument('--data_path', type=str,
                        help='Path where data is saved')
    parser.add_argument('--vector_file_name', type=str, required=True,
                        help='File in which vectors are saved')
    parser.add_argument('--vector_cache_dir', type=str, default = "C:/Users/ioann/Datasets/vector_cache/",
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
    parser.add_argument('--metaphor_dataset_folder', type=str, default = "C:/Users/ioann/Datasets/metaphor_in_context_master/data/VUAsequence/",
                        help='Path to the metaphor dataset')
    parser.add_argument('--hyperpartisan_dataset_folder', type=str,
                        help='Path to the hyperpartisan dataset')
    parser.add_argument('--train_metaphor', type=bool, default=False,
                        help="Whether to train on the metaphor task")
    parser.add_argument('--train_hyperpartisan', type=bool, default=True,
                        help="Whether to train on the hyperpartisan task")

    config = parser.parse_args()

    train_model(config)
