import argparse
import os

from statsmodels.stats.contingency_tables import mcnemar

import torch
from torch.utils.data import DataLoader

import numpy as np

import sys

sys.path.append('..')

from model.JointModel import JointModel
from helpers.utils_helper import UtilsHelper
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan
from helpers.hyperpartisan_loader import HyperpartisanLoader
from datasets.hyperpartisan_dataset import HyperpartisanDataset

from constants import Constants
from enums.training_mode import TrainingMode

utils_helper = UtilsHelper()


def load_model_state(model, model_checkpoint_path):
    if not os.path.isfile(model_checkpoint_path):
        raise Exception('Model checkpoint path is invalid')

    checkpoint = torch.load(model_checkpoint_path)
    if not checkpoint['model_state_dict']:
        raise Exception('Model state dictionary checkpoint not found')

    model.load_state_dict(checkpoint['model_state_dict'])


def initialize_models(
        hyperpartisan_model_checkpoint_path: str,
        joint_model_checkpoint_path: str,
        device: torch.device,
        glove_vectors_dim: int):
    print('Loading model state...\r', end='')

    total_embedding_dim = Constants.DEFAULT_ELMO_EMBEDDING_DIMENSION + glove_vectors_dim

    hyperpartisan_model = JointModel(embedding_dim=total_embedding_dim,
                                     hidden_dim=Constants.DEFAULT_HIDDEN_DIMENSION,
                                     num_layers=Constants.DEFAULT_NUM_LAYERS,
                                     sent_encoder_dropout_rate=Constants.DEFAULT_SENTENCE_ENCODER_DROPOUT_RATE,
                                     doc_encoder_dropout_rate=Constants.DEFAULT_DOCUMENT_ENCODER_DROPOUT_RATE,
                                     output_dropout_rate=Constants.DEFAULT_OUTPUT_ENCODER_DROPOUT_RATE,
                                     device=device).to(device)

    joint_model = JointModel(embedding_dim=total_embedding_dim,
                             hidden_dim=Constants.DEFAULT_HIDDEN_DIMENSION,
                             num_layers=Constants.DEFAULT_NUM_LAYERS,
                             sent_encoder_dropout_rate=Constants.DEFAULT_SENTENCE_ENCODER_DROPOUT_RATE,
                             doc_encoder_dropout_rate=Constants.DEFAULT_DOCUMENT_ENCODER_DROPOUT_RATE,
                             output_dropout_rate=Constants.DEFAULT_OUTPUT_ENCODER_DROPOUT_RATE,
                             device=device).to(device)

    load_model_state(hyperpartisan_model, hyperpartisan_model_checkpoint_path)
    load_model_state(joint_model, joint_model_checkpoint_path)

    print('Loading model state...Done')

    return hyperpartisan_model, joint_model


def get_interesting_articles(targets, hyp_pred, joint_pred):
    assert len(targets) == len(hyp_pred)
    assert len(targets) == len(joint_pred)

    interesting_articles = np.where(targets == 1 and hyp_pred == 0 and joint_pred == 1)[0]

    print('[', end='')
    for i in range(len(targets)):
        if targets[i] == 1 and hyp_pred[i] == 0 and joint_pred[i] == 1:
            print(f'{i},', end='')
            
            # print(f'{targets[i]} - {hyp_pred[i]} - {joint_pred[i]}')
    print(']')

    return interesting_articles


def forward_full_hyperpartisan(
        joint_model: JointModel,
        dataloader: DataLoader,
        device: torch.device):
    all_targets = []
    all_predictions = []

    running_accuracy = 0

    total_length = len(dataloader)
    for step, hyperpartisan_data in enumerate(dataloader):
        print(f'Step {step+1}/{total_length}                  \r', end='')

        batch_inputs = hyperpartisan_data[0].to(device)
        batch_targets = hyperpartisan_data[1].to(device)
        batch_recover_idx = hyperpartisan_data[2].to(device)
        batch_num_sent = hyperpartisan_data[3].to(device)
        batch_sent_lengths = hyperpartisan_data[4].to(device)
        batch_feat = hyperpartisan_data[5].to(device)

        batch_predictions = joint_model.forward(batch_inputs, (batch_recover_idx,
                                                               batch_num_sent, batch_sent_lengths, batch_feat),
                                                task=TrainingMode.Hyperpartisan)

        accuracy = utils_helper.calculate_accuracy(batch_predictions, batch_targets)

        running_accuracy += accuracy
        all_targets += batch_targets.long().tolist()
        all_predictions += batch_predictions.round().long().tolist()

    final_accuracy = running_accuracy / (step + 1)

    return final_accuracy, all_targets, all_predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperpartisan_model_checkpoint', type=str, required=True,
                        help='Path to load the hyperpartisan model')
    parser.add_argument('--joint_model_checkpoint', type=str, required=True,
                        help='Path to load the joint model')
    parser.add_argument('--vector_file_name', type=str, required=True,
                        help='File in which vectors are saved')
    parser.add_argument('--vector_cache_dir', type=str, default=Constants.DEFAULT_VECTOR_CACHE_DIR,
                        help='Directory where vectors would be cached')
    parser.add_argument('--glove_size', type=int,
                        help='Number of GloVe vectors to load initially')
    parser.add_argument('--hyperpartisan_dataset_folder', type=str, required=True,
                        help='Path to the hyperpartisan dataset')
    parser.add_argument('--hyperpartisan_batch_size', type=int, default=Constants.DEFAULT_HYPERPARTISAN_BATCH_SIZE,
                        help='Batch size for training on the hyperpartisan dataset')
    parser.add_argument('--metaphor_batch_size', type=int, default=Constants.DEFAULT_METAPHOR_BATCH_SIZE,
                        help='Batch size for training on the metaphor dataset')
    parser.add_argument('--deterministic', type=int, required=True,
                        help='The seed to be used when running deterministically. If nothing is passed, the program run will be stochastic')
    parser.add_argument('--alpha_value', type=int, default=0.05,
                        help='The alpha value which will be used to statistically test the significant difference')

    config = parser.parse_args()

    utils_helper.initialize_deterministic_mode(config.deterministic)

    device = torch.device("cuda:0")  # if torch.cuda.is_available() else "cpu")

    glove_vectors = utils_helper.load_glove_vectors(
        config.vector_file_name, config.vector_cache_dir, config.glove_size)

    hyperpartisan_model, joint_model = initialize_models(config.hyperpartisan_model_checkpoint,
                                                         config.joint_model_checkpoint, device, glove_vectors.dim)

    _, hyperpartisan_validation_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
        hyperpartisan_dataset_folder=config.hyperpartisan_dataset_folder,
        glove_vectors=glove_vectors,
        lowercase_sentences=False,
        articles_max_length=Constants.DEFAULT_HYPERPARTISAN_MAX_LENGTH,
        load_train=False)

    _, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
        validation_dataset=hyperpartisan_validation_dataset,
        batch_size=config.hyperpartisan_batch_size,
        shuffle=False)

    hyperpartisan_valid_accuracy, hyperpartisan_valid_targets, hyperpartisan_valid_predictions = forward_full_hyperpartisan(
        joint_model=hyperpartisan_model,
        dataloader=hyperpartisan_validation_dataloader,
        device=device)

    hyperpartisan_f1, _, _ = utils_helper.calculate_metrics(hyperpartisan_valid_targets,
                                                            hyperpartisan_valid_predictions)

    print(f'Hyperpartisan F1 score: {hyperpartisan_f1}')

    joint_valid_accuracy, joint_valid_targets, joint_valid_predictions = forward_full_hyperpartisan(
        joint_model=joint_model,
        dataloader=hyperpartisan_validation_dataloader,
        device=device)

    interesting_articles = get_interesting_articles(
        hyperpartisan_valid_targets,
        hyperpartisan_valid_predictions,
        joint_valid_predictions)

    interesting_articles += 2

    print("Interesting articles are: {}".format(interesting_articles))
