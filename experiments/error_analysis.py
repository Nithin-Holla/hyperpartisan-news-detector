import argparse
import os

import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import pearsonr, spearmanr
import sys

from batches.hyperpartisan_batch import HyperpartisanBatch

sys.path.append('..')

from enums.elmo_model import ELMoModel
from model.JointModel import JointModel
from model.Ensemble import Ensemble
from helpers.utils_helper import UtilsHelper
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan
from helpers.hyperpartisan_loader import HyperpartisanLoader

from constants import Constants
from enums.training_mode import TrainingMode

utils_helper = UtilsHelper()


def load_model_state(model, model_checkpoint_path, device):
    if not os.path.isfile(model_checkpoint_path):
        raise Exception('Model checkpoint path is invalid')

    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    if not checkpoint['model_state_dict']:
        raise Exception('Model state dictionary checkpoint not found')

    model.load_state_dict(checkpoint['model_state_dict'])


def initialize_models(
        hyperpartisan_model_checkpoint_path: str,
        joint_model_checkpoint_path: str,
        device: torch.device,
        elmo_model: ELMoModel,
        concat_glove: bool,
        model_type: str):
    print('Loading model state...\r', end='')

    if elmo_model == ELMoModel.Original:
        total_embedding_dim = Constants.ORIGINAL_ELMO_EMBEDDING_DIMENSION
    elif elmo_model == ELMoModel.Small:
        total_embedding_dim = Constants.SMALL_ELMO_EMBEDDING_DIMENSION

    if concat_glove:
        total_embedding_dim += Constants.GLOVE_EMBEDDING_DIMENSION

    if model_type == "ensemble":

        assert os.path.isdir(joint_model_checkpoint_path)
        assert os.path.isdir(hyperpartisan_model_checkpoint_path)

        hyperpartisan_model = Ensemble(path_to_models=hyperpartisan_model_checkpoint_path,
                                       sent_encoder_hidden_dim=Constants.DEFAULT_HIDDEN_DIMENSION,
                                       doc_encoder_hidden_dim=Constants.DEFAULT_DOC_ENCODER_DIM,
                                       num_layers=Constants.DEFAULT_NUM_LAYERS,
                                       skip_connection=Constants.DEFAULT_SKIP_CONNECTION,
                                       include_article_features=Constants.DEFAULT_INCLUDE_ARTICLE_FEATURES,
                                       document_encoder_model=Constants.DEFAULT_DOCUMENT_ENCODER_MODEL,
                                       pre_attention_layer=Constants.DEFAULT_PRE_ATTENTION_LAYER,
                                       total_embedding_dim=total_embedding_dim,
                                       device=device
                                       )

        joint_model = Ensemble(path_to_models=joint_model_checkpoint_path,
                               sent_encoder_hidden_dim=Constants.DEFAULT_HIDDEN_DIMENSION,
                               doc_encoder_hidden_dim=Constants.DEFAULT_DOC_ENCODER_DIM,
                               num_layers=Constants.DEFAULT_NUM_LAYERS,
                               skip_connection=Constants.DEFAULT_SKIP_CONNECTION,
                               include_article_features=Constants.DEFAULT_INCLUDE_ARTICLE_FEATURES,
                               document_encoder_model=Constants.DEFAULT_DOCUMENT_ENCODER_MODEL,
                               pre_attention_layer=Constants.DEFAULT_PRE_ATTENTION_LAYER,
                               total_embedding_dim=total_embedding_dim,
                               device=device
                               )

    else:

        assert os.path.isfile(joint_model_checkpoint_path)
        assert os.path.isfile(hyperpartisan_model_checkpoint_path)

        hyperpartisan_model = JointModel(embedding_dim=total_embedding_dim,
                                         sent_encoder_hidden_dim=Constants.DEFAULT_HIDDEN_DIMENSION,
                                         doc_encoder_hidden_dim=Constants.DEFAULT_DOC_ENCODER_DIM,
                                         num_layers=Constants.DEFAULT_NUM_LAYERS,
                                         sent_encoder_dropout_rate=0.,
                                         doc_encoder_dropout_rate=0.,
                                         output_dropout_rate=0.,
                                         device=device,
                                         skip_connection=Constants.DEFAULT_SKIP_CONNECTION,
                                         include_article_features=Constants.DEFAULT_INCLUDE_ARTICLE_FEATURES,
                                         doc_encoder_model=Constants.DEFAULT_DOCUMENT_ENCODER_MODEL,
                                         pre_attn_layer=Constants.DEFAULT_PRE_ATTENTION_LAYER
                                         ).to(device)

        joint_model = JointModel(embedding_dim=total_embedding_dim,
                                 sent_encoder_hidden_dim=Constants.DEFAULT_HIDDEN_DIMENSION,
                                 doc_encoder_hidden_dim=Constants.DEFAULT_DOC_ENCODER_DIM,
                                 num_layers=Constants.DEFAULT_NUM_LAYERS,
                                 sent_encoder_dropout_rate=0.,
                                 doc_encoder_dropout_rate=0.,
                                 output_dropout_rate=0.,
                                 device=device,
                                 skip_connection=Constants.DEFAULT_SKIP_CONNECTION,
                                 include_article_features=Constants.DEFAULT_INCLUDE_ARTICLE_FEATURES,
                                 doc_encoder_model=Constants.DEFAULT_DOCUMENT_ENCODER_MODEL,
                                 pre_attn_layer=Constants.DEFAULT_PRE_ATTENTION_LAYER
                                 ).to(device)

        load_model_state(hyperpartisan_model, hyperpartisan_model_checkpoint_path, device)
        load_model_state(joint_model, joint_model_checkpoint_path, device)

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


def get_predictions(
        model,
        dataloader: DataLoader,
        device: torch.device):
    hyp_targets = []
    hyp_predictions = []
    metaphor_predictions = []

    total_length = len(dataloader)

    model.eval()

    n_metaphors = 0
    n_words = 0

    with torch.no_grad():
        for step, hyperpartisan_data in enumerate(dataloader):
            print(f'Step {step+1}/{total_length}                  \r', end='')

            hyperpartisan_batch = HyperpartisanBatch(10000)  # just something big
            hyperpartisan_batch.add_data(*hyperpartisan_data[:-1])  # exclude last element (id)
            hyperpartisan_data = hyperpartisan_batch.pad_and_sort_batch()

            batch_inputs = hyperpartisan_data[0].to(device)
            batch_targets = hyperpartisan_data[1].to(device)
            batch_recover_idx = hyperpartisan_data[2].to(device)
            batch_num_sent = hyperpartisan_data[3].to(device)
            batch_sent_lengths = hyperpartisan_data[4].to(device)
            batch_feat = hyperpartisan_data[5].to(device)

            batch_hyp_predictions = model.forward(batch_inputs, (batch_recover_idx,
                                                                 batch_num_sent, batch_sent_lengths, batch_feat),
                                                  task=TrainingMode.Hyperpartisan)
            hyp_targets.append(batch_targets.long().item())
            hyp_predictions.append(batch_hyp_predictions.item())

            batch_metaphor_predictions = model.forward(batch_inputs, batch_sent_lengths, task=TrainingMode.Metaphor)
            batch_metaphor_predictions = batch_metaphor_predictions.round().long()
            batch_metaphor_predictions = torch.sum(batch_metaphor_predictions, dim=1)

            n_metaphors += torch.sum(batch_metaphor_predictions).item()
            n_words += torch.sum(batch_sent_lengths).item()

            batch_metaphor_predictions = batch_metaphor_predictions > 0
            batch_metaphor_predictions = torch.mean(batch_metaphor_predictions.float())
            metaphor_predictions.append(batch_metaphor_predictions.item())

    print('No. of metaphors = {}'.format(n_metaphors))
    print('No. of words = {}'.format(n_words))
    print('Percentage of metaphor words = {}'.format(n_metaphors / n_words))

    return np.array(hyp_targets), np.array(hyp_predictions), np.array(metaphor_predictions)


def correlation_analysis(model, hyperpartisan_validation_dataloader, device):
    hyp_targets, hyp_predictions, metaphor_predictions = get_predictions(model, hyperpartisan_validation_dataloader,
                                                                         device)
    hyp_articles = np.where(hyp_targets == 1)[0]
    non_hyp_articles = np.where(hyp_targets == 0)[0]
    metaphor_and_hyp = np.sum(metaphor_predictions[hyp_articles]) / len(hyp_articles)
    metaphor_and_non_hyp = np.sum(metaphor_predictions[non_hyp_articles]) / len(non_hyp_articles)
    print('Percentage of hyperpartisan articles that are predicted to contain metaphorical sentences: {}'.format(
        metaphor_and_hyp))
    print('Percentage of non-hyperpartisan articles that are predicted to contain metaphorical sentences: {}'.format(
        metaphor_and_non_hyp))

    plt.scatter(metaphor_predictions[hyp_articles], hyp_predictions[hyp_articles], c='red')
    plt.scatter(metaphor_predictions[non_hyp_articles], hyp_predictions[non_hyp_articles], c='blue')
    plt.xlabel('Percentage of metaphorical sentences')
    plt.ylabel('Hyperpartisan score')
    plt.savefig('correlation.png')

    pearson_corr = pearsonr(metaphor_predictions, hyp_predictions)
    spearman_corr = spearmanr(metaphor_predictions, hyp_predictions)
    print('Pearson correlation = {}'.format(pearson_corr))
    print('Spearman correlation = {}'.format(spearman_corr))


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
    parser.add_argument('--deterministic', type=int,
                        help='The seed to be used when running deterministically. If nothing is passed, the program run will be stochastic')
    parser.add_argument('--alpha_value', type=int, default=0.05,
                        help='The alpha value which will be used to statistically test the significant difference')
    parser.add_argument('--elmo_model', type=ELMoModel, choices=list(ELMoModel), default=ELMoModel.Original,
                        help='ELMo model from which vectors are used')
    parser.add_argument('--concat_glove', action='store_true',
                        help='Whether GloVe vectors have to be concatenated with ELMo vectors for words')
    parser.add_argument('--include_article_features', action='store_true',
                        help='Whether to append handcrafted article features to the hyperpartisan fc layer')
    parser.add_argument('--model_type', type=str, choices=["ensemble", "single"], default="single",
                        help='Whether to use an ensemble of models or a single model instant')

    config = parser.parse_args()

    utils_helper.initialize_deterministic_mode(config.deterministic)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config.concat_glove:
        glove_vectors = utils_helper.load_glove_vectors(
            config.vector_file_name, config.vector_cache_dir, config.glove_size)
    else:
        glove_vectors = None

    hyperpartisan_model, joint_model = initialize_models(config.hyperpartisan_model_checkpoint,
                                                         config.joint_model_checkpoint,
                                                         device,
                                                         config.elmo_model,
                                                         config.concat_glove,
                                                         config.model_type)

    _, hyperpartisan_validation_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
        hyperpartisan_dataset_folder=config.hyperpartisan_dataset_folder,
        concat_glove=config.concat_glove,
        glove_vectors=glove_vectors,
        elmo_model=config.elmo_model,
        lowercase_sentences=False,
        articles_max_length=Constants.DEFAULT_HYPERPARTISAN_MAX_LENGTH,
        load_train=False)

    _, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
        validation_dataset=hyperpartisan_validation_dataset,
        batch_size=config.hyperpartisan_batch_size,
        shuffle=False)

    correlation_analysis(joint_model, hyperpartisan_validation_dataloader, device)

    # interesting_articles = get_interesting_articles(
    #     hyperpartisan_valid_targets,
    #     hyperpartisan_valid_predictions,
    #     joint_valid_predictions)
    #
    # interesting_articles += 2
    #
    # print("Interesting articles are: {}".format(interesting_articles))
