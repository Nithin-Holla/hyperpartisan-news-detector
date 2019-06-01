import torch
import os
import argparse
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from ast import literal_eval

from enums.elmo_model import ELMoModel
from enums.training_mode import TrainingMode
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan
from helpers.hyperpartisan_loader import HyperpartisanLoader
from model.Ensemble import Ensemble
from model.JointModel import JointModel
from helpers.utils_helper import UtilsHelper
from constants import Constants
from batches.hyperpartisan_batch import HyperpartisanBatch

utils_helper = UtilsHelper()


def title_attention_plot(hyperpartisan_validation_dataset, model, device):
    _, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
        validation_dataset=hyperpartisan_validation_dataset,
        test_dataset=None,
        batch_size=4,
        shuffle=False)

    attn_summary = None

    total_length = len(hyperpartisan_validation_dataloader)
    for idx, hyperpartisan_data in enumerate(hyperpartisan_validation_dataloader):
        print(f'{idx}/{total_length}            \r', end='')

        model.eval()

        hyperpartisan_batch = HyperpartisanBatch(10000)
        hyperpartisan_batch.add_data(*hyperpartisan_data[:-1])
        hyperpartisan_data = hyperpartisan_batch.pad_and_sort_batch()

        article_inputs = hyperpartisan_data[0].to(device)
        article_targets = hyperpartisan_data[1].to(device)
        article_recover_idx = hyperpartisan_data[2].to(device)
        article_num_sent = hyperpartisan_data[3].to(device)
        article_sent_lengths = hyperpartisan_data[4].to(device)
        article_feat = hyperpartisan_data[5].to(device)

        pred, _, batch_sent_attn = model.forward(article_inputs, (article_recover_idx, article_num_sent, article_sent_lengths, article_feat),
                                                 task=TrainingMode.Hyperpartisan, return_attention=True)

        decimals = 4
        batch_sent_attn = torch.round(batch_sent_attn * 10**decimals) / (10**decimals)
        article_attn = torch.zeros(batch_sent_attn.shape[0], 2, device=device)
        article_attn[:, 0] = batch_sent_attn[0]
        article_attn[:, 1] = batch_sent_attn[1:].sum().float() / (article_num_sent.float() - 1)
        
        if idx == 0:
            attn_summary = article_attn
        else:
            attn_summary = torch.cat((attn_summary, article_attn), dim=0)

    attn_summary = attn_summary.mean(dim=0, keepdim=True)
    plt.figure()
    sns.heatmap(attn_summary.detach().cpu().numpy(), square=True,
                annot=True, cmap='Blues', cbar=False)
    plt.show()


def get_attention_weights(hyperpartisan_dataset_folder, hyperpartisan_validation_dataloader, model, article_id, device):
    # Load the CSV file
    article_df = pd.read_csv(os.path.join(hyperpartisan_dataset_folder, 'valid_byart.txt'), sep='\t',
                             converters={'title_tokens': literal_eval, 'body_tokens': literal_eval})

    # Load the article text
    article_entry = article_df.iloc[article_id - 2]
    article_txt = [article_entry['title_tokens']] + \
        article_entry['body_tokens']

    # Load one article
    for idx, hyperpartisan_data in enumerate(hyperpartisan_validation_dataloader):
        model.eval()

        hyperpartisan_batch = HyperpartisanBatch(10000)
        hyperpartisan_batch.add_data(*hyperpartisan_data[:-1])
        hyperpartisan_data = hyperpartisan_batch.pad_and_sort_batch()

        article_inputs = hyperpartisan_data[0].to(device)
        # article_targets = hyperpartisan_data[1].to(device)
        article_recover_idx = hyperpartisan_data[2].to(device)
        article_num_sent = hyperpartisan_data[3].to(device)
        article_sent_lengths = hyperpartisan_data[4].to(device)
        article_feat = hyperpartisan_data[5].to(device)

        if idx + 2 == article_id:
            break

    # Obtain prediction and attention weights from the model
    model.eval()
    pred, word_attn, sent_attn = model.forward(article_inputs, (article_recover_idx, article_num_sent, article_sent_lengths, article_feat),
                                               task=TrainingMode.Hyperpartisan, return_attention=True)

    article_sent_lengths = article_sent_lengths[article_recover_idx]
    word_attn = np.around(word_attn.detach().cpu().numpy(), decimals=4)
    sent_attn = np.around(sent_attn.detach().cpu().numpy(), decimals=4)

    return pred, word_attn, sent_attn, article_num_sent, article_sent_lengths, article_txt


def visualize_article_attention(hyperpartisan_dataset_folder, hyperpartisan_validation_dataloader, model, article_id, device):
    """
    Generates a heatmap of the word-level and attention-level attention weights
    :param hyperpartisan_validation_dataloader: Dataloader for hyperpartisan validation set
    :param model: Trained model
    :param article_id: ID of the article, i.e., the line number in valid_byart.txt
    :return: None
    """

    pred, word_attn, sent_attn, article_num_sent, article_sent_lengths, article_txt = get_attention_weights(
        hyperpartisan_dataset_folder, hyperpartisan_validation_dataloader, model, article_id, device)

    # Display results
    print('Hyperpartisan score: {}'.format(pred))
    for sent_idx in range(article_num_sent):
        print('Sentence {} -> Word-level attention: {}'.format(sent_idx + 1,
                                                               word_attn[
                                                                   sent_idx, range(article_sent_lengths[sent_idx])]))
    print('Sentence-level attention: {}'.format(sent_attn))

    # Plot word-level attention weights per sentence
    article_sent_lengths = article_sent_lengths.cpu().numpy()
    max_len = max(article_sent_lengths)
    article_array = np.empty((len(article_txt), max_len), dtype=object)
    for i in range(len(article_txt)):
        article_array[i, range(article_sent_lengths[i])] = article_txt[i]
    article_array[article_array == None] = ''
    plt.figure(figsize=(60, 5))
    plt.axis('off')

    sns.heatmap(word_attn, cmap='Blues', yticklabels=False, xticklabels=False, annot=article_array, fmt='',
                annot_kws={'size': 7}, cbar=False, square=True)
    plt.tight_layout()

    # Plot sentence-level attention weights
    article_txt = np.expand_dims(np.array([' '.join(sent) for sent in article_txt]), axis=1)
    sent_attn = np.expand_dims(sent_attn, axis=1)
    plt.figure()
    plt.axis('off')
    sns.heatmap(sent_attn, cmap='Blues', yticklabels=False, xticklabels=False, annot=article_txt, fmt='',
                annot_kws={'size': 7}, cbar=False)
    plt.tight_layout()
    plt.show()


def show_sentence_attention_difference(hyperpartisan_dataset_folder, hyperpartisan_validation_dataloader,
                                       hyperpartisan_model, joint_model, article_id, sentence_id, device):
    _, hyperpartisan_word_attn, _, _, _, _ = get_attention_weights(
        hyperpartisan_dataset_folder,
        hyperpartisan_validation_dataloader,
        hyperpartisan_model,
        article_id,
        device)

    _, joint_word_attn, _, _, _, article_array = get_attention_weights(
        hyperpartisan_dataset_folder,
        hyperpartisan_validation_dataloader,
        joint_model,
        article_id,
        device)

    sentence_labels = article_array[sentence_id]
    subtracted_word_attention = joint_word_attn[sentence_id] - \
        hyperpartisan_word_attn[sentence_id]

    min_v = min(subtracted_word_attention)
    range_v = max(subtracted_word_attention) - min_v
    normalized_weights = (((subtracted_word_attention - min_v) /
                           range_v)[:len(sentence_labels)])[np.newaxis, :]

    sns.heatmap(normalized_weights, cmap='Blues', yticklabels=False, xticklabels=sentence_labels, cbar=False,
                square=True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


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

        assert not os.path.isfile(joint_model_checkpoint_path)
        assert not os.path.isfile(hyperpartisan_model_checkpoint_path)

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

        load_model_state(hyperpartisan_model,
                         hyperpartisan_model_checkpoint_path, device)
        load_model_state(joint_model, joint_model_checkpoint_path, device)

    print('Loading model state...Done')

    return hyperpartisan_model, joint_model


def predict(config):
    # Define the model
    if config.elmo_model == ELMoModel.Original:
        total_embedding_dim = Constants.ORIGINAL_ELMO_EMBEDDING_DIMENSION
    elif config.elmo_model == ELMoModel.Small:
        total_embedding_dim = Constants.SMALL_ELMO_EMBEDDING_DIMENSION

    if config.concat_glove:
        total_embedding_dim += Constants.GLOVE_EMBEDDING_DIMENSION

    device = torch.device('cpu')

    hyperpartisan_model, joint_model = initialize_models(config.hyperpartisan_model_checkpoint,
                                                         config.joint_model_checkpoint,
                                                         device,
                                                         config.elmo_model,
                                                         config.concat_glove,
                                                         config.model_type)

    # Load GloVe vectors
    if config.concat_glove:
        glove_vectors = utils_helper.load_glove_vectors(
            config.vector_file_name, config.vector_cache_dir, config.glove_size)
    else:
        glove_vectors = None

    # Load the dataset and dataloader
    _, hyperpartisan_validation_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
        hyperpartisan_dataset_folder=config.hyperpartisan_dataset_folder,
        concat_glove=config.concat_glove,
        glove_vectors=glove_vectors,
        elmo_model=config.elmo_model,
        lowercase_sentences=config.lowercase)

    _, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
        validation_dataset=hyperpartisan_validation_dataset,
        test_dataset=None,
        batch_size=1,
        shuffle=False)

    if config.sentence_id:
        show_sentence_attention_difference(
            config.hyperpartisan_dataset_folder,
            hyperpartisan_validation_dataloader,
            hyperpartisan_model,
            joint_model,
            config.article_id,
            config.sentence_id,
            device)
    else:
        title_attention_plot(
            hyperpartisan_validation_dataset, joint_model, device)
        visualize_article_attention(config.hyperpartisan_dataset_folder, hyperpartisan_validation_dataloader,
                                    joint_model,
                                    article_id=config.article_id, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperpartisan_model_checkpoint', type=str, required=True,
                        help='Path to load the hyperpartisan model')
    parser.add_argument('--joint_model_checkpoint', type=str, required=True,
                        help='Path to load the joint model')
    parser.add_argument('--vector_file_name', type=str, required=True,
                        help='File in which vectors are saved')
    parser.add_argument('--article_id', type=int, required=True,
                        help='ID of the article as per the CSV file of the validation set')
    parser.add_argument('--vector_cache_dir', type=str, default='.vector_cache',
                        help='Directory where vectors would be cached')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='Hidden dimension of the recurrent network')
    parser.add_argument('--glove_size', type=int,
                        help='Number of GloVe vectors to load initially')
    parser.add_argument('--lowercase', action='store_true',
                        help='Lowercase the sentences before training')
    parser.add_argument('--hyperpartisan_dataset_folder', type=str,
                        help='Path to the hyperpartisan dataset')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='The number of layers in the biLSTM sentence encoder')
    parser.add_argument('--sentence_id', type=int,
                        help='The sentence id which will be used for calculating the difference')
    parser.add_argument('--skip_connection', action='store_true',
                        help='Indicates whether a skip connection is to be used in the sentence encoder '
                             'while training on hyperpartisan task')
    parser.add_argument('--elmo_model', type=ELMoModel, choices=list(ELMoModel), default=ELMoModel.Original,
                        help='ELMo model from which vectors are used')
    parser.add_argument('--concat_glove', action='store_true',
                        help='Whether GloVe vectors have to be concatenated with ELMo vectors for words')
    parser.add_argument('--include_article_features', action='store_true',
                        help='Whether to append handcrafted article features to the hyperpartisan fc layer')
    parser.add_argument('--model_type', type=str, choices=["ensemble", "single"], default="single",
                        help='Whether to use an ensemble of models or a single model instant')

    config = parser.parse_args()

    predict(config)
