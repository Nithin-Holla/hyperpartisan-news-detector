import torch
import os
import argparse
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from enums.training_mode import TrainingMode
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan
from helpers.hyperpartisan_loader import HyperpartisanLoader
from model.JointModel import JointModel
from train import load_glove_vectors


def title_attention_plot(hyperpartisan_validation_dataset, model):
    _, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
        validation_dataset=hyperpartisan_validation_dataset,
        test_dataset=None,
        batch_size=4,
        shuffle=False)

    attn_summary = None

    for idx, (article, article_target, article_recover_idx, article_num_sent, article_sent_lengths, extra_feat) in enumerate(
            hyperpartisan_validation_dataloader):
        model.eval()
        with torch.no_grad():
            pred, _, batch_sent_attn = model.forward(article,
                                                     (article_recover_idx, article_num_sent, article_sent_lengths, extra_feat),
                                                     task=TrainingMode.Hyperpartisan, return_attention=True)
        batch_sent_attn = np.around(batch_sent_attn.numpy(), decimals=4)
        article_num_sent = article_num_sent.numpy()
        article_attn = np.zeros((batch_sent_attn.shape[0], 2))
        article_attn[:, 0] = batch_sent_attn[:, 0]
        article_attn[:, 1] = np.sum(batch_sent_attn[:, 1:], axis=1) / (article_num_sent - 1)
        if idx == 0:
            attn_summary = article_attn
        else:
            attn_summary = np.vstack([attn_summary, article_attn])
    attn_summary = np.mean(attn_summary, axis=0, keepdims=True)
    plt.figure()
    sns.heatmap(attn_summary, square=True, annot=True)
    plt.show()


def visualize_article_attention(hyperpartisan_validation_dataloader, model, article_id):
    """
    Generates a heatmap of the word-level and attention-level attention weights
    :param hyperpartisan_validation_dataloader: Dataloader for hyperpartisan validation set
    :param model: Trained model
    :param article_id: ID of the article, i.e., the line number in valid_byart.txt
    :return: None
    """
    # Load one article
    for idx, (article, article_target, article_recover_idx, article_num_sent, article_sent_lengths, extra_feat) in enumerate(
            hyperpartisan_validation_dataloader):
        if idx + 2 == article_id:
            break

    # Obtain prediction and attention weights from the model
    model.eval()
    with torch.no_grad():
        pred, word_attn, sent_attn = model.forward(article,
                                                   (article_recover_idx, article_num_sent, article_sent_lengths, extra_feat),
                                                   task=TrainingMode.Hyperpartisan, return_attention=True)
    pred = pred.item()
    article_sent_lengths = article_sent_lengths[article_recover_idx]
    word_attn = np.around(word_attn.numpy(), decimals=4)
    sent_attn = np.around(sent_attn.numpy(), decimals=4)

    # Display results
    print('Hyperpartisan score: {}'.format(pred))
    for sent_idx in range(article_num_sent):
        print('Sentence {} -> Word-level attention: {}'.format(sent_idx + 1,
                                                               word_attn[sent_idx, range(article_sent_lengths[sent_idx])]))
    print('Sentence-level attention: {}'.format(sent_attn))

    # Plot word-level attention weights per sentence
    plt.figure()
    sns.heatmap(word_attn)

    # Plot sentence-level attention weights
    plt.figure()
    sns.heatmap(sent_attn, square=True)
    plt.show()


def predict(config):
    # Define the model
    total_embedding_dim = 1024 + 300
    device = torch.device('cpu')
    joint_model = JointModel(embedding_dim=total_embedding_dim,
                             hidden_dim=config.hidden_dim,
                             num_layers=config.num_layers,
                             sent_encoder_dropout_rate=0,
                             doc_encoder_dropout_rate=0,
                             output_dropout_rate=0,
                             device=device)

    # Load the model if found
    if os.path.isfile(config.model_file):
        checkpoint = torch.load(config.model_file)
        joint_model.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded the model')
    else:
        print('Could not find the model')

    # Load GloVe vectors
    glove_vectors = load_glove_vectors(config.vector_file_name, config.vector_cache_dir, config.glove_size)

    # Load the dataset and dataloader
    _, hyperpartisan_validation_dataset = HyperpartisanLoader.get_hyperpartisan_datasets(
        hyperpartisan_dataset_folder=config.hyperpartisan_dataset_folder,
        glove_vectors=glove_vectors,
        lowercase_sentences=config.lowercase)

    _, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
        validation_dataset=hyperpartisan_validation_dataset,
        test_dataset=None,
        batch_size=1,
        shuffle=False)

    title_attention_plot(hyperpartisan_validation_dataset, joint_model)
    visualize_article_attention(hyperpartisan_validation_dataloader, joint_model, article_id=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to save/load the model')
    parser.add_argument('--vector_file_name', type=str, required=True,
                        help='File in which vectors are saved')
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
    config = parser.parse_args()

    predict(config)
