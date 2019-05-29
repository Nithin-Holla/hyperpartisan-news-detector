import argparse
import os

from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import permutation_test
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader

import numpy as np

import sys

sys.path.append('./../')

from model.JointModel import JointModel
from model.Ensemble import Ensemble
from helpers.utils_helper import UtilsHelper
from helpers.data_helper_hyperpartisan import DataHelperHyperpartisan
from helpers.hyperpartisan_loader import HyperpartisanLoader
from datasets.hyperpartisan_dataset import HyperpartisanDataset
from constants import Constants
from enums.elmo_model import ELMoModel
from enums.training_mode import TrainingMode
from batches.hyperpartisan_batch import HyperpartisanBatch

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
        glove_vectors_dim: int,
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

        hyperpartisan_model =  JointModel(embedding_dim=total_embedding_dim,
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

        joint_model =  JointModel(embedding_dim=total_embedding_dim,
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

def create_contingency_table(targets, predictions1, predictions2):
    assert len(targets) == len(predictions1)
    assert len(targets) == len(predictions2)

    contingency_table = np.zeros((2, 2))

    targets_length = len(targets)
    contingency_table[0, 0] = sum([targets[i] == predictions1[i] and targets[i] == predictions2[i] for i in range(targets_length)]) # both predictions are correct
    contingency_table[0, 1] = sum([targets[i] == predictions1[i] and targets[i] != predictions2[i] for i in range(targets_length)]) # predictions1 is correct and predictions2 is wrong
    contingency_table[1, 0] = sum([targets[i] != predictions1[i] and targets[i] == predictions2[i] for i in range(targets_length)]) # predictions1 is wrong and predictions2 is correct
    contingency_table[1, 1] = sum([targets[i] != predictions1[i] and targets[i] != predictions2[i] for i in range(targets_length)]) # both predictions are wrong

    print(contingency_table)

    return contingency_table

def forward_full_hyperpartisan(
        model,
        dataloader: DataLoader,
        device: torch.device):

    all_targets = []
    all_predictions = []

    total_length = len(dataloader)
    for step, hyperpartisan_data in enumerate(dataloader):
        print(f'Step {step+1}/{total_length}                  \r', end='')

        hyperpartisan_batch = HyperpartisanBatch(10000) # just something big
        hyperpartisan_batch.add_data(*hyperpartisan_data[:-1]) # exclude last element (id)
        hyperpartisan_data = hyperpartisan_batch.pad_and_sort_batch()

        batch_inputs = hyperpartisan_data[0].to(device)
        batch_targets = hyperpartisan_data[1].to(device)
        batch_recover_idx = hyperpartisan_data[2].to(device)
        batch_num_sent = hyperpartisan_data[3].to(device)
        batch_sent_lengths = hyperpartisan_data[4].to(device)
        batch_feat = hyperpartisan_data[5].to(device)

        batch_predictions = model.forward(batch_inputs, (batch_recover_idx,
                                                        batch_num_sent, batch_sent_lengths, batch_feat), task=TrainingMode.Hyperpartisan)

        all_targets.append(batch_targets.long().item())
        all_predictions.append(batch_predictions.round().long().item())

    return all_targets, all_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperpartisan_model_checkpoint', type=str, required=True,
                        help='Path to load the hyperpartisan model. IMPORTANT: specify the whole folder for using the ensemble')
    parser.add_argument('--joint_model_checkpoint', type=str, required=True,
                        help='Path to load the joint model. IMPORTANT: specify the whole folder for using the ensemble')
    parser.add_argument('--vector_file_name', type=str, required=True,
                        help='File in which vectors are saved')
    parser.add_argument('--vector_cache_dir', type=str, default=Constants.DEFAULT_VECTOR_CACHE_DIR,
                        help='Directory where vectors would be cached')
    parser.add_argument('--glove_size', type=int,
                        help='Number of GloVe vectors to load initially')
    parser.add_argument('--hyperpartisan_dataset_folder', type=str, required=True,
                        help='Path to the hyperpartisan dataset')
    parser.add_argument('--hyperpartisan_batch_size', type=int, default=1,
                        help='Batch size for training on the hyperpartisan dataset. *** its easier with batch of 1')
    parser.add_argument('--metaphor_batch_size', type=int, default=Constants.DEFAULT_METAPHOR_BATCH_SIZE,
                        help='Batch size for training on the metaphor dataset')
    parser.add_argument('--deterministic', type=int, required=True,
                        help='The seed to be used when running deterministically. If nothing is passed, the program run will be stochastic')
    parser.add_argument('--alpha_value', type=int, default=0.05,
                        help='The alpha value which will be used to statistically test the significant difference')
    parser.add_argument('--elmo_model', type=ELMoModel, choices=list(ELMoModel), default=ELMoModel.Original,
                        help='ELMo model from which vectors are used')
    parser.add_argument('--concat_glove', action='store_true',
                        help='Whether GloVe vectors have to be concatenated with ELMo vectors for words')
    parser.add_argument('--include_article_features', action='store_true',
                        help='Whether to append handcrafted article features to the hyperpartisan fc layer')
    parser.add_argument('--model_type', type=str, choices=["ensemble", "single"], default = "single",
                        help='Whether to use an ensemble of models or a single model instant')

    config = parser.parse_args()

    utils_helper.initialize_deterministic_mode(config.deterministic)

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

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
                                                         glove_vectors.dim,
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

    hyperpartisan_valid_targets, hyperpartisan_valid_predictions = forward_full_hyperpartisan(
        model=hyperpartisan_model,
        dataloader=hyperpartisan_validation_dataloader,
        device=device)

    hyperpartisan_accuracy = accuracy_score(hyperpartisan_valid_targets, hyperpartisan_valid_predictions)
    hyperpartisan_f1, hyperpartisan_precision, hyperpartisan_recall = utils_helper.calculate_metrics(hyperpartisan_valid_targets, hyperpartisan_valid_predictions)

    print(f'Hyperpartisan F1 score: {hyperpartisan_f1}, Precision: {hyperpartisan_precision}, Recall: {hyperpartisan_recall}, Accuracy: {hyperpartisan_accuracy}')

    # was getting a weird error if the same dataloader was used
    _, hyperpartisan_validation_dataloader, _ = DataHelperHyperpartisan.create_dataloaders(
        validation_dataset=hyperpartisan_validation_dataset,
        batch_size=config.hyperpartisan_batch_size,
        shuffle=False)

    joint_valid_targets, joint_valid_predictions = forward_full_hyperpartisan(
        model=joint_model,
        dataloader=hyperpartisan_validation_dataloader,
        device=device)

    joint_accuracy = accuracy_score(joint_valid_targets, joint_valid_predictions)
    joint_f1, joint_precision, joint_recall = utils_helper.calculate_metrics(joint_valid_targets, joint_valid_predictions)

    print(f'Joint F1 score: {joint_f1}, Precision: {joint_precision}, Recall: {joint_recall}, Accuracy: {joint_accuracy}')

    print('Calculating p value using McNemar\'s test...\r', end='')

    # roll your own t-test:
    N=10
    hyperpartisan_valid_predictions = np.array(hyperpartisan_valid_predictions)
    joint_valid_predictions = np.array(joint_valid_predictions)
    var_a = hyperpartisan_valid_predictions.var(ddof=1) # unbiased estimator, divide by N-1 instead of N
    var_b = joint_valid_predictions.var(ddof=1)
    s = np.sqrt( (var_a + var_b) / 2 ) # balanced standard deviation
    t = (hyperpartisan_valid_predictions.mean() - joint_valid_predictions.mean()) / (s * np.sqrt(2.0/N)) # t-statistic
    df = 2*N - 2 # degrees of freedom
    p = 1 - stats.t.cdf(np.abs(t), df=df) # one-sided test p-value
    print("t:\t", t, "p:\t", 2*p) # two-sided test p-value

    t2, p2 = ttest_ind(hyperpartisan_valid_predictions, joint_valid_predictions)
    print("t2:\t", t2, "p2:\t", p2)

    # contingency_table = create_contingency_table(
    #     hyperpartisan_valid_targets,
    #     hyperpartisan_valid_predictions,
    #     joint_valid_predictions)
    
    # result = mcnemar(contingency_table, exact=True)

    # p_value = permutation_test(hyperpartisan_valid_predictions, joint_valid_predictions,
    #                        method='approximate',
    #                        num_rounds=10000,
    #                        seed=0)

    # print('Calculating p value using McNemar\'s test...Done')

    # # summarize the finding
    # print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    
    # print(f'p-value: {p_value}')

    # # interpret the p-value
    # if p_value > config.alpha_value:
    #     print('Same proportions of errors (non-significant difference)')
    # else:
    #     print('Significant difference found')