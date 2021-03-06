import argparse

from enums.elmo_model import ELMoModel
from enums.training_mode import TrainingMode
from constants import Constants

class ArgumentParserHelper():
    def parse_arguments(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--model_checkpoint', type=str, required=True,
                            help='Path to save/load the model')
        parser.add_argument('--load_model', action='store_true',
                            help='Whether the program should look for a cached model and resume the state')
        parser.add_argument('--data_path', type=str,
                            help='Path where data is saved')
        parser.add_argument('--vector_file_name', type=str, required=True,
                            help='File in which vectors are saved')
        parser.add_argument('--vector_cache_dir', type=str, default=Constants.DEFAULT_VECTOR_CACHE_DIR,
                            help='Directory where vectors would be cached')
        parser.add_argument('--learning_rate', type=float, default=Constants.DEFAULT_LEARNING_RATE,
                            help='Learning rate')
        parser.add_argument('--max_epochs', type=int, default=Constants.DEFAULT_MAX_EPOCHS,
                            help='Maximum number of epochs to train the model')
        parser.add_argument('--hyperpartisan_batch_size', type=int, default=Constants.DEFAULT_HYPERPARTISAN_BATCH_SIZE,
                            help='Batch size for training on the hyperpartisan dataset')
        parser.add_argument('--metaphor_batch_size', type=int, default=Constants.DEFAULT_METAPHOR_BATCH_SIZE,
                            help='Batch size for training on the metaphor dataset')
        parser.add_argument('--sent_encoder_hidden_dim', type=int, default=Constants.DEFAULT_HIDDEN_DIMENSION,
                            help='Hidden dimension of the recurrent network')
        parser.add_argument('--glove_size', type=int,
                            help='Number of GloVe vectors to load initially')
        parser.add_argument('--weight_decay', type=float, default = Constants.DEFAULT_WEIGHT_DECAY,
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
        parser.add_argument('--deterministic', type=int,
                            help='The seed to be used when running deterministically. If nothing is passed, the program run will be stochastic')
        parser.add_argument('--joint_metaphors_first', action='store_true',
                            help='If joint mode is used, this specifies whether metaphors should be batched first or not')
        parser.add_argument('--sent_encoder_dropout_rate', type=float, default=Constants.DEFAULT_SENTENCE_ENCODER_DROPOUT_RATE,
                            help='Dropout rate to be used in the biLSTM sentence encoder')
        parser.add_argument('--doc_encoder_dropout_rate', type=float, default=Constants.DEFAULT_DOCUMENT_ENCODER_DROPOUT_RATE,
                            help='Dropout rate to be used in the biLSTM document encoder')
        parser.add_argument('--output_dropout_rate', type=float, default=Constants.DEFAULT_OUTPUT_ENCODER_DROPOUT_RATE,
                            help='Dropout rate to be used in the classification layer')
        parser.add_argument('--loss_suppress_factor', type=float, default=Constants.DEFAULT_LOSS_SUPPRESS_FACTOR,
                            help='The factor by which hyperpartisan loss is multiplied by during training')
        parser.add_argument('--hyperpartisan_max_length', type=int, default=Constants.DEFAULT_HYPERPARTISAN_MAX_LENGTH,
                            help='The length after which the hyperpartisan articles will be cut off')
        parser.add_argument('--num_layers', type=int, default=Constants.DEFAULT_NUM_LAYERS,
                            help='The number of layers in the biLSTM sentence encoder')
        parser.add_argument('--skip_connection', action='store_true',
                            help='Indicates whether a skip connection is to be used in the sentence encoder '
                                 'while training on hyperpartisan task')
        parser.add_argument('--elmo_model', type=ELMoModel, choices=list(ELMoModel), default=ELMoModel.Original,
                            help='ELMo model from which vectors are used')
        parser.add_argument('--concat_glove', action='store_true',
                            help='Whether GloVe vectors have to be concatenated with ELMo vectors for words')
        parser.add_argument('--hyperpartisan_batch_max_size', type=int, default=Constants.DEFAULT_HYPERPARTISAN_BATCH_MAX_SIZE,
                            help='The maximum size used for dynamically batching from hyperpartisan dataset')
        parser.add_argument('--include_article_features', action='store_true',
                            help='Whether to append handcrafted article features to the hyperpartisan fc layer')
        parser.add_argument('--load_pretrained', action='store_true',
                            help='Whether a pre-trained model on the snli task should be loaded for the sentence encoder')
        parser.add_argument('--pretrained_path', type=str,
                            help="Path to the pretrained model on the snli")
        parser.add_argument('--freeze_sentence_encoder', action="store_true",
                            help="whether to keep the sentence encoder fixed (no gradients)")
        parser.add_argument('--doc_encoder_hidden_dim', type=int, default=Constants.DEFAULT_DOC_ENCODER_DIM,
                            help='Hidden dimension size of document encoder')
        parser.add_argument('--class_weights', action='store_true',
                            help='wether to use class weighting in the cross entropy loss')
        parser.add_argument('--document_encoder_model', type=str, choices=["LSTM", "GRU"], default=Constants.DEFAULT_DOCUMENT_ENCODER_MODEL,
                            help='type of encoder to be used in DocumentEncoder')
        parser.add_argument('--pre_attention_layer', action='store_true',
                            help='wether to use a fully connected layer before applying the attention in both encoders')
        parser.add_argument('--per_layer_config', action='store_true',
                            help='wether to use different learning rate and weight decay depending on the parameters')

        config = parser.parse_args()

        self._load_config(config)

    def _load_config(self, config):
        self._model_checkpoint: str = config.model_checkpoint
        self._data_path: str = config.data_path
        self._vector_file_name: str = config.vector_file_name
        self._vector_cache_dir: str = config.vector_cache_dir
        self._learning_rate: float = config.learning_rate
        self._max_epochs: int = config.max_epochs
        self._hyperpartisan_batch_size: int = config.hyperpartisan_batch_size 
        self._metaphor_batch_size: int = config.metaphor_batch_size
        self._sent_encoder_hidden_dim: int = config.sent_encoder_hidden_dim
        self._glove_size: int = config.glove_size
        self._weight_decay: float = config.weight_decay
        self._metaphor_dataset_folder: str = config.metaphor_dataset_folder
        self._hyperpartisan_dataset_folder: str = config.hyperpartisan_dataset_folder
        self._mode: TrainingMode = config.mode
        self._lowercase: bool = config.lowercase
        self._tokenize: bool = not config.not_tokenize
        self._only_news: bool = config.only_news
        self._deterministic: int = config.deterministic
        self._joint_metaphors_first: bool = config.joint_metaphors_first
        self._sent_encoder_dropout_rate: float = config.sent_encoder_dropout_rate
        self._doc_encoder_dropout_rate: float = config.doc_encoder_dropout_rate
        self._output_dropout_rate: float = config.output_dropout_rate
        self._load_model: bool = config.load_model
        self._loss_suppress_factor: float = config.loss_suppress_factor
        self._hyperpartisan_max_length: int = config.hyperpartisan_max_length
        self._num_layers: int = config.num_layers
        self._skip_connection: bool = config.skip_connection
        self._elmo_model: ELMoModel = config.elmo_model
        self._concat_glove: bool = config.concat_glove
        self._hyperpartisan_batch_max_size: int = config.hyperpartisan_batch_max_size
        self._include_article_features: bool = config.include_article_features
        self._load_pretrained: bool = config.load_pretrained
        self._pretrained_path: bool = config.pretrained_path
        self._freeze_sentence_encoder: bool = config.freeze_sentence_encoder
        self._doc_encoder_hidden_dim: int = config.doc_encoder_hidden_dim
        self._class_weights: bool = config.class_weights
        self._document_encoder_model: str = config.document_encoder_model
        self._pre_attention_layer: bool = config.pre_attention_layer
        self._per_layer_config: bool = config.per_layer_config

    def print_unique_arguments(self):
        print(f'learning_rate: {self._learning_rate}\n' + 
              f'max_epochs: {self._max_epochs}\n' + 
              f'hyper_batch_size: {self._hyperpartisan_batch_size}\n' + 
              f'metaphor_batch_size: {self._metaphor_batch_size}\n' + 
              f'sent_encoder_hidden_dim: {self._sent_encoder_hidden_dim}\n' + 
              f'glove_size: {self._glove_size}\n' + 
              f'weight_decay: {self._weight_decay}\n' + 
              f'mode: {self._mode}\n' + 
              f'lowercase: {self._lowercase}\n' + 
              f'tokenize: {self.tokenize}\n' + 
              f'only_news: {self._only_news}\n' + 
              f'deterministic: {self._deterministic}\n' + 
              f'joint_metaphors_first: {self._joint_metaphors_first}\n' +
              f'sent_encoder_dropout_rate: {self._sent_encoder_dropout_rate}\n' +
              f'doc_encoder_dropout_rate: {self._doc_encoder_dropout_rate}\n' +
              f'output_dropout_rate: {self._output_dropout_rate}\n' +
              f'loss_suppress_factor: {self._loss_suppress_factor}\n' +
              f'hyperpartisan_max_length: {self._hyperpartisan_max_length}\n' +
              f'num_layers: {self._num_layers}\n' +
              f'skip_connection: {self._skip_connection}\n' +
              f'elmo_model: {self._elmo_model}\n' +
              f'concat_glove: {self._concat_glove}\n' +
              f'hyperpartisan_batch_max_size: {self._hyperpartisan_batch_max_size}\n' +
              f'include_article_features: {self._include_article_features}\n' +
              f'load_pretrained: {self._load_pretrained}\n' +
              f'pretrained_path: {self._pretrained_path}\n' +
              f'freeze_sentence_encoder: {self._freeze_sentence_encoder}\n' +
              f'doc_encoder_hidden_dim: {self._doc_encoder_hidden_dim}\n' +
              f'class_weights: {self._class_weights}\n' +
              f'document_encoder_model: {self._document_encoder_model}\n' +
              f'pre_attention_layer: {self.pre_attention_layer}\n' +
              f'per_layer_config: {self.per_layer_config}\n')

    @property
    def model_checkpoint(self) -> str:
        return self._model_checkpoint

    @property
    def data_path(self) -> str:
        return self._data_path

    @property
    def vector_file_name(self) -> str:
        return self._vector_file_name

    @property
    def vector_cache_dir(self) -> str:
        return self._vector_cache_dir

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    @property
    def hyperpartisan_batch_size(self) -> int:
        return self._hyperpartisan_batch_size
        
    @property
    def metaphor_batch_size(self) -> int:
        return self._metaphor_batch_size

    @property
    def sent_encoder_hidden_dim(self) -> int:
        return self._sent_encoder_hidden_dim

    @property
    def glove_size(self) -> int:
        return self._glove_size

    @property
    def weight_decay(self) -> float:
        return self._weight_decay

    @property
    def metaphor_dataset_folder(self) -> str:
        return self._metaphor_dataset_folder

    @property
    def hyperpartisan_dataset_folder(self) -> str:
        return self._hyperpartisan_dataset_folder

    @property
    def mode(self) -> TrainingMode:
        return self._mode

    @property
    def lowercase(self) -> bool:
        return self._lowercase

    @property
    def tokenize(self) -> bool:
        return self._tokenize

    @property
    def only_news(self) -> bool:
        return self._only_news

    @property
    def deterministic(self) -> int:
        return self._deterministic

    @property
    def joint_metaphors_first(self) -> int:
        return self._joint_metaphors_first

    @property
    def sent_encoder_dropout_rate(self) -> float:
        return self._sent_encoder_dropout_rate

    @property
    def doc_encoder_dropout_rate(self) -> float:
        return self._doc_encoder_dropout_rate

    @property
    def output_dropout_rate(self) -> float:
        return self._output_dropout_rate

    @property
    def load_model(self) -> bool:
        return self._load_model

    @property
    def loss_suppress_factor(self) -> float:
        return self._loss_suppress_factor

    @property
    def hyperpartisan_max_length(self) -> int:
        return self._hyperpartisan_max_length

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def skip_connection(self) -> bool:
        return self._skip_connection

    @property
    def elmo_model(self) -> ELMoModel:
        return self._elmo_model

    @property
    def concat_glove(self) -> bool:
        return self._concat_glove

    @property
    def hyperpartisan_batch_max_size(self) -> int:
        return self._hyperpartisan_batch_max_size

    @property
    def include_article_features(self) -> bool:
        return self._include_article_features

    @property
    def load_pretrained(self) -> bool:
        return self._load_pretrained

    @property
    def pretrained_path(self) -> str:
        return self._pretrained_path

    @property
    def freeze_sentence_encoder(self) -> bool:
        return self._freeze_sentence_encoder

    @property
    def doc_encoder_hidden_dim(self) -> int:
        return self._doc_encoder_hidden_dim

    @property
    def class_weights(self) -> bool:
        return self._class_weights

    @property
    def document_encoder_model(self) -> str:
        return self._document_encoder_model
    
    @property
    def pre_attention_layer(self) -> bool:
        return self._pre_attention_layer
    
    @property
    def per_layer_config(self) -> bool:
        return self._per_layer_config
    
    
    
    