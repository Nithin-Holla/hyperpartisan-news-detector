import argparse

from enums.training_mode import TrainingMode


class ArgumentParserHelper():
    def __init__(self):
        self._model_checkpoint: str = None
        self._data_path: str = None
        self._vector_file_name: str = None
        self._vector_cache_dir: str = None
        self._learning_rate: float = 2e-3
        self._max_epochs: int = 5
        self._hyperpartisan_batch_size: int = 4
        self._metaphor_batch_size: int = 32
        self._hidden_dim: int = 100
        self._glove_size: int = None
        self._weight_decay: float = 0.
        self._metaphor_dataset_folder: str = None
        self._hyperpartisan_dataset_folder: str = None
        self._mode: TrainingMode = None
        self._lowercase: bool = False
        self._tokenize: bool = True
        self._only_news: bool = False
        self._deterministic: int = False
        self._joint_metaphors_first: bool = False
        self._sent_encoder_dropout_rate: float = 0.
        self._doc_encoder_dropout_rate: float = 0.
        self._output_dropout_rate: float = 0.
        self._load_model: bool = False
        self._loss_suppress_factor: float = 1.
        self._hyperpartisan_max_length: int = None
        self._num_layers: int = 1
        self._metaphor_sampling_prob: float = 0.5

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
        parser.add_argument('--vector_cache_dir', type=str, default='.vector_cache',
                            help='Directory where vectors would be cached')
        parser.add_argument('--learning_rate', type=float, default=2e-3,
                            help='Learning rate')
        parser.add_argument('--max_epochs', type=int, default=5,
                            help='Maximum number of epochs to train the model')
        parser.add_argument('--hyperpartisan_batch_size', type=int, default=4,
                            help='Batch size for training on the hyperpartisan dataset')
        parser.add_argument('--metaphor_batch_size', type=int, default=32,
                            help='Batch size for training on the metaphor dataset')
        parser.add_argument('--hidden_dim', type=int, default=100,
                            help='Hidden dimension of the recurrent network')
        parser.add_argument('--glove_size', type=int,
                            help='Number of GloVe vectors to load initially')
        parser.add_argument('--weight_decay', type=float, default = 0.,
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
        parser.add_argument('--sent_encoder_dropout_rate', type=float, default = 0.,
                            help='Dropout rate to be used in the biLSTM sentence encoder')
        parser.add_argument('--doc_encoder_dropout_rate', type=float, default = 0.,
                            help='Dropout rate to be used in the biLSTM document encoder')
        parser.add_argument('--output_dropout_rate', type=float, default = 0.,
                            help='Dropout rate to be used in the classification layer')
        parser.add_argument('--loss_suppress_factor', type=float, default=1,
                            help='The factor by which hyperpartisan loss is multiplied by during training')
        parser.add_argument('--hyperpartisan_max_length', type=int, default=None,
                            help='The length after which the hyperpartisan articles will be cut off')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='The number of layers in the biLSTM sentence encoder')
        parser.add_argument('--metaphor_sampling_prob', type=float, default=0.5,
                            help='The probability of sampling a metaphor batch in each step of joint training')

        config = parser.parse_args()

        self._load_config(config)

    def _load_config(self, config):
        self._model_checkpoint = config.model_checkpoint
        self._data_path = config.data_path
        self._vector_file_name = config.vector_file_name
        self._vector_cache_dir = config.vector_cache_dir
        self._learning_rate = config.learning_rate
        self._max_epochs = config.max_epochs
        self._hyperpartisan_batch_size = config.hyperpartisan_batch_size 
        self._metaphor_batch_size = config.metaphor_batch_size
        self._hidden_dim = config.hidden_dim
        self._glove_size = config.glove_size
        self._weight_decay = config.weight_decay
        self._metaphor_dataset_folder = config.metaphor_dataset_folder
        self._hyperpartisan_dataset_folder = config.hyperpartisan_dataset_folder
        self._mode = config.mode
        self._lowercase = config.lowercase
        self._tokenize = not config.not_tokenize
        self._only_news = config.only_news
        self._deterministic = config.deterministic
        self._joint_metaphors_first = config.joint_metaphors_first
        self._sent_encoder_dropout_rate = config.sent_encoder_dropout_rate
        self._doc_encoder_dropout_rate = config.doc_encoder_dropout_rate
        self._output_dropout_rate = config.output_dropout_rate
        self._load_model = config.load_model
        self._loss_suppress_factor = config.loss_suppress_factor
        self._hyperpartisan_max_length = config.hyperpartisan_max_length
        self._num_layers = config.num_layers
        self._metaphor_sampling_prob = config.metaphor_sampling_prob

    def print_unique_arguments(self):
        print(f'learning_rate: {self._learning_rate}\n' + 
              f'max_epochs: {self._max_epochs}\n' + 
              f'hyper_batch_size: {self._hyperpartisan_batch_size}\n' + 
              f'metaphor_batch_size: {self._metaphor_batch_size}\n' + 
              f'hidden_dim: {self._hidden_dim}\n' + 
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
              f'metaphor_sampling_prob: {self._metaphor_sampling_prob}\n')

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
    def hidden_dim(self) -> int:
        return self._hidden_dim

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
    def metaphor_sampling_prob(self) -> float:
        return self._metaphor_sampling_prob
