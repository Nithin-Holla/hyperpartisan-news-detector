class Constants():

    DEFAULT_VECTOR_CACHE_DIR: str = '.vector_cache'
    DEFAULT_LEARNING_RATE: float = 0.01
    DEFAULT_MAX_EPOCHS: int = 5
    DEFAULT_HYPERPARTISAN_BATCH_SIZE: int = 8
    DEFAULT_METAPHOR_BATCH_SIZE: int = 64
    DEFAULT_HIDDEN_DIMENSION: int = 128
    DEFAULT_WEIGHT_DECAY: float = 0.
    DEFAULT_SENTENCE_ENCODER_DROPOUT_RATE: float = 0.
    DEFAULT_DOCUMENT_ENCODER_DROPOUT_RATE: float = 0.
    DEFAULT_OUTPUT_ENCODER_DROPOUT_RATE: float = 0.
    DEFAULT_LOSS_SUPPRESS_FACTOR: float = 1
    DEFAULT_HYPERPARTISAN_MAX_LENGTH: int = None
    DEFAULT_NUM_LAYERS: int = 1
    SMALL_ELMO_EMBEDDING_DIMENSION: int = 256
    ORIGINAL_ELMO_EMBEDDING_DIMENSION: int = 1024
    GLOVE_EMBEDDING_DIMENSION: int = 300
    DEFAULT_SKIP_CONNECTION: bool = False
