from enum import Enum


class ELMoModel(Enum):
    Small = 'small'
    Original = 'original'

    def __str__(self):
        return self.value
