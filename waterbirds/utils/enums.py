from enum import Enum


class Task(Enum):
    ERM = 'erm'
    VAE = 'vae'
    CLASSIFY = 'classify'


class Environment(Enum):
    BAMBOO_FOREST = 'bamboo_forest'
    FOREST = 'forest'
    LAKE = 'lake'
    OCEAN = 'ocean'


class EvalStage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'