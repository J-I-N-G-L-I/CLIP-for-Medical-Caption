import os


class TrainingConfig:
    # data configs
    data_root: str = '../data/ROCO/rocov2'
    train_split: str = 'train'
    val_split: str = 'valid'
    test_split: str = 'test'