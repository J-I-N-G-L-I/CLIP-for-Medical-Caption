import os


class TrainingConfig:
    # data configs
    data_root: str = r'C:\D\Programming\Datasets\rocov2'
    train_split: str = 'train'
    val_split: str = 'valid'
    test_split: str = 'test'