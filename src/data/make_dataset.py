# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms

from src.settings import DATA_PATH


def get_mnist_data(train: bool = False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = datasets.MNIST(
        os.path.join(DATA_PATH),
        download=True, train=train, transform=transform
    )

    return dataset


def get_data_loader(dataset, batch_size=64):
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    get_mnist_data(train=True)
    get_mnist_data(train=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main('', '')
