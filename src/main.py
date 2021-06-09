import argparse
import os
from datetime import datetime

import torch
from torch import optim, nn

from src.data.make_dataset import get_mnist_data, get_data_loader
from src.models.classifier import Classifier
from src.models.train_model import ModelHandler
from src.settings import MODELS_PATH
from src.visualization.visualize import Visualizer

NOW = datetime.strftime(datetime.now(), '%Y-%m-%d_%H_%M_%S')


class ArgumentParser:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Script for either training or evaluating',
            usage='python main.py <command>'
        )
        parser.add_argument(
            '--train', '-t', dest='train', action='store_true', help='Train model')
        parser.add_argument(
            '--save', '-s', dest='save', action='store_true', help='Save model after training')
        parser.add_argument(
            '--test', '-v', dest='test', help='Test model from given path')

        self.args = parser.parse_args()
        self.model_handler = ModelHandler()

    def run_train(self) -> None:
        train_dataset = get_mnist_data(train=True)
        test_dataset = get_mnist_data(train=False)
        train_loader = get_data_loader(train_dataset)
        test_loader = get_data_loader(test_dataset)

        model = Classifier()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        self.model_handler.train(model, train_loader, test_loader, criterion, optimizer)
        Visualizer.plot_training_loss(
            self.model_handler.running_loss_list, self.model_handler.test_loss_list, NOW)

        if self.args.save:
            current_date = NOW
            torch.save(model.state_dict(), os.path.join(MODELS_PATH, f'{current_date}.pth'))

    def run_test(self, model_path: str = '') -> None:
        model = torch.load(model_path)
        test_loader = get_mnist_data(train=False)
        criterion = nn.NLLLoss()
        self.model_handler.validation(model, test_loader, criterion)


def main():
    argument_parser = ArgumentParser()

    if argument_parser.args.train:
        argument_parser.run_train()

    elif argument_parser.args.test:
        argument_parser.run_test(argument_parser.args.test)


if __name__ == '__main__':
    main()
