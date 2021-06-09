import os

import torch
from torch import nn
from tqdm import tqdm

from src.settings import MODELS_PATH


class Predictor:
    def __init__(self):
        self.model = None
        self.dataloader = None

    def read_model(self, filename):
        self.model = torch.load(os.path.join(MODELS_PATH, filename))

    def create_dataloader(self, data):
        self.dataloader = None

    def run_prediction(self):
        criterion = nn.NLLLoss()

        accuracy, test_loss = 0, 0

        i = 0
        with tqdm(
            total=len(self.dataloader.dataset),
            desc=f'[Epoch {i + 1:3d} / {len(self.dataloader.dataset)}]',
            leave=False
        ) as pbar:
            for index, (data, labels) in enumerate(self.dataloader):

                output = self.model.forward(data)
                test_loss += criterion(output, labels).item()

                ps = torch.exp(output)
                equality = (labels.data == ps.max(1)[1])
                accuracy += equality.type_as(torch.FloatTensor()).mean()

                pbar.set_postfix({
                    'Test Loss'    : f'{test_loss / len(self.dataloader):.3f}',
                    'Test Accuracy': f'{accuracy / len(self.dataloader):.3f}',
                })

                pbar.update(data.shape[0])
                i += 1

        return accuracy, test_loss
