from typing import Tuple

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm


class ModelHandler:

    @staticmethod
    def validation(
        model: nn.Module,
        test_loader: data.DataLoader,
        criterion
    ) -> Tuple[float, float]:

        accuracy = 0
        test_loss = 0
        i = 0
        with tqdm(
            total=len(test_loader.dataset),
            desc=f'[Epoch {i + 1:3d} / {len(test_loader.dataset)}]',
            leave=False
        ) as pbar:
            for index, (images, labels) in enumerate(test_loader):

                output = model.forward(images)
                test_loss += criterion(output, labels).item()

                ps = torch.exp(output)
                equality = (labels.data == ps.max(1)[1])
                accuracy += equality.type_as(torch.FloatTensor()).mean()

                pbar.set_postfix({
                    'Test Loss'    : f'{test_loss / len(test_loader):.3f}',
                    'Test Accuracy': f'{accuracy / len(test_loader):.3f}',
                })

                pbar.update(images.shape[0])
                i += 1

        return test_loss, accuracy

    def train(
        self,
        model: nn.Module,
        train_loader: data.DataLoader,
        test_loader: data.DataLoader,
        criterion,
        optimizer,
        epochs: int = 5,
        print_every: int = 40
    ) -> None:

        running_loss = 0
        for epoch in range(epochs):
            model.train()

            with tqdm(
                total=len(train_loader.dataset),
                desc=f'[Epoch {epoch + 1:3d} / {epochs}]'
            ) as pbar:
                for i, (images, labels) in enumerate(train_loader):

                    optimizer.zero_grad()

                    output = model(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    if i % print_every == 0:
                        model.eval()

                        with torch.no_grad():
                            test_loss, accuracy = self.validation(model, test_loader, criterion)

                        running_loss = 0

                        model.train()

                    pbar.set_postfix({
                        'Training loss': f'{running_loss / print_every:.3f}',
                        'Test Loss'    : f'{test_loss / len(test_loader):.3f}',
                        'Test Accuracy': f'{accuracy / len(test_loader):.3f}',
                    })

                    pbar.update(images.shape[0])
