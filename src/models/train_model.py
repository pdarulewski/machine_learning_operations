from typing import Tuple

import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ModelHandler:

    def __init__(self):
        self.running_loss_list, self.accuracy_list, self.test_loss_list = [], [], []
        self.writer = SummaryWriter()

    def validation(
        self,
        model: nn.Module,
        test_loader: data.DataLoader,
        criterion
    ) -> Tuple[float, float]:
        model.eval()
        accuracy, test_loss = 0, 0

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

                test_loss_actual = test_loss / len(test_loader)
                accuracy_actual = accuracy / len(test_loader)

                pbar.set_postfix({
                    'Test Loss'    : f'{test_loss_actual:.3f}',
                    'Test Accuracy': f'{accuracy_actual:.3f}',
                })

                pbar.update(images.shape[0])
                i += 1
        model.train()
        return accuracy, test_loss

    def train(
        self,
        model: nn.Module,
        train_loader: data.DataLoader,
        test_loader: data.DataLoader,
        criterion,
        optimizer,
        epochs: int = 1,
        print_every: int = 40
    ) -> None:

        running_loss = 0
        for epoch in range(epochs):
            model.train()

            epoch_loss = 0

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

                    with torch.no_grad():
                        test_loss, accuracy = self.validation(model, test_loader, criterion)

                    self.writer.add_scalar('Loss/train', running_loss, i)

                    pbar.set_postfix({
                        'Training loss': f'{running_loss:.3f}',
                        'Test Loss'    : f'{test_loss / len(test_loader):.3f}',
                        'Test Accuracy': f'{accuracy / len(test_loader):.3f}',
                    })
                    pbar.update(images.shape[0])
                    epoch_loss += running_loss
                    running_loss = 0

                self.writer.add_scalar('Loss/test', test_loss / len(test_loader), i)
                self.writer.add_scalar('Accuracy/test', accuracy / len(test_loader), i)

            self.running_loss_list.append(epoch_loss / print_every)
            self.test_loss_list.append(test_loss / len(test_loader))
            self.accuracy_list.append(accuracy / len(test_loader))
