import pytest
import torch

from src.data.make_dataset import get_mnist_data, get_data_loader


class TestMakeDataset:

    @staticmethod
    def dataset(train):
        return get_mnist_data(train)

    def dataloader(self, train, batch_size=1):
        return get_data_loader(self.dataset(train), batch_size)

    @pytest.mark.parametrize(
        'train, expected', [
            (True, 60000),
            (False, 10000)
        ])
    def test_dataset_length(self, train, expected):
        length = len(self.dataset(train))
        assert length == expected

    @pytest.mark.parametrize(
        'train, shape', [
            (True, torch.Size([1, 1, 28, 28])),
            (False, torch.Size([1, 1, 28, 28])),
        ])
    def test_shape_of_data_items(self, train, shape):
        for item, _ in self.dataloader(train, 1):
            assert shape == item.shape

    @pytest.mark.parametrize(
        'train, labels', [
            (True, torch.Size([1, 1, 28, 28])),
            (False, torch.Size([1, 1, 28, 28])),
        ])
    def test_shape_of_data_items(self, train, labels):
        dataset = get_mnist_data(train)
        dataloader = get_data_loader(dataset, batch_size=1)
        for _, label in dataloader:
            assert label in range(10)
