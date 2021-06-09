import pytest
import torch

from src.models.classifier import Classifier


@pytest.fixture
def model():
    return Classifier()


class TestClassifier:

    def test_output_shape(self, model):
        x = torch.rand(1, 1, 28, 28)
        output = model.forward(x)
        assert output.shape == torch.Size([1, 10])

    def test_dimension_error(self, model):
        x = torch.rand(1, 1, 1, 28, 28)

        with pytest.raises(ValueError):
            model.forward(x)
