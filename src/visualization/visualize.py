import os
from typing import List

from matplotlib import pyplot as plt

from src.settings import VISUALIZATION_PATH


class Visualizer:

    @staticmethod
    def plot_training_loss(
        training_losses: List[float], validation_losses: List[float], model_name: str
    ) -> None:
        plt.plot(training_losses, label='Training')
        plt.plot(validation_losses, label='Validation')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(VISUALIZATION_PATH, f'{model_name}.png'))
