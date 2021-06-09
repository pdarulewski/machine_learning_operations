from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms


def get_mnist_data(train: bool = False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = datasets.MNIST(
        '.pytorch/MNIST_data/', download=True, train=train, transform=transform
    )
    data_loader = data.DataLoader(dataset, batch_size=64, shuffle=True)

    return data_loader
