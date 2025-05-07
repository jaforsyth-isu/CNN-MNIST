from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

def mnist(train: bool, batch_size: int) -> DataLoader:
    #Retrieve a batch of the training data or the test data with labels

    dataset = MNIST(
        root="dataset",
        train=train,
        download=True,
        transform=Compose([ToTensor(), Normalize(mean=(0.1307,), std=(0.3081,))])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader