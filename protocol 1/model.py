# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loaders(train_batch_size=32, test_batch_size=128, rank=None, num_clients=None):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    if rank is not None and num_clients is not None and num_clients > 1:
        n = len(train)
        shard = n // num_clients
        start = rank * shard
        end = (rank + 1) * shard if rank < num_clients - 1 else n
        idx = list(range(start, end))
        train = Subset(train, idx)

    train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=test_batch_size)
    return train_loader, test_loader

def get_test_loader(test_batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(test, batch_size=test_batch_size)
