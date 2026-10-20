from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent / "data"
MEAN, STD = 0.2860, 0.3530
BATCH_SIZE = 128


def get_loaders(root=DATA_DIR, batch_size=BATCH_SIZE, seed=42):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    full_train = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)

    g = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [55_000, 5_000], generator=g)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )