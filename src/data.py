import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataset_and_stats(dataset_name):
    if dataset_name == "mnist":
        return datasets.MNIST, (0.1307,), (0.3081,)

    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST, (0.2860,), (0.3530,)
    
    if dataset_name == "cifar10":
        return datasets.CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_image_loaders(dataset_name="mnist", root="data", batch_size=128, val_size=0.2, num_workers=2, seed=42):

    DatasetClass, mean, std = get_dataset_and_stats(dataset_name)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train_dataset = DatasetClass(root=root, train=True, download=True, transform=transform)
    test_set = DatasetClass(root=root, train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(seed)

    train_size = 1 - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
