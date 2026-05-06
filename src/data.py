import torch
from torchvision import datasets, transforms


def get_mnist_loaders(root="data", batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader
