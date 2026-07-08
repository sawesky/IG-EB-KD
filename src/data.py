import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def get_dataset_and_stats(dataset_name):
    if dataset_name == "mnist":
        return datasets.MNIST, (0.1307,), (0.3081,)

    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST, (0.2860,), (0.3530,)

    if dataset_name == "cifar10":
        return datasets.CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    if dataset_name == "cifar100":
        return datasets.CIFAR100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_image_loaders(
    dataset_name="mnist",
    root="data",
    batch_size=128,
    val_size=0.2,
    num_workers=2,
    seed=42,
    augment=None,
):
    DatasetClass, mean, std = get_dataset_and_stats(dataset_name)

    if augment is None:
        augment = dataset_name in ["cifar10", "cifar100"]

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset_name in ["cifar10", "cifar100"] and augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = eval_transform

    # Same training images, two different transform views:
    # one augmented for training, one clean for validation.
    full_train_aug = DatasetClass(root=root, train=True, download=True, transform=train_transform)
    full_train_eval = DatasetClass(root=root, train=True, download=True, transform=eval_transform)

    test_set = DatasetClass(root=root, train=False, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)

    train_size = 1 - val_size

    split_source = DatasetClass(root=root, train=True, download=True, transform=eval_transform)

    train_split, val_split = random_split(
        split_source,
        [train_size, val_size],
        generator=generator,
    )

    train_dataset = Subset(full_train_aug, train_split.indices)
    val_dataset = Subset(full_train_eval, val_split.indices)

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