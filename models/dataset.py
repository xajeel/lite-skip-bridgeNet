import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

DATASET_PATH = '/kaggle/working/PCOS'

def get_data_loaders(batch_size=32, subset_size=5000, train_ratio=0.8):
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=train_transform)
    
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, indices)

    train_size = int(train_ratio * subset_size)
    test_size = subset_size - train_size

    train_dataset, test_dataset = random_split(subset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
