from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(train_dir, val_dir, test_dir, image_size=224, batch_size=32):
    transform = {
        "train": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ]),
        "val": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]),
        "test": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    }

    train_data = datasets.ImageFolder(train_dir, transform=transform["train"])
    val_data = datasets.ImageFolder(val_dir, transform=transform["val"])
    test_data = datasets.ImageFolder(test_dir, transform=transform["test"])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_data.classes
