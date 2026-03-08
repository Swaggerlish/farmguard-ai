from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transforms import get_train_transforms, get_eval_transforms

def build_dataloaders(
    train_dir,
    val_dir,
    test_dir,
    batch_size=32,
    img_size=224,
    num_workers=2,
    pin_memory=True
):
    train_ds = ImageFolder(train_dir, transform=get_train_transforms(img_size))
    val_ds = ImageFolder(val_dir, transform=get_eval_transforms(img_size))
    test_ds = ImageFolder(test_dir, transform=get_eval_transforms(img_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader