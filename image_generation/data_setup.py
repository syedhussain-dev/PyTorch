import torch
from pathlib import Path
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from custom_datasets.ImageDatasets import ImageFolderCustom

num_workers = os.cpu_count()

def create_dataloaders(target_dir:str,train_transforms:transforms,
                       val_transforms:transforms,test_transforms:transforms,
                       batch_size:int):
    train_dir = Path(target_dir/"train")
    val_dir = Path(target_dir/"val")
    test_dir = Path(target_dir/"test")

    train_dataset = ImageFolderCustom(
        root=train_dir,transform=train_transforms
    )
    val_dataset = ImageFolderCustom(
        root=val_dir, transform=val_transforms,
    )
    test_dataset = ImageFolderCustom(
        root=test_dir, transform=test_transforms
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataloader, val_dataloader, test_dataloader