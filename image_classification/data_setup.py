import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from custom_datasets.ImageDatasets import ImageFolderCustom
NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    val_dir: str,
    train_transform: transforms.Compose,
    test_and_val_transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):

  """
  Creates training and testing dataloaders

  Takes training and testing directories to create a dataset and then create and
  then create a dataloader and potentially apply a transformation to add variety
  to the data. Other parameters determine the batch size and amount of cpu cores
  working on the training

  Args:
    train_dir: Directory which contains all the training data
    val_dir: Directory which contains all the validating data
    test_dir: Directory which contains all the testing data
    train_transform: A transformation that can be applied to the train dataloader
    test_transform: A transformation that can be applied to the test dataloader
    batch_size: The amount of data that is funneled into the batch function
    num_workers: The number of cpu cores working on the dataloader

  Returns:
    train_dataloader: The dataloader that funnels the training information
    val_dataloader: The dataloader that funnels the validating information
    test_dataloader: The dataloader that funnels the testing information
    class_names: The names of the classes in the dataset
  """

  train_dataset = ImageFolderCustom(
      root=train_dir, transform=train_transform,
  )

  val_dataset = ImageFolderCustom(
      root=val_dir, transform=test_and_val_transform
  )

  test_dataset = ImageFolderCustom(
      root=test_dir, transform=test_and_val_transform,

  )

  PIN_MEMORY = torch.cuda.is_available()

  train_dataloader = DataLoader(
      dataset=train_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      shuffle=True,
      pin_memory=PIN_MEMORY
  )

  val_dataloader = DataLoader(
      dataset=val_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      shuffle=False,
      pin_memory=PIN_MEMORY
  )

  test_dataloader = DataLoader(
      dataset=test_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      shuffle=False,
      pin_memory=PIN_MEMORY
  )

  class_names = train_dataset.classes

  return train_dataloader, val_dataloader, test_dataloader, class_names
