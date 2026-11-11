import torch 
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
from image_classification.utils import find_classes
from pathlib import Path
from typing import Tuple, Optional, Callable

class ImageFolderCustom(Dataset):
  def __init__(self, root:str, transform=None) -> None:
    self.paths = list(Path(root).glob("*/*.jpg"))
    self.transforms = transform
    self.classes, self.class_to_idx = find_classes(root)

  def _load_image(self, index:int) -> Image.Image:
    image_path = self.paths[index]
    return Image.open(image_path)

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, index:int) -> Tuple[torch.Tensor,int]:
    image = self.load_image(index)
    class_names = self.paths[index].parent.name
    class_idx = self.class_to_idx[class_names]

    if self.transforms:
      return self.transforms(image), class_idx
    else:
      return image, class_idx

class ImageCaptionCustom(Dataset):
  def __init__(self,root:str,transform:Optional[Callable],csv_path:str):
    self.df = pd.read_csv(Path(csv_path))
    self.img_dir = root
    self.transform = transform

    self.image_paths = self.df["filename"].tolist()
    self.captions = self.df["caption"].tolist()

  def _load_image(self,image_path:str) -> Image.Image:
    return Image.open(Path(image_path)).convert("RGB")

  def __len__(self) -> int:
    return len(self.image_paths)

  def __getitem__(self,index:int) -> Tuple[torch.Tensor,str]:
    if not isinstance(index,int):
      raise ValueError(f"Index must be type integer, received {type(index)} instead")

    if index < 0:
      raise ValueError("Index cannot be negative")

    img_path = os.path.join(self.img_dir,self.image_paths[index])
    image = Image.open(img_path).convert("RGB")
    caption = self.captions[index]

    if self.transform:
      image = self.transform(image)

    return image, caption

    