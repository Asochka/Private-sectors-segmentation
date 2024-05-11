import torch
import torchvision
from typing import Tuple, List, Dict
from PIL import Image
import src.create_mask as mask
import os


def image_preprocess(image: Image.Image) -> torch.Tensor:
  return torchvision.transforms.ToTensor()(pic=image)


def get_dataset_subimage_tensor(
    subimage: Image.Image,
    subimage_labeled: Image.Image,
    classes: Dict[Tuple[int, int, int], Tuple[int, str]],
    dtype: torch.FloatType = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

  subimage_tensor = image_preprocess(image=subimage)
  subimage_mask_tensor = torch.tensor(
      data=mask.get_image_mask_from_labeled(
          image_labeled=subimage_labeled,
          classes=classes
      ),
      dtype=dtype
  )

  return subimage_tensor, subimage_mask_tensor
