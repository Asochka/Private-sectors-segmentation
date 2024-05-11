import src.tensor_func as tensor_func
from PIL import Image
from typing import Tuple, List, Dict
import torch


class Dataset(torch.utils.data.Dataset):
  def __init__(self,
      dataset_subimages_id: List[str],
      classes: Dict[Tuple[int, int, int], Tuple[int, str]],
  ):
    self.dataset_subimages_id = dataset_subimages_id
    self.classes = classes

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    subimage = Image.open(fp=f'dataset/originals/{self.dataset_subimages_id[idx]}.tif')
    subimage_labeled = Image.open(fp=f'dataset/labeleds/{self.dataset_subimages_id[idx]}_labeled.tif')
    subimage_tensor, subimage_mask_tensor = tensor_func.get_dataset_subimage_tensor(
        subimage=subimage,
        subimage_labeled=subimage_labeled,
        classes=self.classes
    )
    return subimage_tensor, subimage_mask_tensor

  def __len__(self) -> int:
    return len(self.dataset_subimages_id)
