import numpy as np
from PIL import Image
from typing import Tuple, Dict


def get_image_mask_from_labeled(
    image_labeled: Image.Image,
    classes: Dict[Tuple[int, int, int], Tuple[int, str]]
) -> np.ndarray:

  image_mask = np.zeros(shape=(len(classes),image_labeled.size[0],image_labeled.size[1]))

  image_labeled_ndarray = np.array(object=image_labeled)
  for r in np.arange(stop=image_labeled_ndarray.shape[0]):
    for c in np.arange(stop=image_labeled_ndarray.shape[1]):
      class_rgb = tuple(image_labeled_ndarray[r][c])[:3]
      class_value = classes.get(class_rgb)
      if class_value != None:
        image_mask[class_value[0]][r][c] = 1.0
      else:
        image_mask[0][r][c] = 1.0

  return image_mask


def get_image_labeled_from_mask(
    image_mask: np.ndarray,
    classes_by_id: Dict[Tuple[int, int, int], Tuple[int, str]]
) -> Image.Image:

  image_labeled_ndarray = np.zeros(
      shape=(image_mask.shape[1],image_mask.shape[2],3),
      dtype=np.uint8
  )

  image_mask_hot = image_mask.argmax(axis=0)
  for r in np.arange(stop=image_mask_hot.shape[0]):
    for c in np.arange(stop=image_mask_hot.shape[1]):
      class_id = image_mask_hot[r][c]
      class_by_id_value = classes_by_id.get(class_id)
      image_labeled_ndarray[r][c] = np.array(object=class_by_id_value[0])

  image_labeled = Image.fromarray(obj=image_labeled_ndarray)

  return image_labeled
