import torch
from typing import List, Dict, Any
from IPython.display import clear_output
from src.metrics import metric_pixel_accuracy, metric_iou
import src.create_mask as mask
from PIL import Image
from tensor_func import image_preprocess


classes = {
    (0, 0, 0): (0, '__background__'),
    (255, 255, 255): (1, 'private_sectors'),
}

classes_by_id = dict()
for rgb, (id, name) in classes.items():
  classes_by_id[id] = (rgb, name)


def predict(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.DeviceObjType,
) -> Image.Image:

  image_tensor = image_preprocess(image=image)

  with torch.no_grad():
    output_image_mask = model(image_tensor.unsqueeze(0).to(device))['out'][0].cpu().numpy()

  predicted_image_labeled = mask.get_image_labeled_from_mask(
      image_mask=output_image_mask,
      classes_by_id=classes_by_id
  )

  return predicted_image_labeled
