import torch
import numpy as np
from typing import List, Dict, Any
from IPython.display import clear_output
from src.metrics import metric_pixel_accuracy, metric_iou


def test(
    model: torch.nn.Module,
    device: torch.DeviceObjType,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: Any,
) -> Dict[str, List[float]]:

  history_metrics = {
      'pixel_accuracy': list(),
      'iou': list()
  }

  for b, data in enumerate(test_dataloader, start=1):
    subimage_tensor, subimage_mask_tensor = data

    if device.type == 'cuda':
      subimage_tensor = subimage_tensor.to(device)
      subimage_mask_tensor = subimage_mask_tensor.to(device)

    with torch.no_grad():
      output = model(subimage_tensor)

    pixel_accuracy = metric_pixel_accuracy(output['out'], subimage_mask_tensor)
    iou = metric_iou(output['out'], subimage_mask_tensor)

    history_metrics['pixel_accuracy'].append(pixel_accuracy)
    history_metrics['iou'].append(iou)

    clear_output()
    print(
        'Batch: {}. median Pixel Accuracy: {:.3f} | median IoU: {:.3f}'.format(
            b,
            np.median(a=history_metrics['pixel_accuracy']),
            np.median(a=history_metrics['iou'])
        )
    )

    del subimage_tensor, subimage_mask_tensor, output
    if device.type == 'cuda':
      torch.cuda.empty_cache()

  return history_metrics
