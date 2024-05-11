import torch
from typing import List, Dict, Any
from IPython.display import clear_output
from src.metrics import metric_pixel_accuracy, metric_iou


def train(    
	model: torch.nn.Module,
    device: torch.DeviceObjType,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: Any,
    optim_fn: Any,
    epochs: int
) -> Dict[str, List[float]]:

  history_metrics = {
      'loss': list(),
      'pixel_accuracy': list(),
      'iou': list()
  }

  for e in range(1, epochs + 1):
    for b, data in enumerate(train_dataloader, start=1):
      subimage_tensor, subimage_mask_tensor = data

      if device.type == 'cuda':
        subimage_tensor = subimage_tensor.to(device)
        subimage_mask_tensor = subimage_mask_tensor.to(device)

      optim_fn.zero_grad()
      output = model(subimage_tensor)
      loss = loss_fn(output['out'], subimage_mask_tensor)
      loss.backward()
      optim_fn.step()

      loss_item = loss.item()
      pixel_accuracy = metric_pixel_accuracy(output['out'], subimage_mask_tensor)
      iou = metric_iou(output['out'], subimage_mask_tensor)

      history_metrics['loss'].append(loss_item)
      history_metrics['pixel_accuracy'].append(pixel_accuracy)
      history_metrics['iou'].append(iou)
      
      clear_output()
      print(
          'Epoch: {}. Batch: {}. Loss: {:.3f} | Pixel Accuracy: {:.3f} | IoU: {:.3f}'.format(
              e, b,
              loss, pixel_accuracy, iou
          )
      )

      del subimage_tensor, subimage_mask_tensor, output, loss
      if device.type == 'cuda':
        torch.cuda.empty_cache()

  return history_metrics
