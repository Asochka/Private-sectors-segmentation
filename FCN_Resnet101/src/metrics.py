import torch


def metric_pixel_accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
) -> float:

  y_pred_argmax = y_pred.argmax(dim=1)
  y_true_argmax = y_true.argmax(dim=1)

  correct_pixels = (y_pred_argmax == y_true_argmax).count_nonzero()
  uncorrect_pixels = (y_pred_argmax != y_true_argmax).count_nonzero()
  result = (correct_pixels / (correct_pixels + uncorrect_pixels)).item()

  return result


def metric_iou(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
) -> float:

  y_pred_hot = y_pred >= 0.51

  intersection = torch.logical_and(y_pred_hot, y_true).count_nonzero()
  union = torch.logical_or(y_pred_hot, y_true).count_nonzero()
  result = (intersection / union).item()

  return result
