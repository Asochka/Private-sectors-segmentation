from PIL import Image


def predict_png(model, img):
    results = model.predict(img)
    result = results[0]
    masks = result.masks
    if masks[0]:
        mask = masks[0].data[0].numpy()
        for i in range(1, len(masks)):
            mask += masks[i].data[0].numpy()
    mask_img = Image.fromarray(mask,"I")
    return mask_img
