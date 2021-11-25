from ser.transforms import transforms, normalize, flip
from ser.data import train_dataloader, val_dataloader, test_dataloader

def _select_test_image(label, flip_image):
    # TODO we should be able to switch between these abstractions without
    #   having to change any code.
    #   make it happen!
    if flip_image:
        ts = [normalize, flip]
    else:
        ts = [normalize]
    dataloader = test_dataloader(1, transforms(*ts))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return images