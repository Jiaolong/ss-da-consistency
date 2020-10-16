from PIL import Image
from torchvision import transforms

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class ForceFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)

def get_jig_train_transformers(args):
    size = args.img_transform.random_resize_crop.size
    scale = args.img_transform.random_resize_crop.scale
    img_tr = [transforms.RandomResizedCrop((int(size[0]), int(size[1])), (scale[0], scale[1]))]
    if args.img_transform.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.img_transform.random_horiz_flip))
    if args.img_transform.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(
            brightness=args.img_transform.jitter, contrast=args.img_transform.jitter,
            saturation=args.jitter, hue=min(0.5, args.jitter)))

    tile_tr = []
    if args.jig_transform.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.jig_transform.tile_random_grayscale))
    mean = args.normalize.mean
    std = args.normalize.std
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)

def get_train_transformers(args):
    size = args.img_transform.random_resize_crop.size
    scale = args.img_transform.random_resize_crop.scale
    img_tr = [transforms.RandomResizedCrop((int(size[0]), int(size[1])), (scale[0], scale[1]))]
    if args.img_transform.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.img_transform.random_horiz_flip))
    if args.img_transform.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(
            brightness=args.img_transform.jitter, contrast=args.img_transform.jitter,
            saturation=args.jitter, hue=min(0.5, args.jitter)))

    mean = args.normalize.mean
    std = args.normalize.std
    img_tr += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    return transforms.Compose(img_tr)

def get_val_transformer(args):
    size = args.img_transform.random_resize_crop.size
    mean = args.normalize.mean
    std = args.normalize.std
    img_tr = [transforms.Resize(tuple(size)), transforms.ToTensor(),
              transforms.Normalize(mean, std=std)]
    return transforms.Compose(img_tr)

def get_image_train_transformer(args, resize_size=256, crop_size=224): 

    mean = args.normalize.mean
    std = args.normalize.std
    normalize = transforms.Normalize(mean, std=std)
    return  transforms.Compose([
        ResizeImage(resize_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])

def get_image_test_transformer(args, resize_size=256, crop_size=224):
    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1
 
    mean = args.normalize.mean
    std = args.normalize.std
    normalize = transforms.Normalize(mean, std=std)

    return transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
        ])

def get_multi_crop_transformers(args, resize_size=256):
    
    crop_size = args.img_transform.random_resize_crop.size
    crop_size = int(crop_size[0])
    mean = args.normalize.mean
    std = args.normalize.std
    normalize = transforms.Normalize(mean, std=std)

    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1
    data_transforms = [
        transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_first, start_first),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_last, start_last),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_last, start_first),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_first, start_last),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
        ])
    ]
    return data_transforms
