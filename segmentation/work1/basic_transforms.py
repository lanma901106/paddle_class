import cv2
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Normalize(object):
    def __init__(self, mean_val, std_val, val_scale=1):
        # set val_scale = 1 if mean and std are in range (0,1)
        # set val_scale to other value, if mean and std are in range (0,255)
        self.mean = np.array(mean_val, dtype=np.float32)
        self.std = np.array(std_val, dtype=np.float32)
        self.val_scale = 1/255.0 if val_scale==1 else 1
    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        image = image * self.val_scale
        image = image - self.mean
        image = image * (1 / self.std)
        return image, label


class ConvertDataType(object):
    def __call__(self, image, label=None):
        if label is not None:
            label = label.astype(np.int64)
        return image.astype(np.float32), label


class Pad(object):
    def __init__(self, size, ignore_label=255, mean_val=0, val_scale=1):
        # set val_scale to 1 if mean_val is in range (0, 1)
        # set val_scale to 255 if mean_val is in range (0, 255)
        factor = 255 if val_scale == 1 else 1

        self.size = size
        self.ignore_label = ignore_label
        self.mean_val=mean_val
        # from 0-1 to 0-255
        if isinstance(self.mean_val, (tuple,list)):
            self.mean_val = [int(x* factor) for x in self.mean_val]
        else:
            self.mean_val = int(self.mean_val * factor)


    def __call__(self, image, label=None):
        h, w, c = image.shape
        pad_h = max(self.size - h, 0)
        pad_w = max(self.size - w, 0)

        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:

            image = cv2.copyMakeBorder(image,
                                       top=pad_h_half,
                                       left=pad_w_half,
                                       bottom=pad_h - pad_h_half,
                                       right=pad_w - pad_w_half,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=self.mean_val)
            if label is not None:
                label = cv2.copyMakeBorder(label,
                                           top=pad_h_half,
                                           left=pad_w_half,
                                           bottom=pad_h - pad_h_half,
                                           right=pad_w - pad_w_half,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
        return image, label


# TODO
class CenterCrop(object):


# TODO
class Resize(object):


# TODO
class RandomFlip(object):


# TODO
class RandomCrop(object):


# TODO
class Scale(object):


# TODO
class RandomScale(object):





def main():
   image = cv2.imread('./dummy_data/JPEGImages/2008_000064.jpg')
    label = cv2.imread('./dummy_data/GroundTruth_trainval_png/2008_000064.png')

    # TODO: crop_size
    # TODO: Transform: RandomSacle, RandomFlip, Pad, RandomCrop

    for i in range(10):
        # TODO: call transform
        # TODO: save image

if __name__ == "__main__":
    main()
