import torchvision.transforms as transforms
import torch
from PIL import Image


class CustomRescalse(object):
    def __init__(self, size): # 960, 1024 (x, y)
        self.size = size

    def __call__(self, img):
        if img.size != self.size:
            x, y = img.size
            rel_x, rel_y = float(x) / float(self.size[0]), float(y) / float(self.size[1])
            if rel_x > rel_y:
                new_x = self.size[0]
                new_y = int(self.size[0]/x * y)
            else:
                new_y = self.size[1]
                new_x = int(self.size[1] / y * x)
            img = img.resize((new_x, new_y), Image.ANTIALIAS)
            new_img = Image.new("RGB", self.size)
            new_img.paste(img, ((self.size[0]-new_x)//2,
                                (self.size[1]-new_y)//2))
            img = new_img
        return img


def get_transform(config):
    if config:
        return transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
