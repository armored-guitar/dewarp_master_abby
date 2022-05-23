import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from PIL import Image
from libs.data.utils import get_transform

global_cv = 0
global_pil = 0

class CustomRescalsePIL(object):
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
            global global_cv
            if global_cv < 2:
                print(img.size)
            global_cv += 1
        return img


class CustomRescalseCV2(object):
    def __init__(self, size): # 960, 1024 (x, y)
        self.size = size

    def __call__(self, img):
        if img.shape[:-1] != self.size:
            x, y, _ = img.shape
            rel_x, rel_y = float(x) / float(self.size[0]), float(y) / float(self.size[1])
            new_img = np.zeros((1024, 960, 3))
            if rel_x > rel_y:
                new_x = self.size[0]
                new_y = int(self.size[0]/x * y)
                img = cv2.resize(img, (new_y, new_x))
                new_img[:, (self.size[1]-new_y)//2: -(self.size[1]-new_y)//2, :] = img
            else:
                new_y = self.size[1]
                new_x = int(self.size[1] / y * x)
                img = cv2.resize(img, (new_y, new_x))
                new_img[(self.size[0]-new_x)//2: -(self.size[0]-new_x)//2, :, :] = img
            img = new_img
            assert img.shape == (1024, 960, 3)
        return img


class BaselineDewarpDataset(Dataset):
    def __init__(self, path, transforms=None, train: bool =True,
                 test: bool = False, return_img: bool = False, return_name: bool = False, use_gt: bool = False):
        super().__init__()
        self.rescale_cv2 = CustomRescalseCV2((1024, 960))
        self.rescale_PIL = CustomRescalsePIL((960, 1024))
        self.train = train
        self.transforms = get_transform(transforms)
        print(self.transforms)
        self.path = path
        self.return_img = return_img
        self.return_name = return_name
        self.use_gt = use_gt
        files = os.listdir(path)
        if self.train:
            self.images = [img_name for img_name in tqdm(files) if ((img_name + ".npy.npz" in files)
                                                                    and (img_name + "mask.png" in files))]
            self.masks = [img_name + "mask.png" for img_name in self.images]
            self.flows = [img_name + ".npy.npz" for img_name in self.images]
            self.gt = [f"{img_name.split('.')[0]}_gt.{img_name.split('.')[1]}" for img_name in self.images
                       if f"{img_name.split('.')[0]}_gt.jpg" in files]
        else:
            self.gt = []
            if not test:
                self.images = [img_name for img_name in
                               tqdm(files) if ((img_name + ".npy.npz" in files) and (img_name + "mask.png" in files))]
            else:
                self.images = [img_name for img_name in
                               tqdm(files) if img_name.split(".")[-1] in ["png", "jpeg", "jpg"]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return_value = []
        if self.train:
            mask = cv2.imread(os.path.join(self.path, self.masks[index]))[:, :, 0] // 255
            flow = next(iter(np.load(os.path.join(self.path, self.flows[index])).values()))

            if self.transforms is not None:
                img = Image.open(os.path.join(self.path, self.images[index]))
                mod_img = self.transforms(img)
            else:
                img = cv2.imread(os.path.join(self.path, self.images[index]))
                mod_img = torch.from_numpy(img.transpose(2, 0, 1)).float()

            img = cv2.imread(os.path.join(self.path, self.images[index]))
            mask = torch.from_numpy(mask).float()
            flow = mask * torch.from_numpy(flow.transpose(2, 0, 1)).float()
            return_value.extend([mod_img, mask, flow])
        else:
            if self.transforms is not None:
                img = Image.open(os.path.join(self.path, self.images[index]))
                img = self.rescale_PIL(img)
                mod_img = self.transforms(img)
            else:
                img = cv2.imread(os.path.join(self.path, self.images[index]))
                img = self.rescale_cv2(img)
                mod_img = torch.from_numpy(img.transpose(2, 0, 1)).float()

            img = cv2.imread(os.path.join(self.path, self.images[index]))
            img = self.rescale_cv2(img)
            return_value.append(mod_img)

        if self.return_img:
            return_value.append(img)
            return_value.append(self.images[index])

        if len(self.gt) != 0 and self.use_gt:
            gt_image = cv2.imread(os.path.join(self.path, self.gt[index]))
            return_value.append(gt_image)

        if self.return_name:
            return_value.append(self.images[index].split("/")[-1])

        return tuple(return_value)


