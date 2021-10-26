import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class BaselineDewarpDataset(Dataset):
    def __init__(self, path, transforms=None, train=True):
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.path = path
        files = os.listdir(path)
        if self.train:
            self.images = [img_name for img_name in tqdm(files) if ((img_name + ".npy.npz" in files) and (img_name + "mask.png" in files))]
            self.masks = [img_name + "mask.png" for img_name in self.images]
            self.flows = [img_name + ".npy.npz" for img_name in self.images]
        else:
            self.images = [img_name for img_name in tqdm(files) if ((img_name + ".npy.npz" in files) and (img_name + "mask.png" in files))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.train:
            mask = cv2.imread(os.path.join(self.path, self.masks[index]))[:, :, 0]  // 255
            flow = next(iter(np.load(os.path.join(self.path, self.flows[index])).values()))
            img = cv2.imread(os.path.join(self.path, self.images[index]))

            if self.transforms is not None:
                img = self.transforms(img)
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).float()
            flow = torch.from_numpy(flow.transpose(2, 0, 1)).float()
            return img, mask, flow
        else:
            img = cv2.imread(os.path.join(self.path, self.images[index]))
            if self.transforms is not None:
                img = self.transforms(img)
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            return img


