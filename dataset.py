import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

image_size = 768


def image_to_tensor(image, mode="bgr"):  # image mode
    if mode == "bgr":
        image = image[:, :, ::-1]
    x = image
    x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x


def mask_to_tensor(mask):
    x = mask
    x = torch.tensor(x, dtype=torch.float)
    return x


tensor_list = ["mask", "image", "organ"]


class HapDataset(Dataset):
    def __init__(self, df, config, augment=None):

        self.df = df
        self.augment = augment
        self.length = len(self.df)
        ids = pd.read_csv(config.LABELS).id.astype(str).values
        self.config = config
        self.fnames = [
            fname for fname in os.listdir(config.TRAIN) if fname.split("_")[0] in ids
        ]
        self.organ_to_label = {
            "kidney": 0,
            "prostate": 1,
            "largeintestine": 2,
            "spleen": 3,
            "lung": 4,
        }

    def __str__(self):
        string = ""
        string += "\tlen = %d\n" % len(self)

        d = self.df.organ.value_counts().to_dict()
        for k in ["kidney", "prostate", "largeintestine", "spleen", "lung"]:
            string += "%24s %3d (%0.3f) \n" % (
                k,
                d.get(k, 0),
                d.get(k, 0) / len(self.df),
            )
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        fname = self.fnames[index]
        d = self.df.iloc[index]
        organ = self.organ_to_label[d.organ]

        image = cv2.cvtColor(
            cv2.imread(os.path.join(self.config.TRAIN, fname)), cv2.COLOR_BGR2RGB
        )
        mask = cv2.imread(os.path.join(self.config.MASKS, fname), cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32) / 255

        s = d.pixel_size / 0.4 * (image_size / 3000)

        if self.augment is not None:
            image, mask = self.augment(image, mask, organ)

        r = {}
        r["index"] = index
        r["id"] = fname
        r["organ"] = torch.tensor([organ], dtype=torch.long)
        r["image"] = image_to_tensor(image)
        r["mask"] = mask_to_tensor(mask > 0.5)
        return r
