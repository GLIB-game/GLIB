import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms, utils
import torchvision.transforms as T
from PIL import Image
import numpy as np
import config as config
import pandas as pd

SQUEEZENET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SQUEEZENET_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
preprocess = transforms.Compose([
    T.ToTensor(),
    T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                std=SQUEEZENET_STD.tolist()),
])


class GameDataloader(torch.utils.data.Dataset):
    def __init__(self, path):
        self.img_list = pd.read_csv(path,header=None)

    def __getitem__(self, index):
        img_path = self.img_list.iloc[index][0]
        label = self.img_list.iloc[index][1]
        img_pil = Image.open(img_path)# Image.open(os.path.join("process",img_path))
        img_pil = img_pil.resize((config.INPUT_W, config.INPUT_H))

        return preprocess(img_pil), label

    def __len__(self):
        return len(self.img_list)