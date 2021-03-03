import pandas as pd
import cv2
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import shutil
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from GameDataLoader import GameDataloader
import config as config
import argparse


def calcu_saliency_map(model_path, test_data_path):

    test_file = pd.read_csv(test_data_path, header=None)

    saliency_map_path = "saliency_map"
    if not os.path.exists(saliency_map_path):
        os.mkdir(saliency_map_path)

    cnn = torch.load(model_path)
    cnn = cnn.cuda()
    cnn.eval()

    for param in cnn.parameters():
        param.requires_gard = False

    test_data = GameDataloader(test_data_path)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=config.TEST_BATCH_SIZE, shuffle=False)

    test_img_idx = 0
    for test_step, (t_x, t_y) in enumerate(test_loader):
        t_x = t_x.cuda()
        t_x.requires_grad_()
        for param in cnn.parameters():
            param.requires_gard = False
        test_output, last_layer = cnn(t_x)
        test_output = test_output.cpu()

        show_saliency = True
        if show_saliency:
            score = test_output.gather(1, torch.max(test_output, 1)[1].data.view(-1, 1)).squeeze()

            for param in cnn.parameters():
                param.requires_gard = False
            score.backward(torch.FloatTensor([1.0] * score.shape[0]))
            saliency, _ = torch.max(t_x.grad.data.abs(), dim=1)

            for to, sa in zip(test_output, saliency):
                _img = test_file[0][test_img_idx]
                # print(_img)
                # np.save(os.path.join(saliency_map_path, _img.split("/")[-1].split(".")[0] + ".npy"), sa)
                scale = 100
                cam = sa - sa.min()
                cam = cam / cam.max()
                heatmap = cv2.applyColorMap(np.uint8(255 * cam * scale), cv2.COLORMAP_HOT)

                cv2.imwrite(os.path.join(saliency_map_path, _img.split("/")[-1].split(".")[0] + ".jpg"),
                            heatmap)

                test_img_idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="model path", type=str)
    parser.add_argument("-t", "--test_data", help="test data path", type=str, required=True)

    args = parser.parse_args()

    calcu_saliency_map(args.model, args.test_data)