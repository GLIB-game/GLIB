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
import argparse
import config as config
from NNArch import CNN
from GameDataLoader import GameDataloader
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def test(model_path, test_data_path):
    test_data = GameDataloader(test_data_path)

    cnn = torch.load(model_path)
    cnn = cnn.cuda()
    cnn.eval()

    with torch.no_grad():
        pred_y_list = []
        test_loader = Data.DataLoader(dataset=test_data, batch_size=config.TEST_BATCH_SIZE, shuffle=False)
        test_y_list = []
        print("start test")
        for test_step, (t_x, t_y) in enumerate(test_loader):
            t_x = t_x.cuda()
            test_output, last_layer = cnn(t_x)
            test_output = test_output.cpu()
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            pred_y_list += list(pred_y)
            test_y_list += list(t_y)

        test_prec = precision_score(test_y_list, pred_y_list, pos_label=1)
        test_recall = recall_score(test_y_list, pred_y_list, pos_label=1)
        test_acc = accuracy_score(test_y_list, pred_y_list)
        print('test precision: {}'.format(test_prec))
        print('test recall: {}'.format(test_recall))
        print('test accuracy: {}'.format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="model path", type=str)
    parser.add_argument("-t", "--test_data", help="test data path", type=str, required=True)

    args = parser.parse_args()

    test(args.model, args.test_data)



