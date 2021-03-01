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
import config as config
from NNArch import CNN
from GameDataLoader import GameDataloader
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


game_type = "l12_plus_l17"
aug_type = "realbug"
# code_plus_rule_randomRGB iml
# rule_RGB intelligenttest
# code_plus_rule_RGB xunlianrenwu
# rule_randomRGB  l12_rl
# code   intelligenttest2
# realbug intelligenttest


aug_name = game_type + "_" + aug_type + "_base"
base_path = "/root/Anomaly-Classification/collect_imgs/" + game_type + "img"

test_data = GameDataloader(
    "/root/Anomaly-Classification/collect_imgs/" + game_type + "img/" + game_type + "_error_real_plus_normal_test.csv")

model_res = []

def test():
    model_name = "model.pkl"

    cnn = torch.load(model_name)
    cnn = cnn.cuda()
    cnn.eval()

    with torch.no_grad():
        accuracy = []
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
            accuracy += list(pred_y == t_y.data.numpy().astype(int))

        test_prec = precision_score(test_y_list, pred_y_list, pos_label=1)
        test_recall = recall_score(test_y_list, pred_y_list, pos_label=1)
        test_acc = accuracy_score(test_y_list, pred_y_list)
        print('test precision: {}'.format(test_prec))
        print('test recall: {}'.format(test_recall))
        print('test accuracy: {}'.format(test_acc))


if __name__ == '__main__':
    test()



