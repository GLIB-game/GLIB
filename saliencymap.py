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

game_type = "l12_plus_l17"
aug_type = game_type + "_code_plus_rule_randomRGB"
model_idx = "_base_72_850.pkl"
# model_name = "/root/Anomaly-Classification/l12_plus_l17_code_plus_rule_RGB_base_models/l12_plus_l17_code_plus_rule_RGB_base_72_850.pkl"
# test_file_path = "/root/Anomaly-Classification/collect_imgs/realTestImg/test_label.csv"

model_name = "model/" + aug_type + "_models/" + aug_type +"_72_850.pkl"
test_file_path = "data/images/testDataSet/test_label.csv"


def calcu_saliency_map():
    # model_name = "models/" + aug_type + "_base_155_50.pkl"  # "models_cnn_fc/" + neg_name + "_80.pkl"
    # test_file_path = "data/realBugImg/test_label.csv"

    test_file = pd.read_csv(test_file_path, header=None)

    saliency_map_path = "saliency_map"
    if not os.path.exists(saliency_map_path):
        os.mkdir(saliency_map_path)

    cnn = torch.load(model_name)
    cnn = cnn.cuda()
    cnn.eval()

    for param in cnn.parameters():
        param.requires_gard = False

    test_data = GameDataloader(test_file_path)
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
                np.save(os.path.join(saliency_map_path, _img.split("/")[-1].split(".")[0] + ".npy"), sa)
                scale = 100
                cam = sa - sa.min()
                cam = cam / cam.max()
                heatmap = cv2.applyColorMap(np.uint8(255 * cam * scale), cv2.COLORMAP_HOT)

                cv2.imwrite(os.path.join(saliency_map_path, _img.split("/")[-1].split(".")[0] + ".jpg"),
                            heatmap)

                test_img_idx += 1


if __name__ == '__main__':
    calcu_saliency_map()