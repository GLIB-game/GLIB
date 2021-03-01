import pandas as pd
import cv2
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms, utils
import torchvision.transforms as T
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import config as config
from NNArch import CNN
from GameDataLoader import GameDataloader

torch.cuda.set_device(0)


game_type = "l12_plus_l17"
aug_type = "rule_RGB"
# code_plus_rule_randomRGB
# code
# rule_randomRGB
# rule_RGB
# code_plus_rule_RGB

base_path = "data/" + game_type + "img"
aug_name = game_type + "_" + aug_type +"_base"

if not os.path.exists(aug_name + "_models"):
    os.mkdir(aug_name + "_models")

# print(os.path.join(bb_path, game_type + "_error_" + error_type + "_plus_normal_train.csv"))
# print(os.path.join(bb_path, game_type + "_error_" + error_type + "_plus_normal_test.csv"))


def train():
    train_data = GameDataloader(os.path.join(base_path, game_type + "_error_" + aug_type + "_plus_normal_train.csv"))
    eval_data = GameDataloader(os.path.join(base_path, game_type + "_error_" + aug_type + "_plus_normal_test.csv"))
    # test_data = gameDataloader("/root/Anomaly-Classification/collect_imgs/" + game_type + "img/" + game_type + "_error_real_plus_normal_test.csv")
    train_loader = Data.DataLoader(dataset=train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    eval_loader = Data.DataLoader(dataset=eval_data, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
    # test_loader = Data.DataLoader(dataset=test_data, batch_size=30, shuffle=False)
    cnn = CNN()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=config.LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

    cnn = cnn.cuda()
    loss_func = loss_func.cuda()

    for epoch in range(config.EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            optimizer.zero_grad()  # clear gradients for this training step
            output = cnn(b_x)[0]

            loss = loss_func(output, b_y)  # cross entropy loss
            loss = loss.cpu()
            print("loss: {}".format(loss))

            loss.backward()
            optimizer.step()  # apply gradients

            if step % config.EVAL_STEP == 0:

                with torch.no_grad():
                    accuracy = []
                    pred_y_list = []
                    eval_y_list = []
                    for eval_step, (t_x, t_y) in enumerate(eval_loader):
                        t_x = t_x.cuda()

                        eval_output, last_layer = cnn(t_x)
                        eval_output = eval_output.cpu()

                        pred_y = torch.max(eval_output, 1)[1].data.numpy()
                        pred_y_list += list(pred_y)
                        eval_y_list += list(t_y)
                        accuracy += list(pred_y == t_y.data.numpy().astype(int))

                    eval_prec = precision_score(eval_y_list, pred_y_list, pos_label=1)
                    eval_recall = recall_score(eval_y_list, pred_y_list, pos_label=1)
                    eval_acc = accuracy_score(eval_y_list, pred_y_list)

                    print('eval precision: {}, recall: {}, accuracy: {}'.format(eval_prec, eval_recall, eval_acc))

            if step % config.SAVE_STEP == 0:
                if not os.path.exists(aug_name + "_models"):
                    os.mkdir(aug_name + "_models")
                torch.save(cnn, aug_name + "_models/{}_{}_{}.pkl".format(aug_name, epoch, step))


if __name__ == '__main__':
    train()