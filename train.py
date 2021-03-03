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
import argparse
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import config as config
from NNArch import CNN
from GameDataLoader import GameDataloader

torch.cuda.set_device(0)


def train(train_data_path, eval_data_path, aug_type, model_path=None):
    # load DataSet
    train_data = GameDataloader(train_data_path)
    eval_data = GameDataloader(eval_data_path)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    eval_loader = Data.DataLoader(dataset=eval_data, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)

    # init model
    cnn = CNN()
    # load from checkpoint
    if model_path is not None:
        cnn = torch.load(model_path)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=config.LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # define loss

    # Use GPU
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

            # apply gradients
            loss.backward()
            optimizer.step()

            # Eval model
            if step % config.EVAL_STEP == 0:
                with torch.no_grad():
                    pred_y_list = []
                    eval_y_list = []
                    for eval_step, (t_x, t_y) in enumerate(eval_loader):
                        t_x = t_x.cuda()

                        eval_output, last_layer = cnn(t_x)
                        eval_output = eval_output.cpu()

                        pred_y = torch.max(eval_output, 1)[1].data.numpy()
                        pred_y_list += list(pred_y)
                        eval_y_list += list(t_y)

                    eval_prec = precision_score(eval_y_list, pred_y_list, pos_label=1)
                    eval_recall = recall_score(eval_y_list, pred_y_list, pos_label=1)
                    eval_acc = accuracy_score(eval_y_list, pred_y_list)

                    print('eval precision: {}, recall: {}, accuracy: {}'.format(eval_prec, eval_recall, eval_acc))

            if step % config.SAVE_STEP == 0:
                model_save_path = os.path.join("model", aug_type)
                if not os.path.exists(model_save_path):
                    os.mkdir(model_save_path)
                torch.save(cnn, os.path.join(model_save_path, "{}_{}_{}.pkl".format(aug_type, epoch, step)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help="model path", type=str)
    parser.add_argument("-t", "--train_data", help="train data path", type=str, required=True)
    parser.add_argument("-e", "--eval_data", help="eval data path", type=str, required=True)
    parser.add_argument("-a", "--augType", help="augmentation Type",type=str,
                        required=True, choices=["Base", "Rule(F)", "Rule(R)", "Code", "Code_Rule(F)", "Code_Rule(R)"])

    args = parser.parse_args()

    # print(args.train_data, args.eval_data, args.augType)

    train(args.train_data, args.eval_data, args.augType, args.model_path)