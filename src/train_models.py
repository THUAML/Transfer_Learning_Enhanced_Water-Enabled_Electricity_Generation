# -*- coding: utf-8 -*-

import torch
import numpy as np
from loss import loss_F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from lr_schedule import lr_scheduler
from noise_utils import get_random_fluctuation
import os
#from torch.utils.tensorboard import SummaryWriter
    
def test(model, config):
    device = config["device"]
    add_noise = config["add_noise"]
    noise_std = config["noise_std"]
    model = model.to(device)
    criterion = torch.nn.MSELoss()#loss_F
    dataloader = config["dataloader"].test_loader
    model.eval()
    flag = True
    total_loss = 0
    total_num = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            if(add_noise):
                labels = get_random_fluctuation(labels, noise_std, device)
            pred = model(inputs)
            #loss = criterion(pred, labels, device)
            loss = criterion(pred, labels)
            loss = loss.cpu().detach().numpy()
            total_loss += loss * len(inputs)
            total_num += len(inputs)
    return total_loss/total_num

def train(model, config, index):
    print("Train the ", index, "st model.")
    device = config["device"]
    model = model.to(device)
    optimizer = config["optimizer"]
    criterion = torch.nn.MSELoss()#loss_F
    dataloader = config["dataloader"].train_loader
    log_path = config["log_path"]
    model_path = config["model_path"]
    lr = config["lr"]
    gamma = config["gamma"]
    power = config["power"]
    weight_decay = config["weight_decay"]
    noise_std = config["noise_std"]
    model_name = config["name"]
    interval = config["interval"]
    valid_interval = config["valid_interval"]
    #writer = SummaryWriter(config["log_path"])
    
    training_loss = []
    valid_loss = []
    training_loss_interval = []
    valid_iter = []
    loss_sum = []
    min_loss = 1E9
    min_epoch = 0
    iteration = 0
    tic = time.time()
    for epoch in range(config["epoches"]):
        model.train()
        for inputs, labels in dataloader:
            iteration += 1
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            labels = get_random_fluctuation(labels, noise_std, device)
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            loss = loss.cpu().detach().numpy()
            optimizer = lr_scheduler(optimizer, iteration, gamma=gamma, power=power, weight_decay=weight_decay)
            training_loss.append(loss)
            loss_sum.append(loss)
            #writer.add_scalar("Loss/Train Loss", loss, dataloader.posi)
            if (iteration % interval) == 0:
                toc = time.time()
                print("epoch: {:03d}, iter: {:07d}, training loss: {:.5f}, best_valid_loss: {:.5f}, time consuming: {:.1f} s".format(epoch, iteration, np.mean(loss_sum), min_loss, (toc-tic)))
                tic = time.time()
                training_loss_interval.append(np.mean(loss_sum))
                loss_sum = []
                np.save(os.path.join(log_path, str(index) + "_training_loss.npy"), training_loss)
                np.save(os.path.join(log_path, str(index) + "_training_loss_interval.npy"), training_loss_interval)
                
            if (iteration % valid_interval) == 0:
                val_loss = test(model, config)
                model.train()
                valid_loss.append(val_loss)
                valid_iter.append(iteration)
                np.save(os.path.join(log_path, str(index) + "_valid_loss.npy"), valid_loss)
                np.save(os.path.join(log_path, str(index) + "_valid_position.npy"), valid_iter)
                #writer.add_scalar("Loss/Val Loss", val_loss, epoch/config["interval"])
                if val_loss < min_loss:
                    min_loss = val_loss
                    min_epoch = epoch
                    torch.save(model, os.path.join(model_path, str(index) + model_name))
        
def predict(model, config, num):
    device = config["device"]
    model = model.to(device)
    criterion = torch.nn.MSELoss()#loss_F
    dataloader = config["dataloader"].test_loader
    model.eval()
    
    total_loss = 0
    total_num = 0
    model_pred = []
    for params, labels in dataloader:
        inputs, labels = params.to(device), labels.to(device)
        with torch.no_grad():
            #labels = get_random_fluctuation(labels, noise_std, device)
            pred = model(inputs)
            if(labels.shape[1] > 0):
                #loss = criterion(pred, labels, device)
                loss = criterion(pred, labels)
                loss = loss.cpu().detach().numpy()
                total_loss += loss * len(inputs)
                total_num += len(inputs)
            pred = pred.cpu().detach().numpy()
            params = np.concatenate((params, pred), axis = 1)
            model_pred.append(params)
    model_pred = np.concatenate(model_pred, axis = 0)
    if(labels.shape[1] > 0):
        np.save(os.path.join(config["output_path"], str(num) + "_prediction.npy"), [model_pred, total_loss/total_num])
        print("The MSE of the test set is", total_loss/total_num)
    else:
        np.save(os.path.join(config["output_path"], str(num) + "_prediction.npy"), model_pred)