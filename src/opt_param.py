# -*- coding:utf-8 -*-

import math
from scipy.optimize import differential_evolution, brute
import torch
import os
import numpy as np
import argparse
import json
from multiprocessing.dummy import Pool as ThreadPool



# bound for each parameter
T = None
RH = None
P = None
tau = None
d = None
l = None
Zeta = None
C = None

bound = None

device = None
metadata = None
output_path = None
fixed = None
model_list = None
optim_algo = None
Brute_N = None
DE_mutation = None
DE_recombination = None
DE_maxiter = None
DE_popsize = None
DE_strategy = None

def is_fixed():
    global fixed, bound, metadata
    for index in range(8):
        if(abs(bound[index][0] - bound[index][1]) < 1e-6*metadata['target_mean'][index]):
            fixed.append(True)
        else:
            fixed.append(False)


def get_model_pred(x, param):
    global fixed, metadata, device, model_list
    param_index = 0
    inputs = []
    unnormalized_input = []
    for index in range(8):
        if(fixed):
            inputs.append(((bound[index][0] + bound[index][1])/2-metadata["target_mean"][index])/metadata["target_std"][index])
            unnormalized_input.append((bound[index][0] + bound[index][1])/2)
        else:
            inputs.append((x[param_index]-metadata["target_mean"][index])/metadata["target_std"][index])
            unnormalized_input.append(x[param_index])
            param_index += 1
    for index in range(8):
        inputs.append((math.log(abs(unnormalized_input[index]))-metadata["target_mean"][8+index])/metadata["target_std"][8+index])
    model = model_list[param]
    inputs = torch.Tensor(inputs).unsqueeze(0).to(device)
    pred = model(inputs)[0].cpu().detach().numpy()
    return -1*pred
    

def optimize_parameters(index):
    global bound, fixed, output_path, optim_algo, Brute_N, DE_mutation, DE_recombination, DE_maxiter, DE_popsize, DE_strategy
    param_bounds = []
    for i in range(8):
        if(not fixed[i]):
            param_bounds.append(bound[i])
    
    if(optim_algo == "Brute"):
        args = (index, )
        parameters = brute(get_model_pred, param_bounds, args = args, Ns=Brute_N, full_output=True)
        np.save(os.path.join(output_path, "optim_param_" + str(index) + ".npy"), parameters)
    else:
        args = (index, )
        parameters = differential_evolution(get_model_pred, 
                                            param_bounds, 
                                            args = args, 
                                            strategy=DE_strategy, 
                                            mutation=DE_mutation, 
                                            recombination=DE_recombination, 
                                            maxiter=DE_maxiter, 
                                            popsize=DE_popsize)
        np.save(os.path.join(output_path, "optim_param_" + str(index) + ".npy"), [parameters.fun, parameters.x])

def get_optimal_parameters(models, config):
    global model_list, T, RH, P, tau, d, l, Zeta, C, bound, device, output_path, metadata, fixed, optim_algo, Brute_N, DE_mutation, DE_recombination, DE_maxiter, DE_popsize, DE_strategy
    T = config["T"]
    RH = config["RH"]
    P = config["P"]
    tau = config["tau"]
    d = config["d"]
    l = config["l"]
    Zeta = config["Zeta"]
    C = config["C"]
    bound = [l, Zeta, C, tau, T, P, RH, d]
    fixed = []

    device = config["device"]
    metadata = config["metadata"]
    output_path = config["output_path"]
    optim_algo = config["optim_algo"]
    Brute_N = config["Brute_N"]
    DE_mutation = config["DE_mutation"]
    DE_recombination = config["DE_recombination"]
    DE_maxiter = config["DE_maxiter"]
    DE_popsize = config["DE_popsize"]
    DE_strategy = config["DE_strategy"]

    is_fixed()
    model_list = models

    index = [i for i in range(len(models))]
    pool = ThreadPool()
    pool.map(optimize_parameters, index)
    pool.close()
    pool.join()