# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import os
from models_def import target_predictor, source_predictor
import warnings
import json
import collections
from dataloader import Dataloader

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Transfer learning enhanced water-enabled electricity generation in highly oriented two-dimensional graphene oxide nanochannels.')

parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'optimize'], help="Train model, test accuracy or predict the optimal parameter combination.")
parser.add_argument('--gpu_id', type=int, default=-1, help="GPU device id used.")
parser.add_argument('--model', type=str, default='target', choices=['source', 'target'], help="Source model or opt-model.")
parser.add_argument('--n', type=int, default=2, help="number of base models.")
parser.add_argument('--optim', type=str, default='Adam', choices=['SGD', 'Adam'], help="Use SGD optimizer or Adam optimizer.")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
parser.add_argument('--norm_lr', type=float, default=1e-4, help="NormLayer learning rate.")
parser.add_argument('--finetune_lr', type=float, nargs='+', default=[1e-6,1e-6,3e-6,5e-6], help="Learning rate of fine-tuned layers.")
parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay.")
parser.add_argument('--gamma', type=float, default=1e-4, help="Multiplicative factor of learning rate decay.")
parser.add_argument('--power', type=float, default=0.75, help="Exponential of learning rate decay.")
parser.add_argument('--s_path', type=str, default='../data/source', help="The path for source dataset (streaming potential dataset).")
parser.add_argument('--t_path', type=str, default='../data/target', help="The path for target dataset (2D-WEG dataset).")
parser.add_argument('--o_path', type=str, default='../outputs', help="The path for saving outputs (e.g. predicted optimal parameter combinations and loss function trace).")
parser.add_argument('--m_path', type=str, default='../saved_models', help="The path for saving checkpoints of source model and opt-model.")
parser.add_argument('--p_path', type=str, default='../pic', help="The path for saving pictures.")
parser.add_argument('--batch_size', type=int, default = 16, metavar='N', help="The batch size.")
parser.add_argument('--epoches', type=int, default=500, help="The number of epochs for training")
parser.add_argument('--loss_interval', type=int, default=1000, help="The interval for output training loss.")
parser.add_argument('--valid_interval', type=int, default=200, help="The interval for testing validation loss.")
parser.add_argument('--trans_num', type=int, default=4, help="The number of layers transferred from the source model.")
parser.add_argument('--name', type=str, default='_best.pt', help="The base name of the source model.")
parser.add_argument('--target_name', type=str, default='_target_v.pt', help="The base name of the opt-model.")
parser.add_argument('--source_layers', type=list, default=[[16,16],[64,64],[64,128],[64,128],[16,64],[2,2]])
parser.add_argument('--new_layers', type=list, default=[[32,128],[16,32],[1,1]])
parser.add_argument('--load', action='store_true', help="Whether to transfer layers from the source model.")
parser.add_argument('--add_noise', action='store_true', help="Whether to add noise.")
parser.add_argument('--noise_std', type=float, default=5e-3, help="The std deviation of the noise.")
parser.add_argument('--T', type=float, nargs='+', default=[0, 100], help="The range of temperature values for parameter optimization. No optimization for temperature when equal.")
parser.add_argument('--RH', type=float, nargs='+', default=[0, 100], help="The range of relative humidity for parameter optimization. No optimization for relative humidity when equal.")
parser.add_argument('--P', type=float, nargs='+', default=[0, 1000], help="The range of pressure for parameter optimization. No optimization for pressure when equal.")
parser.add_argument('--tau', type=float, nargs='+', default=[1, 4], help="The range of structural tortuosity for parameter optimization. No optimization for structural tortuosity when equal.")
parser.add_argument('--d', type=float, nargs='+', default=[10, 1000], help="The range of channel spacing for parameter optimization. No optimization for channel spacing when equal.")
parser.add_argument('--l', type=float, nargs='+', default=[0, 22], help="The range of device length for parameter optimization. No optimization for device length when equal.")
parser.add_argument('--Zeta', type=float, nargs='+', default=[-80, 0], help="The range of Zeta potential for parameter optimization. No optimization for Zeta potential when equal.")
parser.add_argument('--C', type=float, nargs='+', default=[0, 1], help="The range of ion concentration for parameter optimization. No optimization for ion concentration when equal.")
parser.add_argument('--optim_algo', type=str, default='DE', choices=['DE', 'Brute'], 
                                    help="Use differential evolution algorithm or brute force algorithm to explore optimal parameter combinations.")
parser.add_argument('--Brute_N', type=int, default=30, help="Number of grids for the brute force algorithm.")
parser.add_argument('--DE_mutation', type=float, default=1.0, help="The differential weight F of differential evolution.")
parser.add_argument('--DE_recombination', type=float, default=0.7, help="The crossover probability of differential evolution.")
parser.add_argument('--DE_maxiter', type=int, default=1000, help="The maximum number of generations of differential evolution.")
parser.add_argument('--DE_popsize', type=int, default=100, help="The total population size of differential evolution.")
parser.add_argument('--DE_strategy', type=str, default='randtobest1bin', choices=['best1bin','best1exp','rand1exp','randtobest1exp','currenttobest1exp','best2exp',
                                                                                  'rand2exp','randtobest1bin','currenttobest1bin','best2bin','rand2bin','rand1bin'])

args = parser.parse_args()

if (torch.cuda.is_available() and (args.gpu_id > -1)):
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)   
else:
    device = torch.device("cpu")

Stats = collections.namedtuple('Stats', ['mean', 'std'])
cast = lambda v: np.array(v, dtype=np.float32)

def _read_metadata(data_path):
    assert os.path.exists(os.path.join(data_path, 'metadata.json')), "metadata file does not exist!"
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

def _get_number_list(str_list):
    return_list = True
    for nums in str_list:
        if(not isinstance(nums, list)):
            return_list = False
        else:
            for num in nums:
                if(not isinstance(num, int)):
                    return_list = False
    if(return_list):
        return str_list
    nums = []
    num_i = []
    number = ''
    last_char = ''
    digital = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', 'e']
    for index in range(len(str_list)):
        if(str_list[index] == '['):
            num_i = []
            last_char = '['
        elif(str_list[index] == ']'):
            last_char = ']'
            if(len(number) > 0):
                num_i.append(number)
                number = ''
            if(len(num_i) > 0):
                nums.append(num_i)
                num_i = []
        elif(str_list[index] == ','):
            if(last_char in digital):
                num_i.append(number)
                number = ''
            last_char = ','
        elif(str_list[index] in digital):
            last_char = str_list[index]
            number += str_list[index]
        else:
            last_char = str_list[index]
    num_list = []
    for num_i in nums:
        num_list.append( list(map(int, num_i)) )
    return num_list

def train(args):
    # train config
    config = {}
    config["device"] = device
    config["epoches"] = args.epoches
    config["interval"] = args.loss_interval
    config["valid_interval"] = args.valid_interval
    # output path 
    if (not os.path.exists(args.o_path)):
        os.makedirs(args.o_path)
    if not os.path.exists(os.path.join(args.o_path, "log/")):
        os.makedirs(os.path.join(args.o_path, "log/"))
    # model path 
    if not os.path.exists(args.m_path):
        os.makedirs(args.m_path)
    
    
    config["name"] = args.name
    config["pic"] = args.p_path
    config["model_path"] = args.m_path
    config["log_path"] = os.path.join(args.o_path, "log/")
    
    if args.optim == "SGD":
        config["optim"] = {"algo":torch.optim.SGD, "params":{'lr':args.lr, "momentum":0.9, \
                        "weight_decay":args.weight_decay, "nesterov":True}}
    else:
        config["optim"] = {"algo":torch.optim.Adam, "params":{"lr" : args.lr,\
            "betas" : (0.9, 0.999), "weight_decay" : args.weight_decay, "eps" : 1E-9}}
    
    config["lr"] = args.lr
    config["finetune_lr"] = args.finetune_lr
    config["gamma"] = args.gamma
    config["power"] = args.power
    config["weight_decay"] = args.weight_decay
    config["noise_std"] = args.noise_std

    config["load"] = args.load
    config["batch_size"] = args.batch_size
    config["add_noise"] = args.add_noise

    if args.model == "source":
        assert os.path.exists(os.path.join(args.s_path, "data.npy")), "Source domain dataset does not exist!"
        config["layers"] = args.source_layers
        metadata = _read_metadata(args.s_path)
        source_stats = Stats(
            cast(metadata['source_mean']),
            cast(metadata['source_std']))
        config["dataloader"] = Dataloader(data_path=os.path.join(args.s_path, "data.npy"), normalization_stats=source_stats, batch_size=args.batch_size, split=True)
        predictor = source_predictor(args.n, config)
    else:
        assert os.path.exists(os.path.join(args.s_path, "data.npy")), "Target domain dataset does not exist!"
        config["target_name"] = args.target_name
        config["norm_lr"] = args.norm_lr
        metadata = _read_metadata(args.t_path)
        target_stats = Stats(
            cast(metadata['target_mean']),
            cast(metadata['target_std']))
        config["dataloader"] = Dataloader(data_path=os.path.join(args.t_path, "data.npy"), normalization_stats=target_stats, batch_size=args.batch_size, split=True)
        predictor = target_predictor(args.n, args.trans_num, args.new_layers, config)
            
    predictor.train(config)
    
def test(args):
    # test config
    config = {}
    config["device"] = device
    # output path 
    if (not os.path.exists(args.o_path)):
        os.makedirs(args.o_path)
    # model path 
    if not os.path.exists(args.m_path):
        os.makedirs(args.m_path)
    
    config["pic"] = args.p_path
    config["model_path"] = args.m_path
    config["output_path"] = args.o_path

    config["name"] = args.name
    config["batch_size"] = args.batch_size

    if args.model == "source":
        config["layers"] = args.source_layers
        metadata = _read_metadata(args.s_path)
        source_stats = Stats(
            cast(metadata['source_mean']),
            cast(metadata['source_std']))
        config["dataloader"] = Dataloader(data_path=os.path.join(args.s_path, "test.npy"), normalization_stats=source_stats, batch_size=args.batch_size, split=False)
        predictor = source_predictor(args.n, config)
    else:
        metadata = _read_metadata(args.t_path)
        target_stats = Stats(
            cast(metadata['target_mean']),
            cast(metadata['target_std']))
        config["target_name"] = args.target_name
        config["load"] = args.load
        config["dataloader"] = Dataloader(data_path=os.path.join(args.t_path, "test.npy"), normalization_stats=target_stats, batch_size=args.batch_size, split=False)
        predictor = target_predictor(args.n, args.trans_num, args.new_layers, config)
            
    predictor.test()

def optimize(args):
    # optimize config
    config = {}
    config["device"] = device
    # output path 
    if (not os.path.exists(args.o_path)):
        os.makedirs(args.o_path)
    # model path 
    if not os.path.exists(args.m_path):
        os.makedirs(args.m_path)

    config["model_path"] = args.m_path
    config["output_path"] = args.o_path

    config["name"] = args.name
    config["target_name"] = args.target_name
    config["load"] = args.load
    config["batch_size"] = args.batch_size

    metadata = _read_metadata(args.t_path)
    config["metadata"] = metadata
    target_stats = Stats(
        cast(metadata['target_mean']),
        cast(metadata['target_std']))
    
    config["T"] = args.T
    config["RH"] = args.RH
    config["P"] = args.P
    config["tau"] = args.tau
    config["d"] = args.d
    config["l"] = args.l
    config["Zeta"] = args.Zeta
    config["C"] = args.C
    config["optim_algo"] = args.optim_algo
    config["Brute_N"] = args.Brute_N
    config["DE_mutation"] = args.DE_mutation
    config["DE_recombination"] = args.DE_recombination
    config["DE_maxiter"] = args.DE_maxiter
    config["DE_popsize"] = args.DE_popsize
    config["DE_strategy"] = args.DE_strategy

    predictor = target_predictor(args.n, args.trans_num, args.new_layers, config)
    predictor.optimal_param()


if __name__ == "__main__":
    args.source_layers = _get_number_list(args.source_layers)
    args.new_layers = _get_number_list(args.new_layers)
    if args.mode == "train":
        train(args)
    if args.mode == "test":
        test(args)
    if args.mode == "optimize":
        optimize(args)