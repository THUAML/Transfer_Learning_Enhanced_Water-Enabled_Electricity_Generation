# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import train_models
import random
import os
import opt_param

class SingleConnectionFunction(torch.autograd.Function):
    """Forward propagation and backpropagation of NormLayer."""
    @staticmethod
    def forward(ctx, inputs, weight, bias):
        """Forward propagation implementation of NormLayer.
        Args:
            inputs: Inputs of the opt-model with shape [batch_size, 16].
            weight: Learnable tensor of shape [16 ] representing the scale of normalization.
            bias: Learnable tensor of shape [16 ] representing the offset of normalization.
        Returns:
            Normalized inputs of the opt-model 
        """
        ctx.save_for_backward(inputs, weight, bias)
        output = torch.mul(inputs, weight)
        output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backpropagation implementation of NormLayer.
        Args:
            grad_output: The gradient of the normalized inputs with shape [16 ].
        Returns:
            Normalized inputs of the opt-model 
        """
        inputs, weight, bias = ctx.saved_tensors
        grad_input = torch.mul(grad_output, weight)
        grad_weight = torch.sum(torch.mul(grad_output, inputs), dim=0).unsqueeze(0)
        grad_bias = torch.sum(grad_output, dim=0)
        return grad_input, grad_weight, grad_bias
    
    
class SingleConnection(torch.nn.Module):
    """NormLayer performs element-wise normalization of each characteristic parameter."""
    def __init__(self, input_size):
        super(SingleConnection, self).__init__()
        self._input_size = input_size
        self._weight = torch.nn.Parameter(torch.ones(input_size))
        self._bias = torch.nn.Parameter(torch.Tensor(input_size))
    

    def forward(self, inputs):
        SingConnF = SingleConnectionFunction.apply
        return SingConnF(inputs, self._weight, self._bias)


class source_model(torch.nn.Module):
    """The source model.
    Args:
        sizes: Number of neurons in each layer.
    """
    def __init__(self, sizes):
        super(source_model, self).__init__()
        self._mlp = []
        for index in range(len(sizes)-2):
            self._mlp.append(nn.Linear(sizes[index], sizes[index + 1]))
            self._mlp.append(nn.ReLU())
        self._mlp.append(nn.Linear(sizes[-2], sizes[-1]))
        self._mlp = nn.Sequential(*self._mlp)
    
    def init_weights(self):
        for layers in self._mlp.modules():
            if isinstance(layers, nn.Linear): 
                nn.init.kaiming_normal_(layers.weight)
                nn.init.zeros_(layers.bias)

    def forward(self, x):
        x = self._mlp(x)
        return x


class BaseModel(torch.nn.Module):
    """The opt-model.
    Args:
        ori_model: The source model for weight transfer.
        trans_num: Number of layers transferred from the source model.
        sizes: Number of neurons of the new layers, i.e., each layer of Decoder, in the opt-model.
        _load: Whether to transfer weights from the source model.
    """
    def __init__(self, ori_model, trans_num, sizes, _load):
        super(BaseModel, self).__init__()
        self._norm = SingleConnection(16)
        self._trans_mlp = []
        self._ori_model = ori_model
        self._trans_num = trans_num
        self._load = _load
        num = 0
        for layers in self._ori_model.modules():
            if isinstance(layers, nn.Linear):
                num += 1
                if num > trans_num:
                    break
                self._trans_mlp.append(torch.nn.Linear(layers.in_features,
                                                       layers.out_features))
                self._trans_mlp[-1].weight = layers.weight
                self._trans_mlp[-1].bias = layers.bias
                last_size = layers.out_features
            elif isinstance(layers, nn.ReLU):
                self._trans_mlp.append(torch.nn.ReLU(layers.inplace))
            elif isinstance(layers, nn.LeakyReLU):
                self._trans_mlp.append(torch.nn.LeakyReLU(layers.negative_slope,
                                                          layers.inplace))
            elif isinstance(layers, nn.BatchNorm1d):
                self._trans_mlp.append(nn.BatchNorm1d(layers.num_features,
                                                      layers.eps,
                                                      layers.momentum, 
                                                      layers.affine, 
                                                      layers.track_running_stats))
                self._trans_mlp[-1].weight = layers.weight
                self._trans_mlp[-1].bias = layers.bias
            elif isinstance(layers, nn.LayerNorm):
                self._trans_mlp.append(nn.LayerNorm(layers.normalized_shape, 
                                                    layers.eps, 
                                                    layers.elementwise_affine))
                self._trans_mlp[-1].weight = layers.weight
                self._trans_mlp[-1].bias = layers.bias
            elif isinstance(layers, nn.Dropout):
                self._trans_mlp.append(nn.Dropout(layers.p,
                                                  layers.inplace))

        self._trans_mlp = torch.nn.Sequential(*self._trans_mlp)
        self._task_mlp = []
        sizes.insert(0, last_size)
        for index in range(len(sizes)-2):
            self._task_mlp.append(nn.Linear(sizes[index], sizes[index + 1]))
            self._task_mlp.append(nn.ReLU())
        self._task_mlp.append(nn.Linear(sizes[-2], sizes[-1]))
        self._task_mlp = nn.Sequential(*self._task_mlp)
        self.init_weights()

    def init_weights(self):
        for layers in self._task_mlp.modules():
            if isinstance(layers, nn.Linear): 
                nn.init.kaiming_normal_(layers.weight)
        if (not self._load):
            for layers in self._trans_mlp.modules():
                if isinstance(layers, nn.Linear): 
                    nn.init.kaiming_normal_(layers.weight)

    def forward(self, x):
        x = self._norm(x)
        x = self._trans_mlp(x)
        x = self._task_mlp(x)
        return x


        

class source_predictor:
    """Set of source models.
    Args:
        num: Number of source models trained.
        config: Some hyperparameters.
    """
    def __init__(self, num, config):
        self._num = num
        self._config = config
        self._model_list = []
        for index in range(num):
            layer_size = []
            for layer_num in range(len(config["layers"])):
                # Randomly select the number of neurons per layer within the given range.
                layer_size.append(random.randint(config["layers"][layer_num][0], config["layers"][layer_num][1]))
            model = source_model(layer_size)
            model.init_weights()
            self._model_list.append(model)
    
    def train(self, config):
        for index in range(len(self._model_list)):
            config["optimizer"] = config["optim"]["algo"](\
                                  [{'params': self._model_list[index].parameters()},],\
                                  **config["optim"]["params"])
            train_models.train(self._model_list[index], config, index)
    
    def load_models(self):
        self._model_list = []
        for index in range(self._num):
            model_name = os.path.join(self._config["model_path"], str(index) + self._config["name"])
            assert os.path.exists(model_name), "Training first!"
            model = torch.load(model_name, map_location=self._config["device"])
            self._model_list.append(model)

    def test(self):
        self.load_models()
        for index in range(self._num):
            model = self._model_list[index]
            train_models.predict(model, self._config, index)


class target_predictor:
    """Set of opt-models.
    Args:
        num: Number of opt-models trained.
        trans_num: Number of layers transferred from the source model.
        sizes: Range of the number of neurons in each layer of the Decoder.
        config: Some hyperparameters.
    """
    def __init__(self, num, trans_num, sizes, config):
        self._num = num
        self._trans_num = trans_num
        self._model_list = []
        self._config = config
        for i in range(num):
            model_name = os.path.join(config["model_path"], str(i) + config["name"])
            assert os.path.exists(model_name), "Train the source model first!"
            source_model_i = torch.load(model_name, map_location=config["device"])
            size_i = []
            for j in range(len(sizes)):
                # Randomly select the number of neurons per layer within the given range.
                size_i.append(random.randint(sizes[j][0], sizes[j][1]))
            self._model_list.append(BaseModel(source_model_i, trans_num, size_i, config["load"]))
    
    def train(self, config):
        config["name"] = config["target_name"]
        assert len(config["finetune_lr"]) == self._trans_num
        for index in range(len(self._model_list)):
            layer_posi = []
            fine_tune_lr = []
            for layer_num, layer in enumerate(self._model_list[index]._trans_mlp.modules()):
                if isinstance(layer, nn.Linear):
                    layer_posi.append(layer_num-1)
                    if(len(fine_tune_lr) < self._trans_num):
                        fine_tune_lr.append(0.01 * config["finetune_lr"][len(fine_tune_lr)])#config["lr"])
            layer_posi.append(layer_num)
            lr_dict = [{'params': self._model_list[index]._norm.parameters(), 'lr':config["norm_lr"]},
                       {'params': self._model_list[index]._task_mlp.parameters()}]
            for i in range(self._trans_num):
                lr_dict.append({'params': self._model_list[index]._trans_mlp[layer_posi[i] : layer_posi[i+1]].parameters(), 'lr': fine_tune_lr[i]})
            config["optimizer"] = config["optim"]["algo"](\
                lr_dict,\
                **config["optim"]["params"])
            train_models.train(self._model_list[index], config, index)
    
    def load_models(self):
        self._model_list = []
        for index in range(self._num):
            model_name = os.path.join(self._config["model_path"], str(index) + self._config["target_name"])
            assert os.path.exists(model_name), "Training first!"
            model = torch.load(model_name, map_location=self._config["device"])
            self._model_list.append(model)

    def test(self):
        self.load_models()
        for index in range(self._num):
            model = self._model_list[index]
            train_models.predict(model, self._config, index)
    
    def optimal_param(self):
        self.load_models()
        opt_param.get_optimal_parameters(self._model_list, self._config)


