# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class DataList(Dataset):
    def __init__(self, data):
        self._data = torch.Tensor(data)

    def __getitem__(self, index):
        return self._data[index, :16], self._data[index, 16:]
    
    def __len__(self):
        return len(self._data)

class Dataloader():
    def __init__(self, data_path, normalization_stats, batch_size, split = True):
        self._data = np.load(data_path)
        self._normalization_stats = normalization_stats
        # Feature engineering, such as logarithmic transformations of characteristic parameters.
        self.preprocessor()
        # Normalization of model inputs
        self._data = (self._data - self._normalization_stats.mean[:self._data.shape[1]])/self._normalization_stats.std[:self._data.shape[1]]
        if(split):
            #In training phase, the dataset needs to be randomly split into training set and validation set.
            self.splitting_datasets()
            train_dataset = DataList(self._training_set)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            test_dataset = DataList(self._valid_set)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size*100, shuffle=False, pin_memory=True)
        else:
            # In testing phase, the test set does not need to be split.
            test_dataset = DataList(self._data)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    def splitting_datasets(self):
        """Splitting the dataset."""
        index = np.random.rand(self._data.shape[0]) < 0.63
        self._training_set = self._data[index]
        self._valid_set = self._data[~index]
        
        
    def preprocessor(self):
        """Logarithmic transformation of each characteristic parameter."""
        log_features = []
        for index in range(8):
            param = np.log(np.abs(self._data[:,index]))
            param = param[:, np.newaxis]
            log_features.append(param)
        log_features =np.concatenate(log_features, axis = 1)
        self._data = np.concatenate((self._data[:, :8], log_features, self._data[:, 8:]), axis = 1)
        
