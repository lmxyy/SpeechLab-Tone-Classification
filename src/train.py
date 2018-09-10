#!/usr/bin/env python3
import logging
import sys,os
import numpy as np
import torch
import csv
from cnn import CNN
from solver import Solver

logger = logging.getLogger('prework')
stream = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(fmt='%(name)s:  %(message)s - %(asctime)s',datefmt = '[%d/%b/%Y %H:%M:%S]')
stream.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
stream.setFormatter(formatter)
logger.addHandler(stream)

def work():
    prefix = '../data/cache'
    files = os.listdir(prefix)
    X_train = torch.tensor(np.load(prefix+'/X_training.npy'),dtype = torch.float32)
    y_train = torch.tensor(np.load(prefix+'/y_training.npy'),dtype = torch.int64)
    data = {'X_train':X_train,'y_train':y_train}
    if 'X_valid.npy' in files:
        X_val = torch.tensor(np.load(prefix+'/X_valid.npy'),dtype = torch.float32)
        data['X_val'] = X_val
    if 'y_valid.npy' in files:
        y_val = torch.tensor(np.load(prefix+'/y_valid.npy'),dtype = torch.int64)
        data['y_val'] = y_val
    X_test = torch.tensor(np.load(prefix+'/X_test.npy'),dtype = torch.float32)
    conv_dims = [(32,3),(64,3),(128,3)]
    # conv_dims = [(32,3)]
    hidden_dims = [256,128,4]
    # hidden_dims = []
    # connect_conv = (256,3)
    connect_conv = None
    use_batchnorm = True
    
    cnn = CNN(conv_dims = conv_dims,hidden_dims = hidden_dims,input_dim = X_train.numpy().shape[1:],connect_conv = connect_conv,use_batchnorm = use_batchnorm)
    solver = Solver(
        model = cnn,
        data = data,
        num_epochs = 20,
        batch_size=20,
        update_rule='adam',
        optim_config={'lr': 1e-3,'betas':(0.9,0.999),'eps':1e-8,'weight_decay':0.05},
        print_every=100,
        verbose = True
    )
    solver.train()
    y_pred = solver.predict(X_test)
    results = [[i+1,y_pred[i]+1] for i in range(len(y_pred))]
    with open('../data/result.csv','w',newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['id','classes'])
        writer.writerows(results)

if __name__ == '__main__':
    work()
