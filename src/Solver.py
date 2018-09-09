#!/usr/bin/env python3
import logging
import sys
import torch
import numpy as np
from torch import nn
import torch.utils.data as Data

logger = logging.getLogger('Solver')
stream = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(fmt='%(name)s:  %(message)s - %(asctime)s',datefmt = '[%d/%b/%Y %H:%M:%S]')
stream.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
stream.setFormatter(formatter)
logger.addHandler(stream)

class Solver(object):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.X_train = data.get('X_train')
        self.y_train = data.get('y_train')
        self.X_val = data.get('X_val','X_train')
        self.y_val = data.get('y_val','y_train')
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
    
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            logger.error('Unrecognized arguments %s' % extra)
            raise ValueError('Unrecognized arguments %s' % extra)
        if self.update_rule == 'sgd':
            self.optimizer = torch.optim.SGD(
                params = model.parameters(),
                lr = self.optim_config.get('lr',0.001),
                momentum = self.optim_config.get('momentum',0),
                weight_decay = self.optim_config.get('weight_decay',0)
            )
        elif self.update_rule == 'adam':
            self.optimizer = torch.optim.Adam(
                params = model.parameters(),
                lr = self.optim_config.get('lr',0.001),
                betas = self.optim_config.get('betas',(0.9,0.999)),
                eps = self.optim_config.get('eps',1e-8),
                weight_decay = self.optim_config.get('weight_decay',0)
            )
        else:
            logger.error('Unrecognized update rule %s' % update_rule)
            raise ValueError('Unrecognized update rule %s' % update_rule)
        
        train_dataset = Data.TensorDataset(self.X_train,self.y_train)
        self.loader = Data.DataLoader(
            dataset = train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 1
        )
        
    def _step(self,X_batch,y_batch):
        out = self.model(X_batch)
        loss = self.loss_func(out,y_batch)
        self.loss_history.append(loss.data.numpy())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def check_accuracy(self,X,y):
        out = self.model(X)
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        acc = np.mean(y.numpy() == pred_y)
        return acc

    def predict(self,X):
        out = self.model(X)
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        return pred_y
    
    def train(self):
        num_train = self.X_train.shape[0]
        total_iterations = self.num_epochs*((num_train+self.batch_size-1)//self.batch_size)
        iteration = 0
        for epoch in range(self.num_epochs):
            for step,(X_batch,y_batch) in enumerate(self.loader):
                self._step(X_batch,y_batch)
                if self.verbose and iteration % self.print_every == 0:
                    logger.info('(Iteration %d / %d) loss: %f' % (iteration + 1, total_iterations, self.loss_history[-1]))
                iteration += 1
            train_acc = self.check_accuracy(self.X_train,self.y_train) # TODO
            self.train_acc_history.append(train_acc)
            val_acc = self.check_accuracy(self.X_val,self.y_val)
            self.val_acc_history.append(val_acc)
            if self.verbose:
                logger.info('(Epoch %d / %d) train acc: %f; val_acc: %f' % (epoch, self.num_epochs, train_acc, val_acc))
                
