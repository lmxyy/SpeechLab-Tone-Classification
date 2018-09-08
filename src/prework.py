#!/usr/bin/env python3
import sys,getopt,os
import logging
import numpy as np
import random
import torch
from torch.autograd import Variable

logger = logging.getLogger('prework')
stream = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(fmt='%(name)s:  %(message)s - %(asctime)s',datefmt = '[%d/%b/%Y %H:%M:%S]')
stream.setLevel(logging.DEBUG)
stream.setFormatter(formatter)
logger.addHandler(stream)

threshold = 0.1

def getArr(name):
    f = open(name,'r');
    ret = []
    for line in f.readlines():
        ret.append(int(line))
    return np.array(ret)

def spline(y,x,x_pred):
    f = UnivariateSpline(x, y, s=1)
    return f(x_pred)

def getData(name):
    f0 = getArr(name+'.f0')
    engy = getArr(name+'.engy')
    index = []
    engy_mx = np.max(engy)
    f0_mx = np.max(engy)
    for i in range(len(engy)):
        if engy[i]/engy_mx < threshold:
            continue
        index.append(i)
    f0 = f0[index]
    engy = engy[index]
    f0 /= f0_mx
    engy /= engy_mx
    index = random.sample([i for i in range(len(f0))],min(len(f0),75))
    x = [i/len(engy) for i in range(engy)]
    x_pred = np.linspace(0, 1, 60)
    f0 = spline(f0[index],x[index],x_pred)
    engy = spline(engy[index],x[index],x_pred)
    return f0,engy

def vectorize(data):
    pass

def work_train(train_set):
    logger.info('Start handling train data.')
    labels = ['one','two','three','four']
    data = {'one':[],'two':[],'three':[],'four':[]}
    for label in labels:
        path = train_set+label+'/'
        files = os.listdir(path)
        nfiles = len(files)-int('.DS_Store' in files)
        cnt = 0
        prefix_set = set()
        logger.info('Start handling the label',label+'.')
        for fle in files:
            if fle == '.DS_Store':
                continue
            prefix = fle.strip('.f0').strip('.engy')
            if prefix in prefix_set:
                continue
            prefix_set.add(prefix)
            f0,engy = getData(path+prefix)
            item = {'f0':f0,'engy':engy}
            data[label].append(item)
            cnt += 1
            if cnt % 100 == 0:
                logger.info('Finish {.2f} data of the label {}.'.format(cnt/nfiles,label))
        logger.info('Finish handling the label',label+'.')
    vectorize(data)
    logger.info('Finish the prework.')
    
def work_test(test_set):
    pass
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        logger.error('Please input one argument.')
        raise ValueError('Please input one argument.')
    else:
        data_set = sys.argv[1]
        if data_set != 'train' and data_set != 'dev' and data_set != 'test' and data_set != 'all':
            logger.error("The argument can only be 'train', 'dev', 'test'or 'all'.")
            raise ValueError('Unrecognized arguments %s.' % data_set)

        else:
            if data_set == 'test':
                data_set = '../data/'+train_set+'/'
                work_test(data_set)
            else:
                work_train(data_set)
