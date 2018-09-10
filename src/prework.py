#!/usr/bin/env python3
import sys,os
import logging
import numpy as np
import random
import scipy.interpolate
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger('prework')
stream = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(fmt='%(name)s:  %(message)s - %(asctime)s',datefmt = '[%d/%b/%Y %H:%M:%S]')
stream.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
stream.setFormatter(formatter)
logger.addHandler(stream)

threshold_f0 = 0.2
threshold_engy = 0.4
qualify_f0 = 0.6
baseline_engy = 0.01
C = 2
H = 1
W = 256

def getArr(name):
    f = open(name,'r');
    ret = []
    for line in f.readlines():
        ret.append(float(line))
    return np.array(ret)

def spline(y,x,x_pred):
    f = scipy.interpolate.interp1d(x, y, kind='cubic')
    # f = UnivariateSpline(x,y,s = 1)
    return f(x_pred)

def getData(name):
    f0 = getArr(name+'.f0')
    engy = getArr(name+'.engy')
    x = np.array([i/len(engy) for i in range(len(engy))])
    index = []
    f0_mx = np.max(f0)
    engy_mx = np.max(engy)
    f0 /= f0_mx
    engy /= engy_mx
    for i in range(len(engy)):
        if (engy[i] < threshold_engy or f0[i] < threshold_f0) and f0[i] < qualify_f0:
            continue
        index.append(i)
    f0 = f0[index]
    engy = engy[index]
    x = x[index]
    x = (x-np.min(x))/(np.max(x)-np.min(x))

    index = random.sample([i for i in range(1,len(f0)-1)],min(len(f0),10))
    index.append(0)
    index.append(len(f0)-1)
    x_pred = np.linspace(0, 1, H*W)
    f0 = spline(f0[index],x[index],x_pred)
    engy = spline(engy[index],x[index],x_pred)
    return f0,engy

def work_train(train_set,attr):
    logger.info('Start handling %s data.' %(attr))
    labels = ['one','two','three','four']
    data = {'one':[],'two':[],'three':[],'four':[]}
    total = 0
    for label in labels:
        path = '../data/'+train_set+'/'+label+'/'
        files = os.listdir(path)
        nfiles = len(files)-int('.DS_Store' in files)
        cnt = 0
        prefix_set = set()
        logger.info('Start handling the label '+label+'.')
        for fle in files:
            if fle == '.DS_Store':
                continue
            prefix = fle.rstrip('.f0').rstrip('.engy')
            if prefix in prefix_set:
                continue
            prefix_set.add(prefix)
            f0,engy = getData(path+prefix)
            item = {'f0':f0,'engy':engy}
            data[label].append(item)
            cnt += 1
            if cnt % 100 == 0:
                logger.info('Finish %.2f%% data of the label %s.'%(100*cnt/nfiles,label))
        total += cnt
        logger.info('Finish handling the label '+label+'.')
    logger.info('Start vectorization.')
    X = np.zeros((total,2,H,W))
    y = []
    row = 0
    for k,v in data.items():
        for item in v:
            y.append(labels.index(k))
            X[row,0,:,:] = item['f0'].reshape((H,W))
            X[row,1,:,:] = item['engy'].reshape((H,W))
            row += 1
    logger.info('Finish vectorization.')

    np.save('../data/cache/X_'+attr+'.npy',X)
    np.save('../data/cache/y_'+attr+'.npy',y)

    logger.info('Save the vectors to ../data/cache.')
    logger.info('Finish handling %s data.' %(attr))

def work_test(test_set):
    logger.info('Start handling %s data.' %('test'))
    path = '../data/'+test_set+'/'
    files = os.listdir(path)
    nfiles = len(files)-int('.DS_Store' in files)
    data = []
    cnt = 0
    prefix_set = set()
    for i in range(nfiles>>1):
        prefix = str(i+1)
        f0,engy = getData(path+prefix)
        item = {'f0':f0,'engy':engy}
        data.append(item)
        cnt += 1
        if cnt % 100 == 0:
            logger.info('Finish %.2f%% data of the test data.'%(100*cnt/nfiles))
    logger.info('Start vectorization.')
    X = np.zeros((cnt,2,H,W))
    row = 0
    for item in data:
        X[row,0,:,:] = item['f0'].reshape((H,W))
        X[row,1,:,:] = item['engy'].reshape((H,W))
        row += 1
    logger.info('Finish vectorization.')

    np.save('../data/cache/X_test.npy',X)

    logger.info('Save the vectors to ../data/cache.')
    logger.info('Finish handling %s data.' % ('test'))
        
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
                data_set = '../data/'+data_set+'/'
                work_test(data_set)
            else:
                if data_set == 'train' or data_set == 'all':
                    work_train(data_set,'training')
                else:
                    work_train(data_set,'valid')
