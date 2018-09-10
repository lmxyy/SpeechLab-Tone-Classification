#!/usr/bin/env python3
import sys,os
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import UnivariateSpline

prefix = '../data/cluster/'
logger = logging.getLogger('plot')
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
H = 1
W = 128

def getArr(name):
    f = open(name,'r')
    ret = []
    for line in f.readlines():
        ret.append(float(line))
    return np.array(ret)

def spline(y,x,x_pred):
    f = scipy.interpolate.interp1d(x, y, kind='cubic')
    # f = UnivariateSpline(x,y,s = 1)
    return f(x_pred)

def work(path):
    f0 = getArr(path+'.f0')
    engy = getArr(path+'.engy')
    x = np.array([i/len(engy) for i in range(len(engy))])
    index = []
    f0_mx = np.max(f0)
    engy_mx = np.max(engy)

    f0 /= f0_mx
    engy /= engy_mx

    plt.figure(1)    
    plt.plot(x,f0,color = 'red',label = 'f0')
    plt.plot(x,engy,color = 'blue',label = 'energy')
    plt.legend(loc = 'upper left')
    plt.title('origin')
    # plt.show()

    for i in range(len(engy)):
        if ((engy[i] < threshold_engy or f0[i] < threshold_f0) and f0[i] < qualify_f0) or (engy[i] < baseline_engy):
            continue
        index.append(i)

    f0 = f0[index]
    engy = engy[index]
    x = x[index]
    x = (x-np.min(x))/(np.max(x)-np.min(x))

    plt.figure(2)    
    plt.plot(x,f0,color = 'red',label = 'f0')
    plt.plot(x,engy,color = 'blue',label = 'energy')
    plt.legend(loc = 'upper left')
    plt.title('sample')
    # plt.show()
    
    index = random.sample([i for i in range(1,len(f0)-1)],min(len(f0),10))
    index.append(0)
    index.append(len(f0)-1)
    index.sort()
    x_pred = np.linspace(0, 1, H*W)
    f0 = spline(f0[index],x[index],x_pred)
    engy = spline(engy[index],x[index],x_pred)
    
    plt.figure(3)
    plt.plot(x_pred,f0,color = 'red',label = 'f0')
    plt.plot(x_pred,engy,color = 'blue',label = 'energy')
    plt.legend(loc = 'upper left')
    plt.title('derivitive')
    plt.show()
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        logger.error('Please input one argument.')
        raise ValueError('Please input one argument.')
    else:
        name = sys.argv[1]
        if os.path.isfile(prefix+name+'.f0') == False or os.path.isfile(prefix+name+'.engy') == False:
            logger.error("File %s doesn't exist." % (name))
            raise ValueError("File %s doesn't exist." % (name))
        work(prefix+name)
        
        
