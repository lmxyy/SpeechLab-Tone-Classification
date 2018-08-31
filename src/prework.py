#!/usr/bin/env python3
import sys,getopt,os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('prework')

def work(train_set):
    labels = ['one','two','three','four']
    for label in labels:
        path = train_set+tone+'/'
        files = os.listdir(path)
        for fle in files:
            if fle == '.DS_Store':
                continue
            pass
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        pass
    else:
        train_set = sys.argv[1]
        if train_set != 'train' and train_set != 'all' and train_set != 'dev':
            pass
        train_set = '../data/'+train_set+'/'
        work(train_set)
