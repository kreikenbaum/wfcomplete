#! /usr/bin/env python
'''computes percentiles of json file with data [['url', value], ...]
given by first argument'''
import numpy as np
import json
import sys

BINS=5

def parse(filename):
    '''@return second component of value of json of filename'''
    vals = [i[1] for i in json.load(file(filename))]
    return vals

def for_i(values, i, bins):
    '''@return percentile i/(bins*2)'''
    return int(round(np.percentile(values, 100.0*i/(2*bins))))

def splits_and_sizes(filename=sys.argv[1], bins=BINS):
    '''@return sizes and borders of histogram bins based on percentiles'''
    sizes = []
    splits = []
    values = parse(filename)
    for i in range(1, bins*2):
        if i % 2 == 0:
            splits.append(for_i(values, i, bins))
        else:
            sizes.append(for_i(values, i, bins))
    return (sizes, splits)

if __name__ == '__main__':
    print splits_and_sizes()
                            

#     print '{}: {}'.format(i/6.0, )

    
        
