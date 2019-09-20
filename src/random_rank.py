#!/usr/bin/python
# coding=utf-8

"""
@description:
This python file implements the Random approach. 

INPUT:
    1) testing set    : "../experiment/rank_based/"
OUTPUT:
    1) re-ranking results on testing set : "../experiment/random/"

@author  : Yongfeng
@reviewer: Yongfeng
"""

import warnings
import pandas as pd
from pandas import DataFrame
import numpy as np
import random as rd
import os

rd_seed = 5

def randomize(pdcontent):
    """
    Note: randomly rank the configurations with the same predicted perforamnce
    """
    results = DataFrame()

    repeat = pdcontent["pre_performance"].value_counts()
    repeat = repeat.sort_index()

    index = 0
    # print(repeat)
    np.random.seed(rd_seed)
    for rep in repeat:
        # print("%d:%d"%(index, index + rep))
        rd_ranks = pdcontent.reindex(np.random.permutation(pdcontent[index:index + rep].index))
        results = pd.concat([results, rd_ranks], axis=0)
        index += rep

    return results


def random_rank():

    testfolds = '../experiment/rank_based/'
    resultfolds = '../experiment/random_rank/'
    datasetsfolds = '../raw_data/'

    dataset_lists = [f[:-4] for f in os.listdir(datasetsfolds) if ".csv" in f] # dataset_lists = ['AJStats', 'Apache', 'BerkelyC', ...]
    for dataset in dataset_lists: # in each dataset
        if not os.path.exists(resultfolds + dataset):  # resultfolds + dataset = '../experiment/classification/AJStats/'
            os.makedirs(resultfolds + dataset)

        print(dataset)
        # for i in trains: 
        for i in range(50): # in each round
            # print(train_path)
            test_path = testfolds + dataset + '/rank_based' + str(i) + ".csv"  # test_path = "../experiment/rank_based/AJStats/rank_based0.csv"
            test = pd.read_csv(test_path)
            # print(test_path)
            result_rd = randomize(test)
              
            result_rd.to_csv(resultfolds + dataset + '/newRankedList' + str(i) + ".csv", index=False)


################################
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    pd.set_option('display.width', 200)

    random_rank()