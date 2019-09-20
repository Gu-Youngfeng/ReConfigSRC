#!/usr/bin/python
# coding=utf-8

"""
@description:
This python file implements the classification approach (i.e., Random Forest).

INPUT:
    1) sub-train set  : "../parse_data/sub_train"
    2) validation set : "../parse_data/data_split"
    3) testing set    : "../experiment/rank_based/"
OUTPUT:
    1) classification results on testing pool : "../experiment/classification/"

@author  : Yaoyao
@reviewer: Yongfeng
"""

import warnings
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.utils import shuffle


def remark_anomaly(train,validation):
    train_independent = train[train.columns[:-1]]
    train_dependent = train[train.columns[-1]]
    
    validation = validation.sort_values(by=["act_performance"])
    validation['act_rank'] = np.arange(1,len(validation)+1)  # add actual rank for each configuration in validation set
    validation = shuffle(validation)  # randomize the validation set
    
    cart = DecisionTreeRegressor()
    cart.fit(train_independent,train_dependent)
    validation_independent = validation[validation.columns[:-2]]
    predicted = cart.predict(validation_independent)  # predict on validation set
    validation['pre_performance'] = predicted
    validation = validation.sort_values(by=["pre_performance"])
    validation['pre_rank'] = np.arange(1,len(validation)+1)
    # print(validation[:10])
    # print("------")
    result = split_block(validation)
    result_isAnomaly = result[result.columns[:len(validation_independent.columns)]]
    result_isAnomaly['isAnomaly'] = result['isAnomaly']
    # result_isAnomaly.to_csv("../temp_data/round49.csv", index=False)
    return result,result_isAnomaly


def split_block(data):
    
    repeat = data['pre_performance'].value_counts()  # figure out the tied predicted performance
    repeat = repeat.sort_index()  
    index = 0
    result = DataFrame()
   
    for rep in repeat:
        block = data[index:index + rep]
        # block = block.sort_values(block.columns[-4],ascending = True) # BUG-1:here -6 should be -4 
        block = block.sort_values(by=["act_performance"])  
        # print(block.columns[-6], rep)
        noAnomaly_indexes = np.ones(1, np.int)  # 1 for the best performance
        Anomaly_indexes = [-1 * x for x in np.ones(rep-1, np.int)] # -1 for the others
        
        indexes = np.append(noAnomaly_indexes,Anomaly_indexes)
        block['isAnomaly'] = indexes
        index = index + rep
        result = pd.concat([result,block],axis = 0)
    # print(result[:10])
    return result


def rf_Anomaly_ratio(train,test): # here train refers to the prediction results from the validation set
    train_independent = train[train.columns[:-1]]
    train_dependent = train["isAnomaly"]
    test_set = test['feature']
    # test_set_num = [i.split('[')[-1].split(']')[0].split(',') for i in test_set] # BUG-2: test_set should be in the form of numeric [patch:77-81]
    test_set_num = []
    test_set_lst_lst = [i[1:-1].split(', ') for i in test_set]
    for test_str_lst in test_set_lst_lst:
        test_str_lst = [float(tt) for tt in test_str_lst]
        test_set_num.append(test_str_lst)

    clf = RandomForestClassifier()
    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression()
    clf.fit(train_independent, train_dependent)
    test['isAnomaly'] = clf.predict(test_set_num)
    predict_ratio = clf.predict_proba(test_set_num)  # the ratio of prediction, such as [0.9, 0.1]
    # cc1 = 0
    # for res in train_dependent:
    #     if res == 1:
    #         cc1 += 1
    # print("train labels as 1:", cc1)
    # cc = 0
    # predict_results = clf.predict(test_set_num)
    # for res in predict_results:
    #     if res == 1:
    #         cc += 1
    # print("predicted as 1:", cc)
    predict = DataFrame(predict_ratio)
    
    test['Anomaly_ratio'] = predict[predict.columns[0]] # predicted as -1
    test['Normal_ratio'] = predict[predict.columns[1]] # predicted as 1
    # print(test[:10])
    result = block_sort(test)
    
    return result


def block_sort(data):
    repeat = data['pre_performance'].value_counts()
    repeat = repeat.sort_index()
    
    index = 0
    result = DataFrame()
   
    for rep in repeat:
        block = data[index:index + rep]
        block = block.sort_values(by=["Normal_ratio"],ascending = False)
        index = index + rep
        result = pd.concat([result,block],axis = 0)
    
    return result


def classification():
    
    datafolder = "../raw_data/"
    trainfolds = "../parse_data/sub_train/"
    validationfolds = "../parse_data/data_split/"
    testfolds = "../experiment/rank_based/"
    resultfolds = '../experiment/classification/'

    dataset_lists = [f[:-4] for f in os.listdir(datafolder) if ".csv" in f] # dataset_lists = ['AJStats', 'Apache', 'BerkelyC', ...]
    for dataset in dataset_lists: # in each dataset
        if not os.path.exists(resultfolds + dataset):  # resultfolds + dataset = '../experiment/classification/AJStats/'
            os.makedirs(resultfolds + dataset)
        # trains = [f for f in os.listdir(trainfolds + dataset)] # trains = ["subtrain0.csv","subtrain1.csv",...]
        rounds = len(os.listdir(trainfolds + dataset))
        print(dataset)
        # for i in trains: 
        for i in range(rounds): # in each round
            validation_path = validationfolds + dataset + '/rank_based' + str(i) + '/' 
            validation = pd.read_csv(validation_path + 'validation_set.csv')  # validation = "../parse_data/data_split/AJStats/rank_based0/validation_set.csv"
            # print(validation_path)
            sub_train_path = trainfolds + dataset + '/subtrain' + str(i) + ".csv"  # train_path = "../parse_data/sub-train/AJStats/subtraind0.csv"
            sub_train = pd.read_csv(sub_train_path)
            # print(train_path)
            test_path = testfolds + dataset + '/rank_based' + str(i) + ".csv"  # test_path = "../experiment/rank_based/AJStats/rank_based0.csv"
            test = pd.read_csv(test_path)
            # print(test_path)
            
            anomaly_all,anomaly_train = remark_anomaly(sub_train, validation)
            
            result_rf = rf_Anomaly_ratio(anomaly_train, test)
            result_rf.to_csv(resultfolds + dataset + '/newRankedList' + str(i) + ".csv", index=False)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pd.set_option('display.width',200)

    classification()

    
    
