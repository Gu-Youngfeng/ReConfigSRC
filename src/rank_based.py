#!/usr/bin/python
# coding=utf-8

"""
@description:
This python file implements the right version of the rank-based approach.

INPUT:
    1) the original data set, i.e., "../raw_data/"
OUTPUT:
    1) split result, e.g., "../parse_data/data_split/AJStats/rank_based0/" + {train_set.csv; validation.csv; test.csv}
    2) sub-train results on training pool, e.g., "../parse_data/sub_train/AJStats/" + {subtrain0.csv}
    3) rank-based results on testing pool, i.e., "../experiment/rank_based/AJSats/" + {rank_based0.csv}

@author  : Yaoyao
@reviewer: Yongfeng
"""

import pandas as pd
import numpy as np
import os
from pandas import DataFrame
import random
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
import warnings

class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank


class result_holder:
    def __init__(self,decisions,objective,truly_rank,predicted,pre_rank):
        self.decision = decisions
        self.objective = objective
        self.truly_rank = truly_rank
        self.predicted = predicted
        self.pre_rank = pre_rank
 

def split_data(pdcontent,fraction):
    """
    Note: split the pdcontent into train set, validation set and test set by the ratio of {fraction: 0.2: 0.8-fraction}
    """
    indepcolumns = [col for col in pdcontent.columns if "$<" not in col]
    depcolumns = [col for col in pdcontent.columns if "$<" in col]
    sortpdcontent = pdcontent.sort_values(depcolumns[-1])
    content = list()
    for c in range(len(pdcontent)):
        content.append(solution_holder(
                                       c,
                                       sortpdcontent.iloc[c][indepcolumns].tolist(),
                                       sortpdcontent.iloc[c][depcolumns].tolist(),
                                       c
                                       )
                       )
    random.shuffle(content)
    indexes = range(len(content))
    train_indexes, validation_indexes, test_indexes = indexes[:int(fraction*len(indexes))], indexes[int(fraction*len(indexes)):int((fraction+0.2)*len(indexes))],  indexes[int((fraction+0.2)*len(indexes)):]
    assert(len(train_indexes) + len(validation_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    validation_set = [content[i] for i in validation_indexes]
    test_set = [content[i] for i in test_indexes]
    

    return [train_set,validation_set,test_set]


def update_data(data):
    """
    Note: transfer the data into the DataFrame format
    """
    x = [t.decision for t in data]
    y = [t.objective[-1] for t in data]
    data_x = DataFrame(x)
    data_x['act_performance'] = DataFrame(y)
    
    return data_x


def carts(train,test):
    """
    Note: predict ${test} by building a CART model on ${train}, ${test} and ${train} are the lists of solution_holder objects
    """
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    test = test[test.columns[:-1]]
    
    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test)
    
    return predicted


def rank_progressive(train, test):
    """
    Note: calculate the mean rank difference in prediction of test set
          without considering the tied predicted performances, the mean rank difference is usually smaller than before
    """
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    sorted_test = sorted(test, key=lambda x: x.objective[-1])
    for r,st in enumerate(sorted_test): st.rank = r
    test_independent = [t.decision for t in sorted_test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)
    predicted_id = [[i,p] for i,p in enumerate(predicted)]
    predicted_sorted = sorted(predicted_id, key=lambda x: x[-1])
    predicted_rank_sorted = [[p[0], p[-1], i] for i,p in enumerate(predicted_sorted)]
    rank_diffs = [abs(p[0] - p[-1]) for p in predicted_rank_sorted]
    return np.mean(rank_diffs)


def wrapper_rank_progressive(train_set, validation_set):
    """
    Note: return the sub_train with Rank-based method using train set and validation set.  
    """
    initial_size = 10
    training_indexes = range(len(train_set))
    shuffle(training_indexes)
    sub_train_set = [train_set[i] for i in training_indexes[:initial_size]]
    steps = 0
    rank_diffs = []
    while (initial_size+steps) < len(train_set) - 1:
        rank_diffs.append(rank_progressive(sub_train_set, validation_set))
        policy_result = policy(rank_diffs)
        if policy_result != -1: break
        steps += 1
        sub_train_set.append(train_set[initial_size+steps])

    return sub_train_set


def policy(scores, lives=3):
    """
    Note: the stop creteria designed by Nair et al.
    """
    temp_lives = lives
    last = scores[0]
    for i,score in enumerate(scores):
        if i > 0:
            if temp_lives == 0:
                return i
            elif score >= last:
                temp_lives -= 1
                last = score
            else:
                temp_lives = lives
                last = score
    return -1


def find_lowest_rank(train, test):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    sorted_test = sorted(test, key=lambda x: x.objective[-1]) # sort the test set by actual performance
    for r, st in enumerate(sorted_test): st.rank = r
    test_independent = [t.decision for t in sorted_test]
    test_dependent = [t.objective[-1] for t in sorted_test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)
    predicted_id = [[i, p] for i, p in enumerate(predicted)]
    predicted_sorted = sorted(predicted_id, key=lambda x: x[-1]) # sort the test set by predicted performance
    predicted_rank_sorted = [[p[0], p[-1], i] for i,p in enumerate(predicted_sorted)]
    select_few = predicted_rank_sorted[:10]   
    
    test_content = list()
    for c in range(len(sorted_test)):
        test_content.append(result_holder(
                                       test_independent[c],
                                       test_dependent[c],
                                       c+1,  # actual rank
                                       predicted[c],
                                       1   # predicted rank
                                       )
                       )
    random.shuffle(test_content) # randomize the test set
    predicted_sorts = sorted(test_content,key = lambda x : x.predicted) # sort the test set by predicted performance
    for r, st in enumerate(predicted_sorts): st.pre_rank = r+1  # update the predicted rank
    select_few_modify = [t.truly_rank - 1 for t in predicted_sorts]
    
    return [sf[0] for sf in select_few], select_few_modify[:10],predicted_sorts


def rank_based():
    fraction = 0.4
    datafolder = "../raw_data/"
    resultfolder = "../experiment/rank_based/"
    trainfolder = "../parse_data/sub_train/"
    splitfolder = "../parse_data/data_split/"
            
    files = [datafolder + f for f in os.listdir(datafolder) if ".csv" in f]  # file = "../raw_data/AJSats.csv"
    results = {}
    for file in files:
        print(file)
        filename = file.split('/')[-1].split('.')[0]  # filename = "AJStats"
        file_set = pd.read_csv(file)
        if not os.path.exists(resultfolder + filename):
            os.makedirs(resultfolder + filename)
        if not os.path.exists(trainfolder + filename):
            os.makedirs(trainfolder + filename)
        first_rank_min = []
        first_rank_media = []
        results[file] = {}
        results[file]["rank-based"] = {}
        results[file]["rank-based"]["mres"] = []
        results[file]["rank-based"]["train_set_size"] = []
        results[file]["rank-based"]["min_rank"] = []
        results[file]["rank-based"]["min_rank_modify"] = []
        for i in range(50):
            datasets = split_data(file_set, fraction) # split the dataset into 3 parts
            train_set = datasets[0]
            validation_set = datasets[1]
            test_set = datasets[2]
            
            if not os.path.exists(splitfolder + filename +'/rank_based'+str(i)):
                os.makedirs(splitfolder + filename +'/rank_based'+str(i))
            train = update_data(train_set)
            train.to_csv(splitfolder + filename +'/rank_based'+str(i)+ '/train_set.csv', index=False)  # write the train set into csv
            validation = update_data(validation_set)
            validation.to_csv(splitfolder + filename +'/rank_based'+str(i)+ '/validation_set.csv', index=False) # write the validation set into csv
            # data = pd.concat([train,validation],axis = 0)  # what does variable "data" do?
            test = update_data(test_set)
            test.to_csv(splitfolder + filename +'/rank_based'+str(i)+ '/test_set.csv', index=False) # write the test set into csv
        
            sub_train_set_rank = wrapper_rank_progressive(train_set, validation_set)
            sub_train = update_data(sub_train_set_rank)
            sub_train.to_csv(trainfolder + filename + '/subtrain'+str(i)+'.csv', index=False)
            lowest_rank,lowest_rank_modify,predicted_sorts = find_lowest_rank(sub_train_set_rank, test_set)

            test_data = DataFrame()
            test_data['feature'] = [t.decision for t in predicted_sorts]
            test_data['act_performance'] = [t.objective for t in predicted_sorts]
            test_data['act_rank'] = [t.truly_rank for t in predicted_sorts]
            test_data['pre_performance'] = [t.predicted for t in predicted_sorts]
            test_data['pre_rank'] = [t.pre_rank for t in predicted_sorts]
                       
            test_data.to_csv(resultfolder + filename + '/rank_based'+str(i)+'.csv', index=False)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pd.set_option('display.width',200)

    rank_based()
