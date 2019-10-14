#!/usr/bin/python
# coding=utf-8

"""
@description:
This python file implements the ReConfig approach. 
Note that this re-ranking results do not filter out any configurations, the filtering process is conducted in experiment.py

INPUT:
    1) predicted results on validation pool, i.e., "../parse_data/sub_train", "../parse_data/data_split"
    2) predicted results on testing pool, i.e., "../experiment/rank_based/"
OUTPUT:
    1) re-ranking results on testing pool, i.e., "../experiment/reconfig/"

@author  : Yuntianyi
@refactor: Yongfeng
"""

import os
import warnings
from os import listdir
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor

param_ranker  = [0, 1, 2, 3, 4, 6, 7, 8]
param_mid_cmd = [" -r 5 -i 25 -tolerance 0.001 -reg no-regularization", # 0-MART
				 " -epoch 100 -layer 1 -node 10 -lr 0.00005", # 1-RankNet
				 " -round 300 -tc 10", # 2-RankBoost
				 " -metric2t ERR@10 -round 500 -noeq unspecified -tolerance 0.002 -max 5", # 3-AdaRank (list-wise)
				 " -metric2t ERR@10 -r 5 -i 25 -tolerance 0.001 -reg no-regularization", # 4-Coordinate Ascent (list-wise)
				 " -metric2t ERR@10 -r 5 -i 25 -tolerance 0.001 -reg no-regularization", # 6-LambdaMart (list-wise)
				 "", # 7-ListNet (list-wise)
				 " -bag 300 -srate 1.0 -frate 0.3 -rtype 0 -tree 1 -leaf 100 -shrinkage 0.1"] # 8-Random Forest

##########################################
'''
get trainset csvfile
'''
class SolutionHolder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank


def update_data(data):
    x = [t.decision for t in data]
    y = [t.objective[-1] for t in data]
    data_x = DataFrame(x)
    data_x['act_performance'] = DataFrame(y)
    return data_x


def carts(train, test):
    """
    Note: use CART to predict preformance in test set
    """
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]
    test = test[test.columns[:-1]]
    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test)

    return predicted


def read_data(dataset):
    content = list()
    # indepcolumns = [col for col in dataset.columns if "truly" not in col]
    # depcolumns = [col for col in dataset.columns if "truly" in col]
    indepcolumns = dataset.columns[:-1].tolist()
    depcolumns = dataset.columns[-1:].tolist()
    for c in range(len(dataset)):
        content.append(SolutionHolder(
            c,
            dataset.iloc[c][indepcolumns].tolist(),
            dataset.iloc[c][depcolumns].tolist(),
            dataset['act_performance'][c]
        )
        )
    return content


def get_train_csv(data, predicted, name, index):
    """
    Note: add prediction result with the data, then save the data into csv files,
          the sort_values() is implemented by "quick sorting", which is not a stable algorithm.
    """
    data['pre_performance'] = predicted
    data = data.sort_values(by=["act_performance"])  # sort by actual performance
    data['act_rank'] = range(1, len(data) + 1)
    data = data.sort_values(by=["pre_performance"])  # sort by predicted performance
    data['pre_rank'] = range(1, len(data) + 1)
    data['rank_difference'] = np.abs(data['act_rank'] - data['pre_rank'])


    resultfolder = "../temp_data/ltr_trainset/"

    if not os.path.exists(resultfolder + name):
        os.makedirs(resultfolder + name)

    if index < 10:
        name_index = str(0) + str(index)
    else:
        name_index = str(index)

    # print(data[:10]) # print the data
    data.to_csv(resultfolder + name + '/ltr_trainset_' + name_index + '.csv')


def predict_on_validation_set():
    """
    Note: use the sub_train set to predict on the validation set,
          save the results into "../temp_data/ltr_trainset/${project}/ltr_trainset_XX.csv"
    """
    datafolder = "../raw_data/"
    trainfolder = "../parse_data/sub_train/"
    split_datafolder = "../parse_data/data_split/"

    folders = [split_datafolder + f[:-4] for f in listdir(datafolder) if ".csv" in f]  # folder = "../parse_data/data_split/AJStats"
    sub_train_folders = [trainfolder + f for f in listdir(trainfolder)]
    for folderindex in range(len(folders)):  # for each project
        folder = folders[folderindex] # folder = "../parse_data/data_split/Apache"
        name = folder.split("/")[-1] # name = "Apache"
        sub_train_folder = sub_train_folders[folderindex]  # sub_train_folder = "../parse_data/sub_train/Apache"
        print(folder)
        files_folder = [folder + '/' + f for f in listdir(folder) if "rank_based" in f]
        sub_train_data = [sub_train_folder + '/' + f for f in listdir(sub_train_folder) if "subtrain" in f]
        fileindex = 0
        for file_folder in files_folder:  # for each data split
            files = [file_folder + '/' + f for f in listdir(file_folder)] # for each rank_based
            validation_set = []
            for csvfile in files:
                # filename = csvfile.split('/')[-1].split('.')[0]
                if "validation_set" in csvfile:
                    # validation_set = read_data(pd.read_csv(csvfile.title()).iloc[:, 1:])
                    validation_set = read_data(pd.read_csv(csvfile.title()))

            # sub_train_set_rank_raw = pd.read_csv(sub_train_data[fileindex]).iloc[:, 1:]
            sub_train_set_rank_raw = pd.read_csv(sub_train_data[fileindex])
            sub_train_set_rank = read_data(sub_train_set_rank_raw)

            validation = update_data(validation_set)
            dataset_to_test = validation

            cart_predicted = carts(sub_train_set_rank, dataset_to_test)
            get_train_csv(dataset_to_test, cart_predicted, name, fileindex)

            fileindex += 1


#################################### DEPRECATED BY YONGFENG 152:234
'''
rank trainset and testset csvfile by chunk
'''
def rank_by_chunk(dataset):
    """
    @deprecated
    """
    # dataset.rename(columns={"truly_performance": "truly"}, inplace=True)  # do not need to replace the column name
    ranked_dataset = dataset.sort_values(["pre_performance"])
    pre_value = None
    chunkid = 1
    id_list = []

    configuration_id = 0
    for index, row in ranked_dataset.iterrows():
        predicted_performance = row["pre_performance"]
        if configuration_id == 0:
            pre_value = predicted_performance
        if pre_value != predicted_performance:
            chunkid += 1
            id_list.append(chunkid)
            pre_value = predicted_performance
        else:
            id_list.append(chunkid)
        configuration_id += 1

    ranked_dataset["pre_rank"] = id_list
    ranked_dataset["rank_difference"] = abs(ranked_dataset["pre_rank"] - ranked_dataset["act_rank"])

    ranked_dataset["ID"] = id_list
    ranked_dataset = ranked_dataset.sort_values(["pre_performance", "act_performance"])

    return ranked_dataset


def get_csv(data, name, index, select):
    """
    @deprecated
    """
    data_ranked = rank_by_chunk(data)

    resultfolder = ""

    if select == "train":
        resultfolder = "../temp_data/rank_by_chunk_result/trainset_rank_by_chunk/"
    elif select == "test":
        resultfolder = "../temp_data/rank_by_chunk_result/testset_rank_by_chunk/"

    if not os.path.exists(resultfolder + name):
        os.makedirs(resultfolder + name)
    if index < 10:
        name_index = str(0) + str(index)
    else:
        name_index = str(index)

    filename = ""
    if select == "train":
        filename = "/trainset_rank_by_chunk"
    elif select == "test":
        filename = "/testset_rank_by_chunk"
    # print(data_ranked[:10])  # print the ranked data
    data_ranked.to_csv(resultfolder + name + filename + name_index + '.csv')


def chunk(select):
    """
    @deprecated
    """
    datafolder = ''
    if select == "train":
        datafolder = "../temp_data/ltr_trainset/"
    elif select == "test":
        datafolder = "../experiment/rank_based/"

    folders = [datafolder + f for f in listdir(datafolder)]

    for folderindex in range(len(folders)):  # for each project
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)
        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for csvfile in files:  # for each csv file
            dataset = pd.read_csv(csvfile.title()).iloc[:, 1:]
            get_csv(dataset, name, fileindex, select)
            fileindex += 1


#################################### EDITED BY YONGFENG

def block_validation_set():
    """
    Note: rank the validation set in each chunk, and add columns "ID"
          configurations with a same predicted preformance share a same "ID" 
    """
    datafolder = "../temp_data/ltr_trainset/"

    folders = [datafolder + f for f in listdir(datafolder)]

    for folderindex in range(len(folders)):  # for each project
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)
        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for csvfile in files:  # for each csv file
            dataset = pd.read_csv(csvfile.title())
            get_validation_csv(dataset, name, fileindex,)
            fileindex += 1


def get_validation_csv(data, name, index):

    data_ranked = rank_validation_set(data)
    resultfolder = "../temp_data/rank_by_chunk_result/trainset_rank_by_chunk/"

    if not os.path.exists(resultfolder + name):
        os.makedirs(resultfolder + name)
    if index < 10:
        name_index = str(0) + str(index)
    else:
        name_index = str(index)

    filename = filename = "/trainset_rank_by_chunk"

    # print(data_ranked[:10])  # print the ranked data
    data_ranked.to_csv(resultfolder + name + filename + name_index + '.csv', index=False)


def rank_validation_set(dataset):
    """
    Note: rank the validation set by predicted performances and actual performances
    """
    # dataset.rename(columns={"truly_performance": "truly"}, inplace=True)
    ranked_dataset = dataset.sort_values(["pre_performance"])
    pre_value = None
    chunkid = 1
    id_list = []

    configuration_id = 0
    for index, row in ranked_dataset.iterrows():
        predicted_performance = row["pre_performance"]
        if configuration_id == 0:
            pre_value = predicted_performance
        if pre_value != predicted_performance:
            chunkid += 1
            id_list.append(chunkid)
            pre_value = predicted_performance
        else:
            id_list.append(chunkid)
        configuration_id += 1

    ranked_dataset["pre_rank"] = id_list  # why pre_rank is the same with the ID?
    ranked_dataset["rank_difference"] = abs(ranked_dataset["pre_rank"] - ranked_dataset["act_rank"])

    ranked_dataset["ID"] = id_list
    ranked_dataset = ranked_dataset.sort_values(["pre_performance", "act_performance"])

    return ranked_dataset


def block_test_set():
    """
    Note: rank the validation set in each chunk, and add columns "ID"
          each tied predicted preformance share with the same "ID" 
    """
    datafolder = "../experiment/rank_based/"

    folders = [datafolder + f for f in listdir(datafolder)]

    for folderindex in range(len(folders)):  # for each project
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)
        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for csvfile in files:  # for each csv file
            dataset = pd.read_csv(csvfile.title())
            get_test_csv(dataset, name, fileindex)
            fileindex += 1


def get_test_csv(data, name, index):

    data_ranked = rank_test_set(data)
    resultfolder = "../temp_data/rank_by_chunk_result/testset_rank_by_chunk/"

    if not os.path.exists(resultfolder + name):
        os.makedirs(resultfolder + name)
    if index < 10:
        name_index = str(0) + str(index)
    else:
        name_index = str(index)

    filename = "/testset_rank_by_chunk"
    # print(data_ranked[:10])  # print the ranked data
    data_ranked.to_csv(resultfolder + name + filename + name_index + '.csv')


def rank_test_set(dataset):
    """
    Note: rank the test set by predicted performances
    """
    # dataset.rename(columns={"truly_performance": "truly"}, inplace=True)
    ranked_dataset = dataset.sort_values(["pre_performance"])
    pre_value = None
    chunkid = 1
    id_list = []

    configuration_id = 0
    for index, row in ranked_dataset.iterrows():
        predicted_performance = row["pre_performance"]
        if configuration_id == 0:
            pre_value = predicted_performance
        if pre_value != predicted_performance:
            chunkid += 1
            id_list.append(chunkid)
            pre_value = predicted_performance
        else:
            id_list.append(chunkid)
        configuration_id += 1

    ranked_dataset["pre_rank"] = id_list # why pre_rank is the same with the ID?
    ranked_dataset["ID"] = id_list
    ranked_dataset["rank_difference"] = abs(ranked_dataset["pre_rank"])

    # ranked_dataset["rank_difference"] = abs(ranked_dataset["pre_rank"] - ranked_dataset["truly_rank"])  
    # ranked_dataset = ranked_dataset.sort_values(["predicted"])

    return ranked_dataset

####################################### DEPRECATED BY YONGFENG
'''
preprocessing:label the trainset and testset csvfile, parse them to txtfile(for Ranklib to run)
'''
def label_config(select):
    """
    @deprecated
    """
    fromfolder = ""
    tofolder = ""

    if select == "train":
        fromfolder = "trainset_rank_by_chunk"
        tofolder = "trainset_label"
    elif select == "test":
        fromfolder = "testset_rank_by_chunk"
        tofolder = "testset_label"

    datafolder = "../temp_data/rank_by_chunk_result/" + fromfolder + "/"
    resultfolder = "../temp_data/label/" + tofolder + "/"

    folders = [datafolder + f for f in listdir(datafolder)]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)
        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for csvfile in files:
            dataset = pd.read_csv(csvfile.title())
            # dataset.rename(columns={"truly_performance": "truly"}, inplace=True)
            do_label(dataset, name, fileindex, resultfolder)
            fileindex += 1


def do_label(dataset, name, index, resultfolder):
    """
    @deprecated
    """
    datafolder = resultfolder
    if not os.path.exists(datafolder + name):
        os.makedirs(datafolder + name)

    label_list = []
    temp = [1]
    chunkid = None
    relevance = None
    performance_difference = None
    for configuration_id, row in dataset.iterrows():
        row_performance_difference = row["pre_performance"] - row["act_performance"]
        if configuration_id == 0:
            performance_difference = row_performance_difference
            relevance = 1
            chunkid = row["ID"]
        else:
            if chunkid != row["ID"]:
                chunkid = row["ID"]
                label_list += list_transfer(temp)
                temp = [1]
                relevance = 1
                performance_difference = row_performance_difference
            else:
                if performance_difference == row_performance_difference:
                    temp.append(relevance)
                else:
                    performance_difference = row_performance_difference
                    relevance += 1
                    temp.append(relevance)
        if configuration_id == len(dataset) - 1:
            label_list += list_transfer(temp)

    if index < 10:
        name_index = str(0) + str(index)
    else:
        name_index = str(index)

    dataset["label"] = label_list
    print(dataset[:10])
    dataset.to_csv(datafolder + name + '/' + datafolder.split('/')[-2] + name_index + '.csv')


def list_transfer(ranklist):
    """
    @deprecated
    """
    ranklist.reverse()
    dealt_list = []
    element = None
    pre_value = None
    for ids, value in enumerate(ranklist):
        if pre_value != value:
            pre_value = value
            element = ids + 1
            dealt_list.append(ids + 1)
        else:
            dealt_list.append(element)
    dealt_list.reverse()
    return dealt_list


######################################## EDITED BY YONGFENG

def label_validation_set():
    """
    Note: label the validation set with column "label"
          "label" describes the correlation in each ID block
    """
    fromfolder = "trainset_rank_by_chunk"
    tofolder = "trainset_label"

    datafolder = "../temp_data/rank_by_chunk_result/" + fromfolder + "/"
    resultfolder = "../temp_data/label/" + tofolder + "/"

    folders = [datafolder + f for f in listdir(datafolder)]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)
        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for csvfile in files:
            dataset = pd.read_csv(csvfile.title())
            # dataset.rename(columns={"truly_performance": "truly"}, inplace=True)
            add_validation_set_label(dataset, name, fileindex, resultfolder)
            fileindex += 1


def add_validation_set_label(dataset, name, index, resultfolder):
    """
    Note: add labels to validation set
    """
    datafolder = resultfolder
    if not os.path.exists(datafolder + name):
        os.makedirs(datafolder + name)

    label_list = []
    temp = [1]
    chunkid = None
    relevance = None
    performance_difference = None
    for configuration_id, row in dataset.iterrows():
        row_performance_difference = row["pre_performance"] - row["act_performance"]
        if configuration_id == 0:
            performance_difference = row_performance_difference
            relevance = 1
            chunkid = row["ID"]
        else:
            if chunkid != row["ID"]:
                chunkid = row["ID"]
                label_list += list_transfer(temp)
                temp = [1]
                relevance = 1
                performance_difference = row_performance_difference
            else:
                if performance_difference == row_performance_difference:
                    temp.append(relevance)
                else:
                    performance_difference = row_performance_difference
                    relevance += 1
                    temp.append(relevance)
        if configuration_id == len(dataset) - 1:
            label_list += list_transfer(temp)

    if index < 10:
        name_index = str(0) + str(index)
    else:
        name_index = str(index)

    dataset["label"] = label_list
    dataset.to_csv(datafolder + name + '/' + datafolder.split('/')[-2] + name_index + '.csv', index=False)


def list_transfer(ranklist):
    ranklist.reverse()
    dealt_list = []
    element = None
    pre_value = None
    for ids, value in enumerate(ranklist):
        if pre_value != value:
            pre_value = value
            element = ids + 1
            dealt_list.append(ids + 1)
        else:
            dealt_list.append(element)
    dealt_list.reverse()
    return dealt_list


def label_test_set():
    """
    Note: label the test set with column "label"
          since we did not know the actual performance, all "label"s in test set are 0. 
    """
    tofolder = "testset_label"
    fromfolder = "testset_rank_by_chunk"

    datafolder = "../temp_data/rank_by_chunk_result/" + fromfolder + "/"
    resultfolder = "../temp_data/label/" + tofolder + "/"

    folders = [datafolder + f for f in listdir(datafolder)]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)
        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for csvfile in files:
            dataset = pd.read_csv(csvfile.title())
            # dataset.rename(columns={"truly_performance": "truly"}, inplace=True)
            add_test_set_label(dataset, name, fileindex, resultfolder)
            fileindex += 1


def add_test_set_label(dataset, name, index, resultfolder):
    """
    Note: add labels to test set
    """
    datafolder = resultfolder
    if not os.path.exists(datafolder + name):
        os.makedirs(datafolder + name)

    label_list = [0 for x in range(len(dataset))]

    dataset["label"] = label_list

    if index < 10:
        name_index = str(0) + str(index)
    else:
        name_index = str(index)

    dataset.to_csv(datafolder + name + '/' + datafolder.split('/')[-2] + name_index + '.csv', index=False)


######################################### EDITED BY YONGFENG

def parse_config(select):
    """
    Note: transfer the csv file into txt file which can be processed by RankLib.jar
    """
    fromfolder = ""
    tofolder = ""

    if select == "train":
        fromfolder = "trainset_label"
        tofolder = "trainset_txt"
    elif select == "test":
        fromfolder = "testset_label"
        tofolder = "testset_txt"

    datafolder = "../temp_data/label/" + fromfolder + "/"
    resultfolder = "../temp_data/txt/" + tofolder + "/"

    folders = [datafolder + folder for folder in listdir(datafolder)]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)
        files = [folder + '/' + csvfile for csvfile in listdir(folder)]
        fileindex = 0
        for csvfile in files:
            dataset = pd.read_csv(csvfile.title())
            do_parse(dataset, name, fileindex, select, resultfolder)
            fileindex += 1


def do_parse(dataset, name, index, select, resultfolder):
    datafolder = resultfolder
    if not os.path.exists(datafolder + name):
        os.makedirs(datafolder + name)

    if index < 10:
        name_index = str(0) + str(index)
    else:
        name_index = str(index)

    txtfile = open(datafolder + name + '/' + datafolder.split('/')[-2] + name_index + ".txt", 'w')

    if select == "train":
        for configuration_id, row in dataset.iterrows():
            string_write = str(int(row["label"])) + " " + "qid:" + str(int(row["ID"])) + " "
            for featureid in range(len(row[1:-7])):
                string_write += str(featureid + 1) + ":" + str(row[featureid + 1]) + " "
            string_write += "#docid = " + str(int(row[0])) + " " + "fileid = " + str(index) + " "
            string_write += "act_rank = " + str(int(row["act_rank"])) + " " + "act_preformance = " + \
                            str(row["act_performance"]) + " " + "predicted_rank = " + str(int(row["pre_rank"])) + \
                            " " + "predicted_preformance = " + str(row["pre_performance"]) + " " + \
                            "RD = " + str(int(row["rank_difference"])) + " "

            txtfile.write(string_write)
            txtfile.write("\n")

    elif select == "test":
        for configuration_id, row in dataset.iterrows():
            feature = row["feature"].split('[')[-1].split(']')[0].split(', ')
            string_write = str(int(row["label"])) + " " + "qid:" + str(int(row["ID"])) + " "
            for featureid in range(len(feature)):
                string_write += str(featureid + 1) + ":" + str(feature[featureid]) + " "
            string_write += "#docid = " + str(int(row[0])) + " " + "fileid = " + str(index) + " "
            string_write += "act_rank = " + str(int(row["act_rank"])) + " " + "act_preformance = " + \
                            str(row["act_performance"]) + " " + "predicted_rank = " + str(int(row["pre_rank"])) + \
                            " " + "predicted_preformance = " + str(row["pre_performance"]) + " " + \
                            "RD = " + str(abs(int(row["pre_rank"]) - int(row["act_rank"]))) + " "

            txtfile.write(string_write)
            txtfile.write("\n")
    txtfile.close()


#########################################

def cmd(select, ranker):
    infolder = ""
    resultfolder = ""
    if select == "train":
        infolder = "trainset_txt"
        resultfolder = "rank_model"
    elif select == "test":
        infolder = "testset_txt"
        resultfolder = "rank_result_txt"

    datafolder = "../temp_data/txt/" + infolder + "/"
    folders = [datafolder + f for f in listdir(datafolder)]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)

        folderpath = "../temp_data/" + resultfolder + "/" + name

        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for txtfile in files:
            if fileindex < 10:
                name_index = str(0) + str(fileindex)
            else:
                name_index = str(fileindex)
            cmd_line = ""
            if select == "train":
                cmd_line = "java -jar RankLib.jar -train " + txtfile + " -ranker " + str(ranker) + \
                      " -metric2t NDCG@10 -metric2T ERR@10 -save " + folderpath + "/mymodel_" + name_index + ".txt"
            elif select == "test":
                model = "../temp_data/rank_model/" + name + "/mymodel_" + name_index + ".txt"
                output = folderpath + "/NewRankedLists_" + name_index + ".txt"
                cmd_line = "java -jar RankLib.jar -rank " + txtfile + " -load " + model + " -indri " + output
            os.system(cmd_line)
            print("[command]:", cmd_line)
            fileindex += 1


######################################### EDITED BY YONGFENG

def build_l2r_model(ranker_type):
    '''
    Note: execute Ranklib.jar by cmd to get LTR model on validation set(txt file)
    The detail usage of RankLib.jar please refer to website, https://sourceforge.net/p/lemur/wiki/RankLib/

    @param ranke_type, specify which ranking algorithm we use to build a model
        0: MART (gradient boosted regression tree)
        1: RankNet
        2: RankBoost
        3: AdaRank
        4: Coordinate Ascent
        6: LambdaMART
        7: ListNet
        8: Random Forests
    '''
    infolder = "trainset_txt"
    resultfolder = "rank_model"

    datafolder = "../temp_data/txt/" + infolder + "/"
    folders = [datafolder + f for f in listdir(datafolder)]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)

        folderpath = "../temp_data/" + resultfolder + "/" + name
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for txtfile in files:
            if fileindex < 10:
                name_index = str(0) + str(fileindex)
            else:
                name_index = str(fileindex)
            
            cmd_line = "java -jar RankLib.jar -train " + txtfile + " -ranker " + str(param_ranker[ranker_type]) + \
                      param_mid_cmd[ranker_type] + " -save " + folderpath + "/mymodel_" + name_index + ".txt"
            os.system(cmd_line)
            # print("[command]:", cmd_line)
            fileindex += 1


def rerank_test_set():
    infolder = "testset_txt"
    resultfolder = "rank_result_txt"

    datafolder = "../temp_data/txt/" + infolder + "/"
    folders = [datafolder + f for f in listdir(datafolder)]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)

        folderpath = "../temp_data/" + resultfolder + "/" + name
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for txtfile in files:
            if fileindex < 10:
                name_index = str(0) + str(fileindex)
            else:
                name_index = str(fileindex)

            model = "../temp_data/rank_model/" + name + "/mymodel_" + name_index + ".txt"
            output = folderpath + "/NewRankedList_" + name_index + ".txt"

            cmd_line = "java -jar RankLib.jar -rank " + txtfile + " -load " + model + " -indri " + output
            os.system(cmd_line)
            # print("[command]:", cmd_line)
            fileindex += 1


def transform_test_results():
    """
    Note: transform the .txt results into .csv results
    """
    datafolder = "../temp_data/rank_result_txt/"
    testfolder = "../temp_data/rank_by_chunk_result/testset_rank_by_chunk/"

    folders = [datafolder + folder for folder in listdir(datafolder)]
    testfolders = [testfolder + folder for folder in listdir(testfolder)]

    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        csvfolder = testfolders[folderindex]
        foldername = folder.split("/")[-1]
        print(foldername)
        result_folder = "../experiment/reconfig/" + foldername
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        txtfiles = [folder + '/' + txtfile for txtfile in listdir(folder)]
        csvfiles = [csvfolder + '/' + csvfile for csvfile in listdir(csvfolder)]
        for fileindex in range(len(txtfiles)):
            rank_list = []
            for line in open(txtfiles[fileindex], 'r'):
                rank_list.append(int(line.split(" ")[4]))
            csvfile = pd.read_csv(csvfiles[fileindex].title())
            csvfile["index"] = range(len(csvfile))

            ##############
            csvfile['index'] = csvfile['index'].astype('category')
            csvfile['index'].cat.reorder_categories(rank_list, inplace=True)
            csvfile.sort_values("index", inplace=True)
            del csvfile['index']
            ##############

            # if fileindex < 10:
            #     name_index = str(0) + str(fileindex)
            # else:
            #     name_index = str(fileindex)

            csvfile.to_csv(result_folder + '/newRankedList' + str(fileindex) + '.csv', index=False)

################################


def reconfig(ranker_type):

    print("STEP-1: predict on the validation set using the sub-train set ...\n")
    predict_on_validation_set()

    print("STEP-2: set the ID blocks of the validation set & the test set ...\n")
    block_validation_set()
    block_test_set()

    print("STEP-3: set the relative correlation(label) of the validation set ...\n")
    label_validation_set()
    label_test_set()

    print("STEP-4: transform the .csv files into .txt files that RankLib.jar can process ...\n")
    parse_config("train")
    parse_config("test")

    print("STEP-5: build the L2R model to re-rank the prediction of test set ...\n")
    build_l2r_model(ranker_type) 
    rerank_test_set()

    print("STEP-6: transform the .txt results into .csv results ... \n")
    transform_test_results()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    pd.set_option('display.width',200)
 
    reconfig(ranker_type=2) # 0,1,2,3,4,5,6,7
