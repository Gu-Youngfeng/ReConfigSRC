#!/usr/bin/python
# coding=utf-8

"""
@description:
This python file implements the Direct LTR approach. 

INPUT:
    1) sub-train set  : "../parse_data/sub_train"
    2) validation set : "../parse_data/data_split"
    3) testing set    : "../experiment/rank_based/"
OUTPUT:
    1) re-ranking results on testing set : "../experiment/direct_ltr/"

@author  : Yuntianyi
@reviewer: Yongfeng
"""

import os
import warnings
from os import listdir
import pandas as pd
from pandas import DataFrame

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
    """
    Note: transfer the SolutionHolder set to the DataFrame
    """
    x = [t.decision for t in data]
    y = [t.objective[-1] for t in data]
    data_x = DataFrame(x)
    data_x['act_performance'] = DataFrame(y)
    return data_x


def read_data(dataset):
    """
    Note: read configurations from the csv file.
          column "truly" refers to the actual performance
    """
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


def get_train_csv(data, name, index):
    """
    Note: data -> validation set, name -> project, index - > 0-49
    """
    # data['pre_performance'] = None
    data = data.sort_values(by=["act_performance"])  # sorted by the "actual performance"
    data['act_rank'] = range(1, len(data) + 1) # add the column "actual rank" 
    # data['pre_rank'] = None
    # data['rank_difference'] = None

    resultfolder = "../temp_data_compare/ltr_trainset/"

    if not os.path.exists(resultfolder + name):
        os.makedirs(resultfolder + name)
    if index < 10:
        name_index = str(0) + str(index)
    else:
        name_index = str(index)
    # generated csv file name is "../temp_data_compare/ltr_trainset/AJStats/ltr_trainset_0X.csv"
    # print("--------")
    # print(data[:10])
    data.to_csv(resultfolder + name + '/ltr_trainset_' + name_index + '.csv')


def get_validation_set():
    """
    Note: obtain the validation set of each project in each round
          these validation sets will be saved into "../temp_data_compare/ltr_trainset/{$project}/ltr_trainset_{$index}.csv"
    """
    datafolder = "../raw_data/"
    split_datafolder = "../parse_data/data_split/"

    folders = [split_datafolder + f[:-4] for f in listdir(datafolder) if ".csv" in f]  # folder = "../parse_data/data_split/AJStats"
    for folderindex in range(len(folders)):  # for each project
        folder = folders[folderindex]
        name = folder.split("/")[-1]  # name = AJStats
        print(folder)
        files_folder = [folder + '/' + f for f in listdir(folder) if "rank" in f] # files_folder = ["../parse_data/data_split/AJStats/rank_based0/", ...]
        fileindex = 0
        for file_folder in files_folder:  # for each round (50 in total)
            files = [file_folder + '/' + f for f in listdir(file_folder)]  # files = ["../train_set.csv", "../validation_set.csv", "../test_set.csv"]
            # print(files)
            validation_set = []
            for csvfile in files:
                # filename = csvfile.split('/')[-1].split('.')[0]
                if "validation_set" in csvfile:
                    validation_set = read_data(pd.read_csv(csvfile.title()))
            validation = update_data(validation_set)
            dataset_to_test = validation
            # print(dataset_to_test[:10])
            get_train_csv(dataset_to_test, name, fileindex)
            fileindex += 1

####################################  DEPRECATED BY YONGFENG
'''
rank trainset and testset csvfile by chunk
'''
def rank_by_chunk(dataset):
    dataset.rename(columns={"truly_performance": "truly"}, inplace=True)
    ranked_dataset = dataset
    ranked_dataset["ID"] = 1
    ranked_dataset = ranked_dataset.sort_values(["truly"])

    return ranked_dataset


def get_csv(data, name, index, select):
    data_ranked = rank_by_chunk(data)

    resultfolder = ""

    if select == "train":
        resultfolder = "../temp_data_compare/rank_by_chunk_result/trainset_rank_by_chunk/"
    elif select == "test":
        resultfolder = "../temp_data_compare/rank_by_chunk_result/testset_rank_by_chunk/"

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
    data_ranked.to_csv(resultfolder + name + filename + name_index + '.csv')


def chunk(select):
    datafolder = ''
    if select == "train":  # validation set
        datafolder = "../temp_data_compare/ltr_trainset/"
    elif select == "test":  # test set
        datafolder = "../experiment/rank_based/"

    folders = [datafolder + f for f in listdir(datafolder)]  # folders = ["../temp_data_compare/ltr_trainset/AJStats", ...]

    for folderindex in range(len(folders)):  # for each project/dataset
        folder = folders[folderindex]
        name = folder.split("/")[-1] # name = ltr_trainset_00.csv
        # print("folder", folder)
        files = [folder + '/' + f for f in listdir(folder)]
        # print("files", files)
        fileindex = 0
        for csvfile in files:  # for each ltr_trainset_XX.csv
            dataset = pd.read_csv(csvfile.title()).iloc[:, 1:]
            # print(dataset[:10])
            get_csv(dataset, name, fileindex, select)
            fileindex += 1


####################################  EDITED BY YONGFENG

def block_validation_set():
    """
    Note: rank the validation set in each chunk, and add columns "ID"
          configurations with a same predicted preformance share a same "ID" 
    """
    datafolder = "../temp_data_compare/ltr_trainset/"

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
    resultfolder = "../temp_data_compare/rank_by_chunk_result/trainset_rank_by_chunk/"

    if not os.path.exists(resultfolder + name):
        os.makedirs(resultfolder + name)
    if index < 10:
        name_index = str(0) + str(index)
    else:
        name_index = str(index)

    filename = "/trainset_rank_by_chunk"

    # print(data_ranked[:10])  # print the ranked data
    data_ranked.to_csv(resultfolder + name + filename + name_index + '.csv', index=False)


def rank_validation_set(dataset):
    """
    Note: rank the validation set by predicted performances and actual performances
    """
    # dataset.rename(columns={"truly_performance": "truly"}, inplace=True)

    dataset["ID"] = 1
    dataset = dataset.sort_values(by=["act_performance"])

    return dataset


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
    resultfolder = "../temp_data_compare/rank_by_chunk_result/testset_rank_by_chunk/"

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
    
    dataset["ID"] = 1
    dataset = dataset.sort_values(by=["pre_performance"])

    return dataset


#######################################  DEPRECATED BY YONGFENG
'''
preprocessing:label the trainset and testset csvfile, parse them to txtfile(for Ranklib to run)
'''
def label_config(select):
    """
    Note: replace the "truly_performance" with "truly"
    """
    fromfolder = ""
    tofolder = ""

    if select == "train":  # validation set
        fromfolder = "trainset_rank_by_chunk"
        tofolder = "trainset_label"
    elif select == "test":  # test set
        fromfolder = "testset_rank_by_chunk"
        tofolder = "testset_label"

    datafolder = "../temp_data_compare/rank_by_chunk_result/" + fromfolder + "/"
    resultfolder = "../temp_data_compare/label/" + tofolder + "/"

    folders = [datafolder + f for f in listdir(datafolder)]  # folders = ["../temp_data_compare/rank_by_chunk_result/AJStats", ...]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]  # folder = "../temp_data_compare/rank_by_chunk_result/AJStats"
        name = folder.split("/")[-1]  # name = "AJStats"
        print(folder)
        files = [folder + '/' + f for f in listdir(folder)]  # files = ["trainset_label00.csv", "trainset_label01.csv", ...]
        fileindex = 0
        for csvfile in files:
            dataset = pd.read_csv(csvfile.title())  # csvfile = "trainset_label00.csv"
            dataset.rename(columns={"truly_performance": "truly"}, inplace=True)
            do_label(dataset, name, fileindex, resultfolder)
            fileindex += 1


def do_label(dataset, name, index, resultfolder):
    """
    Note: set the column "ID"
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
        row_performance_difference = row["truly"]
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

    dataset.to_csv(datafolder + name + '/' + datafolder.split('/')[-2] + name_index + '.csv')


def list_transfer(ranklist):
    """
    Note: 
    """
    ranklist.reverse()
    dealt_list = []
    element = None
    pre_value = None
    for ids, value in enumerate(ranklist):  # BUG-1: id is the keyword
        if pre_value != value:
            pre_value = value
            element = ids + 1
            dealt_list.append(ids + 1)
        else:
            dealt_list.append(element)
    dealt_list.reverse()
    return dealt_list


#######################################  EDITED BY YONGFENG

def label_validation_set():
    """
    Note: label the validation set with column "label"
          "label" describes the correlation in each ID block
    """
    fromfolder = "trainset_rank_by_chunk"
    tofolder = "trainset_label"

    datafolder = "../temp_data_compare/rank_by_chunk_result/" + fromfolder + "/"
    resultfolder = "../temp_data_compare/label/" + tofolder + "/"

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
        row_performance_difference = row["act_performance"]
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

    datafolder = "../temp_data_compare/rank_by_chunk_result/" + fromfolder + "/"
    resultfolder = "../temp_data_compare/label/" + tofolder + "/"

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


#############################################  EDITED BY YONGFENG

def parse_config(select):
    fromfolder = ""
    tofolder = ""

    if select == "train":
        fromfolder = "trainset_label"
        tofolder = "trainset_txt"
    elif select == "test":
        fromfolder = "testset_label"
        tofolder = "testset_txt"

    datafolder = "../temp_data_compare/label/" + fromfolder + "/"
    resultfolder = "../temp_data_compare/txt/" + tofolder + "/"

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
            for featureid in range(len(row[1:-4])):
                string_write += str(featureid + 1) + ":" + str(row[featureid + 1]) + " "
            string_write += "#docid = " + str(int(row[0])) + " " + "fileid = " + str(index) + " "
            string_write += "act_rank = " + str(int(row["act_rank"])) + " " + "act_preformance = " + \
                            str(row["act_performance"])

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
                            " " + "predicted_preformance = " + str(row["pre_performance"])

            txtfile.write(string_write)
            txtfile.write("\n")
    txtfile.close()


########################################
'''
do Ranklib.jar by cmd to get LTR model and final ranking result(txtfile)
'''
def cmd(select, ranker):
    """
    Note: to execute RankLib.jar to train the Learning2Rank model, "ranker=2" denote the "RankBoost" algorithm
    """
    infolder = ""
    resultfolder = ""
    if select == "train":
        infolder = "trainset_txt"
        resultfolder = "rank_model"
    elif select == "test":
        infolder = "testset_txt"
        resultfolder = "rank_result_txt"

    datafolder = "../temp_data_compare/txt/" + infolder + "/"
    folders = [datafolder + f for f in listdir(datafolder)]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)

        folderpath = "../temp_data_compare/" + resultfolder + "/" + name

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
                model = "../temp_data_compare/rank_model/" + name + "/mymodel_" + name_index + ".txt"
                output = folderpath + "/NewRankedLists_" + name_index + ".txt"
                cmd_line = "java -jar RankLib.jar -rank " + txtfile + " -load " + model + " -indri " + output
            os.system(cmd_line)
            fileindex += 1


#########################################
'''
transfer the final ranking result txtfile to csv file
'''
def transfer():
    datafolder = "../temp_data_compare/rank_result_txt/"
    testfolder = "../temp_data_compare/rank_by_chunk_result/testset_rank_by_chunk/"

    folders = [datafolder + folder for folder in listdir(datafolder)]
    testfolders = [testfolder + folder for folder in listdir(testfolder)]

    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        csvfolder = testfolders[folderindex]
        foldername = folder.split("/")[-1]
        print(foldername)
        result_folder = "../experiment/direct_ltr/" + foldername
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

            if fileindex < 10:
                name_index = str(0) + str(fileindex)
            else:
                name_index = str(fileindex)

            csvfile.to_csv(result_folder + '/newRankedList' + name_index + '.csv', index=0)



#########################################  EDITED BY YONGFENG

def build_l2r_model(ranker):
    '''
    Note: execute Ranklib.jar by cmd to get LTR model on validation set(txt file)
    The detail usage of RankLib.jar please refer to website, https://sourceforge.net/p/lemur/wiki/RankLib/

    @param ranker, specify which ranking algorithm we use to build a model
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

    datafolder = "../temp_data_compare/txt/" + infolder + "/"
    folders = [datafolder + f for f in listdir(datafolder)]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)

        folderpath = "../temp_data_compare/" + resultfolder + "/" + name
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for txtfile in files:
            if fileindex < 10:
                name_index = str(0) + str(fileindex)
            else:
                name_index = str(fileindex)
            
            cmd_line = "java -jar RankLib.jar -train " + txtfile + " -ranker " + str(ranker) + \
                      " -save " + folderpath + "/mymodel_" + name_index + ".txt"
            os.system(cmd_line)
            # print("[command]:", cmd_line)
            fileindex += 1


def rerank_test_set():
    infolder = "testset_txt"
    resultfolder = "rank_result_txt"

    datafolder = "../temp_data_compare/txt/" + infolder + "/"
    folders = [datafolder + f for f in listdir(datafolder)]
    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        name = folder.split("/")[-1]
        print(folder)

        folderpath = "../temp_data_compare/" + resultfolder + "/" + name
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        files = [folder + '/' + f for f in listdir(folder)]
        fileindex = 0
        for txtfile in files:
            if fileindex < 10:
                name_index = str(0) + str(fileindex)
            else:
                name_index = str(fileindex)

            model = "../temp_data_compare/rank_model/" + name + "/mymodel_" + name_index + ".txt"
            output = folderpath + "/NewRankedList_" + name_index + ".txt"

            cmd_line = "java -jar RankLib.jar -rank " + txtfile + " -load " + model + " -indri " + output
            os.system(cmd_line)
            # print("[command]:", cmd_line)
            fileindex += 1


def transform_test_results():
    """
    Note: transform the .txt results into .csv results
    """
    datafolder = "../temp_data_compare/rank_result_txt/"
    testfolder = "../temp_data_compare/rank_by_chunk_result/testset_rank_by_chunk/"

    folders = [datafolder + folder for folder in listdir(datafolder)]
    testfolders = [testfolder + folder for folder in listdir(testfolder)]

    for folderindex in range(len(folders)):
        folder = folders[folderindex]
        csvfolder = testfolders[folderindex]
        foldername = folder.split("/")[-1]
        print(foldername)
        result_folder = "../experiment/direct_ltr/" + foldername
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
            csvfile.sort_values(by="pre_performance", inplace=True)

            csvfile.to_csv(result_folder + '/newRankedList' + str(fileindex) + '.csv', index=False)


def direct_ltr():

    print("STEP-1: obtain the data in the validation set ...\n")
    get_validation_set()

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
    build_l2r_model(ranker=2) # RankBoost
    rerank_test_set()

    print("STEP-6: transform the .txt results into .csv results ... \n")
    transform_test_results() 


################################
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    pd.set_option('display.width', 200)

    direct_ltr()
