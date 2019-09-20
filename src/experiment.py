#!/usr/bin/python
# coding=utf-8

"""
@description:
This python file implements the #Evaluation Process# of ReConfig approach, including
	1) basic information of each dataset/senario
	2) implementation of the measurements RD* and RDtie
	3) simulate random deletion approach
	4) comparison with the rank-based approach (RQ1)
	5) comparison with other comparative approaches (RQ2)
	6) comparison with different filter ratios (RQ3)
	7) comparison with RDTie and RD* (RQ4)
	8) visiualization of above results

INPUT:  
	1) experimental results by 6 approaches, i.e., all files under "../experiment/${method_name}/"
OUTPUT: 
	2) evaluation results on such approaches

@author  : Yongfeng
@reviewer: Yongfeng
"""

import pandas as pd
import numpy as np
import os
import sys

def calculate_min_R(proj):
	"""
	exp6: calculate the rank difference using "RankDiff" (proposed by Nair) on orignal results.
	to return the average of rank difference list of top-1,3,5,10,20.
	"""
	min_rank = []

	root_dir = "../experiment/rank_based/"

	# total = range(50)
	# rd_index = rd.sample(total, 40)

	for index in range(50):  # for each csv
	# for index in rd_index:

		path = root_dir + proj + "/rank_based" + str(index) + ".csv"
		# print(path)
		pdcontent = pd.read_csv(path)
		
		truly_rank = [pdcontent.iloc[i]["truly_rank"] for i in range(30)]
		# print("top-10 actual  rank: ", truly_rank)
		# print("top-10 minimal rank: ", np.min(truly_rank)-1)
		minR_1 = min_of_top_k(truly_rank, k=1) -1
		minR_3 = min_of_top_k(truly_rank, k=3) -1
		minR_5 = min_of_top_k(truly_rank, k=5) -1
		minR_10= min_of_top_k(truly_rank, k=10) -1
		minR_20= min_of_top_k(truly_rank, k=20) -1
		
		min_rank.append([minR_1, minR_3, minR_5, minR_10, minR_20])

	# print(min_rank)
	# result = average_of_couple_lst(min_rank)
	return min_rank


def get_split_predict(predict):
	"""
	to split the predicted list into parts according to the identical valudes.
	Input : [870, 870, 870, 870, 870, 870, 870, 930, 930, 930, 930, 950, 950]
	Output: [7, 4, 2]
	"""
	predict_spilt = []

	count = 0

	for i in range(len(predict)-1):

		if predict[i] == predict[i+1]:
			count += 1
		elif predict[i] != predict[i+1] and count != 0:
			count += 1
			predict_spilt.append(count)
			count = 0
		else:
			pass

	if count != 0:
		predict_spilt.append(count+1)

	return predict_spilt


def average_of_couple_lst(couple_lst):
	"""
	get average list of list of list, e.g.,
	Input  : [[1,2,3], [3,4,7]]
	Output : [2,3,5]
	"""
	lst_size = len(couple_lst)
	couple_size = len(couple_lst[0])

	average_couple_lst = []

	count = 0

	while count < couple_size:
		temp_sum = 0
		for i in range(lst_size):
			temp_sum += couple_lst[i][count]
		average_couple_lst.append(temp_sum/lst_size)
		count += 1

	return average_couple_lst


def remove_list_from(source_lst, remove_lst):
	"""
	to remove the elements in source_lst according to the indexes in remove_lst
	e.g., source_lst=[10,20,30,40,50,60], remove_lst=[0,3,5], then remian_lst=[20,30,50].
	"""
	remain_lst = []
	for i in range(len(source_lst)):
		if i not in remove_lst:
			remain_lst.append(source_lst[i])
		# else:
		# 	remove_lst.remove(source_lst[i])

	return remain_lst


def return_remain_ranks(actual_lst, flags, frac):
	"""
	to remove the 90% ranks in actual ranks (actual_lst) using prediction identical indexes (flags)
	If the identical block contains less than 10 configurations, then we do not remove any configuration.
	"""
	remain_size = []

	for i in range(len(flags)-1):

		added = flags[i+1]-flags[i]
		if added >= 10: # if the block size is greater than or equal to 10
			remain = round(added*(1-frac))
			remain = remain if remain >1 else 1
			# item_size.append(item)
			remain_size.append(remain)
		else: # otherwise we delete nothing
			remain_size.append(added)

	last_added = len(actual_lst) - flags[-1]
	# print(flags[-1])
	if last_added >= 10:
		last_remain = round(last_added*(1-frac))
		last_remain = last_remain if last_remain > 1 else 1
		# item_size.append(last_item)
		remain_size.append(last_remain)
	else:
		remain_size.append(last_added)

	# print("[to be removed] :", item_size)
	# print("[to be remained]:", remain_size)

	# new_lst = []

	remain_lst = []

	for idx, size in zip(flags, remain_size):
		for item in actual_lst[idx:idx+size]:
			remain_lst.append(item)
			# new_lst.append(item)

	return remain_lst


def return_remain_ranks_random(actual_lst, flags, frac=0.9):
	"""
	to remove the 90% ranks in actual ranks (actual_lst) randomly.
	If the identical block contains less than 10 configurations, then we do not remove any configuration.
	"""
	import random as rd 
	rd.seed(0)  # to control the consistence of the experiments, we set a random seed 0.

	remain_lst = []

	for i in range(len(flags)-1):

		added = flags[i+1]-flags[i]
		if added >= 10:
			remain = round(added*(1-frac))
			# print(remain, added, (100-frac))
			remain = remain if remain >1 else 1
			add_lst = rd.sample(actual_lst[flags[i]:flags[i+1]], remain) # random sampling
			for elem in add_lst:
				remain_lst.append(elem)
		else:
			for elem in actual_lst[flags[i]:flags[i+1]]:
				remain_lst.append(elem)

	last_added = len(actual_lst) - flags[-1]
	if last_added >= 10:
		last_remain = round(last_added*(1-frac))
		last_remain = last_remain if last_remain > 1 else 1

		add_lst = rd.sample(actual_lst[flags[-1]:], last_remain) # random sampling
		for elem in add_lst:
			remain_lst.append(elem)
	else:
		for elem in actual_lst[flags[-1]:]:
			remain_lst.append(elem)
	
	return remain_lst


def min_of_top_k(souce_lst, k=10):
	"""
	to return the minest value in top-k in source_lst
	"""
	return np.min(souce_lst[:k])


def P(y, x):
	"""
	to calculate the P(y, x) by formula: P(y,x)=y(y-1)...(y-x+1)
	"""
	if y < x:
		return 1
	else:
		sums = 1 
		for i in range(x):
			sums *= (y-i)
		return sums


def Pr(m, t, j):
	"""
	to calculate the Pr(m,t,j) by formula: Pr(m,t,j)=[P(m-t, j-1)xP(t, 1)]/P(m,j)
	note that j cannot be greater than (m-t+1), otherwise, Pr=0
	"""
	if j <= (m - t + 1):
		numerator = P(m-t, j-1) * P(t, 1)
		denominator = P(m, j)
		return numerator/denominator
	else:
		# print("j",j,"m",m,"t",t)
		return 0


def calculate_rds_proba_comparison(proj):
	"""
	exp7: to calculate the new rank difference RDTie using directly L2R model
	"""
	minRank_lst = [] # to save the list of rank_lst

	cmp_dir = "../experiment/direct_ltr/"

	csv_files = [file for file in os.listdir(cmp_dir + "/" + proj + "/") if ".csv" in file]
	for csv_file in csv_files:

		rank_lst = [] # to save [top-1, top-3, top-5, top-10, top-20]

		csv_file = cmp_dir + proj + "/" + csv_file

		pdcontent = pd.read_csv(csv_file, dtype={"truly": np.float32, "predicted": np.float32})

		predict = [pdcontent.iloc[i]["predicted"] for i in range(len(pdcontent))]
		actual = [pdcontent.iloc[i]["truly"] for i in range(len(pdcontent))]
		# actual_rank_nature = [pdcontent.iloc[i]["truly_rank"] for i in range(len(pdcontent))]

		actual_rank = return_rank_min(actual) 
		# print("[actual rank]:", actual_rank)

		identical_indexes = remove_list_by_fraction(predict)
		identical_indexes = [i+1 for i in identical_indexes]
		# print("[identical list]:", identical_indexes)

		k_values = [1, 3, 5, 10, 20]

		for k in k_values:
			rank_new = generate_new_rank(actual_rank, identical_indexes, k)
			# print(np.min(rank_new), end=" ")
			rank_lst.append(np.min(rank_new)-1) # add [top-1, top-3, top-5, top-10, top-20] to rank_lst
		# print("")
		minRank_lst.append(rank_lst)

	# results = average_of_couple_lst(minRank_lst) # ave of [top-1, top-3, top-5, top-10, top-20]
	# print("\n[ave]:", results)
	return minRank_lst


def return_rank_min(actual):
	"""
	to return the Rank(Rmin) list using the given actual performance list
	e.g., for input list actual=[800, 600, 600, 400, 500], the Rank(Rmin) is [5, 3, 3, 1, 2]
	"""
	actual_id = [[(i+1),p] for i,p in enumerate(actual)] # actual_id = [ [predicted rank, actual value], ...]
	# print("actual_id:\n", actual_id)

	actual_sort = sorted(actual_id, key=lambda x:x[-1])
	# print("actual_sort\n", actual_sort)

	actual_index = [[(i+1),p[0], p[1]] for i,p in enumerate(actual_sort)] # actual_index = [[actual rank, predicted rank, actual value], ...]
	# print("actual_index:\n", actual_index)

	for i in range(len(actual_index)-1): 
		if actual_index[i][-1] == actual_index[i+1][-1]: # for the repeated value
			actual_index[i+1][0] = actual_index[i][0] # assign the rank with min rank

	actual_rank_min = sorted(actual_index, key=lambda x:x[1])
	# print("rank_min_cup:\n", actual_rank_min)

	rank_min = [cup[0] for cup in actual_rank_min]
	return rank_min


def return_rank_median(actual):
	"""
	to return the Rank(Rmedian) list using the given actual performance list.
	e.g., for input list actual=[800, 600, 600, 400, 500], the Rank(Rmin) is [5, 3.5, 3.5, 1, 2]
	"""
	actual_id = [[(i+1), p] for i, p in enumerate(actual)] # actual_id = [ [predicted rank, actual value], ...]
	# print("actual_id:\n", actual_id)

	actual_sort = sorted(actual_id, key=lambda x:x[-1]) 
	# print("actual_sort\n", actual_sort)

	actual_index = [[(i+1), p[0], p[1]] for i,p in enumerate(actual_sort)] # actual_index = [[actual rank, predicted rank, actual value], ...]
	# print("actual_index:\n", actual_index)

	temp_sum = []
	flag = -1

	for i in range(len(actual_index)-1):

		if actual_index[i][-1] == actual_index[i+1][-1] and flag == -1: # for the repeated value
			flag = i
			temp_sum.append(actual_index[i][0]) # start collect rank

		elif actual_index[i][-1] == actual_index[i+1][-1] and flag != -1: # for the repeated value
			temp_sum.append(actual_index[i][0]) # continue collect rank

		elif actual_index[i][-1] != actual_index[i+1][-1] and flag != -1:
			temp_sum.append(actual_index[i][0]) # stop collect rank
			temp_mr = np.median(temp_sum)
			for j in range(len(temp_sum)): # assign the rank with median rank
				actual_index[flag+j][0] = temp_mr 
			flag = -1
			temp_sum = []

		else:
			pass

	if flag != -1:
		num_same_append = len(actual_index) - flag
		for j in range(num_same_append-1):
			temp_sum.append(actual_index[j-1][0])
		temp_mr = np.median(temp_sum)
		for j in range(num_same_append):
			actual_index[j-1][0] = temp_mr

	actual_rank_median = sorted(actual_index, key=lambda x:x[1])
	# print("rank_median_cup:\n", actual_rank_median)

	rank_median = [cup[0] for cup in actual_rank_median]
	return rank_median


def remove_list_by_fraction(source_lst):
	"""
	to return the index list to be removed
	e.g., source_lst=[100,200,200,300,300,300,300,300,400,400], then flags=[0,1,3,8]
	"""
	flags = [0]
	temp = source_lst[0]
	for i in range(len(source_lst)):
		if source_lst[i] != temp:
			flags.append(i)
			temp = source_lst[i]

	return flags


def generate_new_rank(actual_rank, identical_indexes, k):
	"""
	Note: to generate new rank, i.e., RDTie, from the given actual ranks, prediction identical indexes and top-k.
	@ param actual_rank: actual ranks
	@ param identical_indexes: prediction identical indexes
	@ param k: top-k
	"""
	if len(actual_rank) <= k: # if the size of actual rank is smaller than k
		return actual_rank

	rank_new = []
	pl = -1  # upper bound of tied configs
	m = -1  # tied nums

	for i in range(len(identical_indexes)-1):
		if identical_indexes[i+1] > k:
			m = identical_indexes[i+1] - identical_indexes[i]
			pl = identical_indexes[i]
			break

	pu = pl + m - 1  # bottom bound of tied configs
	# print("[m]:", m)
	# print("[pl]:", pl)
	# print("[pu]:", pu)

	if pu == k: # perfectly split by k
		for i in range(pu):
			rank_new.append(actual_rank[i])
	else: # pu > k
		for i in range(pl-1):
			rank_new.append(actual_rank[i])
		
		t = k - pl + 1  # nums above the k
		r_m = actual_rank[pl-1:pu]
		# print(r_m)
		r_m = sorted(r_m)
		# print(r_m)

		avg_rank = 0

		for ii in range(len(r_m)):
			try:
				avg_rank += r_m[ii] * Pr(m, t, (ii+1))
				# print(r_m[ii], "x", Pr(m, t, (ii+1)))
			except OverflowError as e: # in case of the overflow error!
				print(e)
				return []

		# print(avg_rank)
		# print("[t]:", t)
		# print("[m-t]:", (m-t))

		for i in range(t):
			rank_new.append(avg_rank)

	# print("[new ranks]:", rank_new)
	return rank_new


def calculate_RDTie(proj):
	"""
	to calculate the RDTie of dataset ${proj} using Rank based method
	do not remove any configurations
	"""
	minRank_lst = [] # to save the list of rank_lst

	origin_dir = "../experiment/rank_based/"

	for i in range(50):

		rank_lst = [] # to save [top-1, top-3, top-5, top-10, top-20]

		csv_file = origin_dir + proj + "/rank_based" + str(i) + ".csv"
		# csv_file = origin_dir + proj + "/rank_based22.csv"

		pdcontent = pd.read_csv(csv_file, dtype={"act_performance": np.float32, "pre_performance": np.float32})

		# predict = [pdcontent.iloc[i]["pre_performance"] for i in range(len(pdcontent))]
		# actual = [pdcontent.iloc[i]["act_performance"] for i in range(len(pdcontent))]

		predict = pdcontent["pre_performance"]
		actual = pdcontent["act_performance"]

		actual_rank = return_rank_min(actual) 
		# print("[actual rank]:", actual_rank)

		identical_indexes = remove_list_by_fraction(predict)
		identical_indexes = [i+1 for i in identical_indexes]
		# print("[identical list]:", identical_indexes)

		k_values = [1, 3, 5, 10, 20]

		for k in k_values:
			rank_new = generate_new_rank(actual_rank, identical_indexes, k)
			# print(np.min(rank_new), end=" ")
			rank_lst.append(np.min(rank_new)-1) # add [top-1, top-3, top-5, top-10, top-20] to rank_lst
		# print("")
		minRank_lst.append(rank_lst)
	# print(minRank_lst)
	# results = average_of_couple_lst(minRank_lst) # ave of [top-1, top-3, top-5, top-10, top-20]
	# print("\n[ave]:", results)
	return minRank_lst


def calculate_RDTie_filter(proj, method=0, frac=0.9):
	"""
	to calculate the new rank difference RDTie by filtering out the 90% last ranks
	@ param method=0, classification 
			method=1, rankdom rank
			method=2, direct_ltr
			method=3, reconfig
	@ param frac, removation ratio, default frac is 0.9
	"""
	if method > 3:
		print("[ERROR]: Parameter method should be 0, 1, 2, or 3 in function calculate_RDTie_filter()")
		return None
	minRank_lst = [] # to save the list of rank_lst

	filtered_dir = ["../experiment/classification/", # classification
					"../experiment/random_rank/", # random_rank
					"../experiment/direct_ltr", # direct_ltr
					"../experiment/reconfig/"] # reconfig

	csv_files = [file for file in os.listdir(filtered_dir[method] + "/" + proj + "/") if ".csv" in file]
	for csv_file in csv_files:
		rank_lst = [] # to save [top-1, top-3, top-5, top-10, top-20]
		# print(csv_file)
		csv_file = filtered_dir[method] + "/" + proj + "/" + csv_file
		# print(csv_file)
		pdcontent = pd.read_csv(csv_file, dtype={"act_performance": np.float32, "pre_performance": np.float32})

		# predict = [pdcontent.iloc[i]["pre_performance"] for i in range(len(pdcontent))]

		# actual = []
		# if method == 1:
		# 	actual = [pdcontent.iloc[i]["act_performance"] for i in range(len(pdcontent))]
		# else:
		# 	actual = [pdcontent.iloc[i]["act_performance"] for i in range(len(pdcontent))]

		predict = pdcontent["pre_performance"]
		actual = pdcontent["act_performance"]

		actual_rank = return_rank_min(actual) # actual rank
		# print("[actual rank]:", actual_rank)

		identical_indexes = remove_list_by_fraction(predict) # split indexes
		# print("[indexes:]", identical_indexes)

		actual_rank = return_remain_ranks(actual_rank, identical_indexes, frac)  # remove actual rank, which can be 0.9, 0.8, 0.7
		predict = return_remain_ranks(predict, identical_indexes, frac)

		# 
		identical_indexes = remove_list_by_fraction(predict)
		identical_indexes = [i+1 for i in identical_indexes]
		# print("[identical list]:", identical_indexes)

		k_values = [1, 3, 5, 10, 20]

		for k in k_values:
			rank_new = generate_new_rank(actual_rank, identical_indexes, k)
			# print(np.min(rank_new), end=" ")
			rank_lst.append(np.min(rank_new)-1) # add [top-1, top-3, top-5, top-10, top-20] to rank_lst
			# print(k,":", rank_new, ":", np.min(rank_new)-1)
		minRank_lst.append(rank_lst)
	# print(minRank_lst)

	# results = average_of_couple_lst(minRank_lst) # ave of [top-1, top-3, top-5, top-10, top-20]
	# print("\n[ave]:", results)
	return minRank_lst


def calculate_rds_proba_random(proj):
	"""
	to calculate the new rank difference by filtering out the 90% last ranks
	random deletion
	"""
	minRank_lst = [] # to save the list of rank_lst

	random_dir = "../experiment/rank_based/"

	for i in range(50):

		rank_lst = [] # to save [top-1, top-3, top-5, top-10, top-20]

		csv_file = random_dir + proj + "/rank_based" + str(i) + ".csv"
		# csv_file = origin_dir + proj + "/rank_based22.csv"

		pdcontent = pd.read_csv(csv_file, dtype={"truly_performance":np.float32, "predicted":np.float32})

		predict = [pdcontent.iloc[i]["predicted"] for i in range(len(pdcontent))]
		actual = [pdcontent.iloc[i]["truly_performance"] for i in range(len(pdcontent))]
		# actual_rank_nature = [pdcontent.iloc[i]["truly_rank"] for i in range(len(pdcontent))]

		actual_rank = return_rank_min(actual) # actual rank
		# print("[actual rank]:", actual_rank)

		identical_indexes = remove_list_by_fraction(predict) # split indexes
		# print("[indexes:]", identical_indexes)		

		actual_rank = return_remain_ranks_random(actual_rank, identical_indexes, frac=0.9) # remove actual rank
		predict = return_remain_ranks(predict, identical_indexes, frac=0.9)

		#
		identical_indexes = remove_list_by_fraction(predict)
		identical_indexes = [i+1 for i in identical_indexes]
		# print("[identical list]:", identical_indexes)

		k_values = [1, 3, 5, 10, 20]

		for k in k_values:
			rank_new = generate_new_rank(actual_rank, identical_indexes, k)
			# print(np.min(rank_new), end=" ")
			rank_lst.append(np.min(rank_new)-1) # add [top-1, top-3, top-5, top-10, top-20] to rank_lst
			# print(k,":", rank_new, ":", np.min(rank_new)-1)
		minRank_lst.append(rank_lst)

	# results = average_of_couple_lst(minRank_lst) # ave of [top-1, top-3, top-5, top-10, top-20]
	# print("\n[ave]:", results)
	return minRank_lst


def calculate_RDTie_outlier(proj):
	"""
	exp7: to calculate the new rank difference by filtering out the 90% last ranks
	:: outlier deletion
	"""
	minRank_lst = [] # to save the list of rank_lst

	ocs_dir = "../experiment/outlier_detection/"

	for i in range(50):

		rank_lst = [] # to save [top-1, top-3, top-5, top-10, top-20]

		csv_file = ocs_dir + proj + "/rank_based" + str(i) + ".csv"
		# csv_file = origin_dir + proj + "/rank_based22.csv"

		pdcontent = pd.read_csv(csv_file, dtype={"act_performance":np.float32, "pre_performance":np.float32})

		predict = [pdcontent.iloc[i]["pre_performance"] for i in range(len(pdcontent))]
		actual = [pdcontent.iloc[i]["act_performance"] for i in range(len(pdcontent))]
		anomaly_indexes = [i for i in range(len(pdcontent)) if pdcontent.iloc[i]["isAnomaly"] == -1]
		# remove the configuration labeled with -1
		# print(anomaly_indexes)

		actual_rank = return_rank_min(actual) # actual rank
		# print("[actual rank]:", actual_rank)

		actual_rank = remove_list_from(actual_rank, anomaly_indexes)
		predict = remove_list_from(predict, anomaly_indexes)
		
		# print("[actual rank]:", actual_rank)
		# print("[predicted  ]:", predict)

		identical_indexes = remove_list_by_fraction(predict) # split indexes
		# print("[indexes:]", identical_indexes)		

		# actual_rank = return_remain_ranks_random(actual_rank, identical_indexes) # remove actual rank
		# predict = return_remain_ranks(predict, identical_indexes)
		# identical_indexes = remove_list_by_fraction(predict)
		# print("[remain rank   ]:", actual_rank)
		# print("[remain predict]:", predict)
		# print("[indexes:]", identical_indexes)

		identical_indexes = [i+1 for i in identical_indexes]
		# print("[identical list]:", identical_indexes)

		k_values = [1, 3, 5, 10, 20]

		for k in k_values:
			rank_new = generate_new_rank(actual_rank, identical_indexes, k)
			# print(np.min(rank_new), end=" ")
			rank_lst.append(np.min(rank_new)-1) # add [top-1, top-3, top-5, top-10, top-20] to rank_lst
			# print(k,":", rank_new, ":", np.min(rank_new)-1)
		minRank_lst.append(rank_lst)

	# results = average_of_couple_lst(minRank_lst) # ave of [top-1, top-3, top-5, top-10, top-20]
	# print("\n[ave]:", results)
	return minRank_lst


def catch_info_from_line(line):
	"""
	read result data from one line, we keep 3 digits after dicimal point.
	eg. for a line string, "Apache:[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]" will be translated to 2-dimensional array datas
	ouputs: datas = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
	"""
	start_id = line.index("[[")+2
	end_id = line.index("]]")
	data_lst = line[start_id:end_id]
	results = data_lst.split("], [") # list of 50 trails
	# print(len(results))
	
	datas = []

	for trails in results: # each trail has 5 rank differences(top1,3,5,10,20)
		data = [round(float(i), 3) for i in trails.split(", ")]
		datas.append(data)
	# print(datas)

	return datas


def obtain_results_from_result(path):
	"""
	to return the data results from the file in path
	results = [p1_top1, p1_top3, p1_top5, p1_top20, p1_top20,
	           p2_top1, p2_top3, p2_top5, p2_top20, p2_top20,
	           ... ] 
	"""
	file = open(path)
	lines = file.readlines()

	results = []

	for line in lines:
		if ":" in line:
			datas = catch_info_from_line(line) # 2-dimensinal array
			top1 = [datas[i][0] for i in range(len(datas))] # top1 list for this project
			top3 = [datas[i][1] for i in range(len(datas))] # top3 list for this project
			top5 = [datas[i][2] for i in range(len(datas))] # top5 list for this project
			top10= [datas[i][3] for i in range(len(datas))] # top10 list for this project
			top20= [datas[i][4] for i in range(len(datas))] # top20 list for this project
			results.append(top1)
			results.append(top3)
			results.append(top5)
			results.append(top10)
			results.append(top20)
			
	# print(results)
	return results


def get_wdl_bet_lst(lst1, lst2):
	"""
	get win, draw, loss from two list, lst1 and lst2.
	"""
	win = draw = loss = 0
	for i,j in zip(lst1, lst2):
		if i>j:
			win+=1
		elif i==j:
			draw+=1
		else:
			loss+=1

	return [win, draw, loss]


def compare_result_bet_methods(projs, rank_base_path, reconfig_path):
	"""
	to compare the rank differences on 36 datasets using  and the rank-bsed approach
	:: path1 = results on the rank-based approach, path2 = results on ReConfig
	"""
	results_1 = obtain_results_from_result(rank_base_path)
	results_2 = obtain_results_from_result(reconfig_path)

	indx_1 = 0
	indx_3 = 1
	indx_5 = 2
	indx_10= 3
	indx_20= 4

	new_projs = []
	for proj in projs:
		if os.path.exists("../raw_data/"+proj+".csv"):
			new_projs.append(proj)
	proj_id = 0

	print("Results: RDTie on 50 dataset using the rank-based(RaB) and ReConfig(ReC) approach\n")
	print("1) We calculate the RDTie of each dataset in cases of | Top-1 | Top-3 | Top-5 | Top-10 |.")
	print("2) We compare the RDTie of these two appraoch in 50 repeats | mean(Rank-based) | mean(ReConfig) | win /draw/ loss |.\n")

	print("| %-18s | %-8s | %-8s | %-2s / %-2s/ %-2s| %-8s | %-8s | %-2s / %-2s/ %-2s| %-8s | %-8s | %-2s / %-2s/ %-2s| %-8s | %-8s | %-2s / %-2s/ %-2s|"%("Datasets","RaB(1)","ReC(1)","W","D","L","RaB(3)","ReC(3)","W","D","L","RaB(5)","ReC(5)","W","D","L","RaB(10)","ReC(10)","W","D","L"))
	print("----------------------")
	while indx_1 < len(results_1):

		win1_lst = get_wdl_bet_lst(results_1[indx_1], results_2[indx_1])
		win3_lst = get_wdl_bet_lst(results_1[indx_3], results_2[indx_3])
		win5_lst = get_wdl_bet_lst(results_1[indx_5], results_2[indx_5])
		win10_lst = get_wdl_bet_lst(results_1[indx_10], results_2[indx_10])
		# win20_lst = get_wdl_bet_lst(results_1[indx_20], results_2[indx_20])

		### format as you like, e.g., OR-1, CR-1, win/draw/loss | OR-3, CR-3, win/draw/loss | OR-5, CR-5, win/draw/loss | OR-10, CR-10, win/draw/loss |
		# print(indx_1)
		print("| %-18s " % new_projs[proj_id], end="| ")
		print("%-8.3f | %-8.3f | %-2d / %-2d/ %-2d"%(np.mean(results_1[indx_1]), np.mean(results_2[indx_1]), win1_lst[0], win1_lst[1], win1_lst[2]), end="| ")
		print("%-8.3f | %-8.3f | %-2d / %-2d/ %-2d"%(np.mean(results_1[indx_3]), np.mean(results_2[indx_3]), win3_lst[0], win3_lst[1], win3_lst[2]), end="| ")
		print("%-8.3f | %-8.3f | %-2d / %-2d/ %-2d"%(np.mean(results_1[indx_5]), np.mean(results_2[indx_5]), win5_lst[0], win5_lst[1], win5_lst[2]), end="| ")
		print("%-8.3f | %-8.3f | %-2d / %-2d/ %-2d |"%(np.mean(results_1[indx_10]), np.mean(results_2[indx_10]), win10_lst[0], win10_lst[1], win10_lst[2]))

		# E.g., print results in Tex format
		# print("%s &" % projs[proj_id], end=" ")
		# print("%.3f & %.3f & %d/%d/%d &" % (np.mean(results_1[indx_1]), np.mean(results_2[indx_1]), win1_lst[0], win1_lst[1], win1_lst[2]), end=" ")
		# print("%.3f & %.3f & %d/%d/%d &" % (np.mean(results_1[indx_3]), np.mean(results_2[indx_3]), win3_lst[0], win3_lst[1], win3_lst[2]), end=" ")
		# print("%.3f & %.3f & %d/%d/%d &" % (np.mean(results_1[indx_5]), np.mean(results_2[indx_5]), win5_lst[0], win5_lst[1], win5_lst[2]), end=" ")
		# print("%.3f & %.3f & %d/%d/%d \\\\" % (np.mean(results_1[indx_10]), np.mean(results_2[indx_10]), win10_lst[0], win10_lst[1], win10_lst[2]))
		# print("\\hline")
		
		indx_1 += 5
		indx_3 += 5
		indx_5 += 5
		indx_10 += 5
		indx_20 += 5
		proj_id += 1


def compare_result_six_methods(paths):
	"""
	compare rank differences between 6 methods
	:: paths= [ReConfig, Rank_based, Outlier detection, Classification, Random deletion, Direct LTR]
	"""
	results_cr = obtain_results_from_result(paths[0])
	results_rk = obtain_results_from_result(paths[1])
	results_od = obtain_results_from_result(paths[2])
	results_cf = obtain_results_from_result(paths[3])
	results_rd = obtain_results_from_result(paths[4])
	results_dl = obtain_results_from_result(paths[5])

	indx_1 = 0
	indx_3 = 1
	indx_5 = 2
	indx_10= 3
	indx_20= 4

	projs1 = ['rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
	# boolean projects
	projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']
	projs = projs1 + projs2
	proj_id = 0

	print("Rank differences on 36 scenarios using Reconfig and other comparative approaches\n")
	print("1) We calculate the rank difference of each scenario in cases of | Top-1 | Top-3 | Top-5 | Top-10 |.")
	print("2) The order of comparative approaches is | ReConfig, Rank_based, Outlier detection, Classification, Random deletion, Direct LTR |.\n")

	while indx_1 < len(results_cr):

		print("| %-17s |" % projs[proj_id], end=" ")
		print("%.1f %.1f %.1f %.1f %.1f %.1f "%(np.mean(results_cr[indx_1]), np.mean(results_rk[indx_1]), np.mean(results_od[indx_1]), np.mean(results_cf[indx_1]), np.mean(results_rd[indx_1]), np.mean(results_dl[indx_1])), end="| ")
		print("%.1f %.1f %.1f %.1f %.1f %.1f "%(np.mean(results_cr[indx_3]), np.mean(results_rk[indx_3]), np.mean(results_od[indx_3]), np.mean(results_cf[indx_3]), np.mean(results_rd[indx_3]), np.mean(results_dl[indx_3])), end="| ")
		print("%.1f %.1f %.1f %.1f %.1f %.1f "%(np.mean(results_cr[indx_5]), np.mean(results_rk[indx_5]), np.mean(results_od[indx_5]), np.mean(results_cf[indx_5]), np.mean(results_rd[indx_5]), np.mean(results_dl[indx_5])), end="| ")
		print("%.1f %.1f %.1f %.1f %.1f %.1f|"%(np.mean(results_cr[indx_10]), np.mean(results_rk[indx_10]), np.mean(results_od[indx_10]), np.mean(results_cf[indx_10]), np.mean(results_rd[indx_10]), np.mean(results_dl[indx_10])))		
		indx_1 += 5
		indx_3 += 5
		indx_5 += 5
		indx_10 += 5
		indx_20 += 5
		proj_id += 1

	# results in TeX format
	# while indx_1 < len(results_cr):
	# 	print("%s &" % projs[proj_id], end=" ")
	# 	print("%.3f & %.3f & %.3f & %.3f & %.3f & %.3f &" % (np.mean(results_cr[indx_1]), np.mean(results_rk[indx_1]), 
	# 															np.mean(results_od[indx_1]), np.mean(results_cf[indx_1]), 
	# 															np.mean(results_rd[indx_1]), np.mean(results_dl[indx_1])), end=" ")
	# 	print("%.3f & %.3f & %.3f & %.3f & %.3f & %.3f &" % (np.mean(results_cr[indx_3]), np.mean(results_rk[indx_3]), 
	# 															np.mean(results_od[indx_3]), np.mean(results_cf[indx_3]), 
	# 															np.mean(results_rd[indx_3]), np.mean(results_dl[indx_3])), end=" ")
	# 	print("%.3f & %.3f & %.3f & %.3f & %.3f & %.3f &" % (np.mean(results_cr[indx_5]), np.mean(results_rk[indx_5]), 	
	# 															np.mean(results_od[indx_5]), np.mean(results_cf[indx_5]), 
	# 															np.mean(results_rd[indx_5]), np.mean(results_dl[indx_5])), end=" ")
	# 	print("%.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (np.mean(results_cr[indx_10]), np.mean(results_rk[indx_10]), 
	# 															np.mean(results_od[indx_10]), np.mean(results_cf[indx_10]), 
	# 															np.mean(results_rd[indx_10]), np.mean(results_dl[indx_10])))
	# 	print("\\hline")		
	# 	indx_1 += 5
	# 	indx_3 += 5
	# 	indx_5 += 5
	# 	indx_10 += 5
	# 	indx_20 += 5
	# 	proj_id += 1


def compare_results_of_four_methods(paths):
	"""
	compare rank differences between 6 methods
	:: paths= [ReConfig, Rank_based, Classification, Random deletion]
	"""
	results_rc = obtain_results_from_result(paths[0])
	results_rb = obtain_results_from_result(paths[1])
	# results_od = obtain_results_from_result(paths[2])
	results_cf = obtain_results_from_result(paths[2])
	results_rd = obtain_results_from_result(paths[3])
	# results_dl = obtain_results_from_result(paths[5])

	indx_1 = 0
	indx_3 = 1
	indx_5 = 2
	indx_10= 3
	indx_20= 4

	# projs1 = ['rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
	# # boolean projects
	# projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']
	# projs = projs1 + projs2
	# proj_id = 0

	projs = []
	proj_id = 0
	with open(paths[0]) as fi:
		lines = fi.readlines()
		for line in lines:
			projs.append(line.split(":[[")[0])

	print("Results: RDTie on 50 datasets using ReConfig, rank-based, classification, and random_rank approaches\n")
	print("1) We calculate the RDTie of each dataset in cases of | Top-1 | Top-3 | Top-5 | Top-10 |.")
	print("2) The order of comparative approaches is | ReConfig, rank-based, classification, random rank |.\n")

	print("| %-18s | %-5s %-5s %-5s %-5s | %-5s %-5s %-5s %-5s | %-5s %-5s %-5s %-5s | %-5s %-5s %-5s %-5s |"%("Datasets","ReC","RaB","ClF","RaR","ReC","RaB","ClF","RaR","ReC","RaB","ClF","RaR","ReC","RaB","ClF","RaR"))
	print("----------------------")
	while indx_1 < len(results_rc):

		print("| %-18s |" % projs[proj_id], end=" ")
		print("%-5.1f %-5.1f %-5.1f %-5.1f "%(np.mean(results_rc[indx_1]), np.mean(results_rb[indx_1]), np.mean(results_cf[indx_1]), np.mean(results_rd[indx_1])), end="| ")
		print("%-5.1f %-5.1f %-5.1f %-5.1f "%(np.mean(results_rc[indx_3]), np.mean(results_rb[indx_3]), np.mean(results_cf[indx_3]), np.mean(results_rd[indx_3])), end="| ")
		print("%-5.1f %-5.1f %-5.1f %-5.1f "%(np.mean(results_rc[indx_5]), np.mean(results_rb[indx_5]), np.mean(results_cf[indx_5]), np.mean(results_rd[indx_5])), end="| ")
		print("%-5.1f %-5.1f %-5.1f %-5.1f |"%(np.mean(results_rc[indx_10]), np.mean(results_rb[indx_10]), np.mean(results_cf[indx_10]), np.mean(results_rd[indx_10])))		
		indx_1 += 5
		indx_3 += 5
		indx_5 += 5
		indx_10 += 5
		indx_20 += 5
		proj_id += 1


def compare_result_two_metrics(paths):
	"""
	compare rank differences between 6 methods, e.g, paths= [Origin, Nair]
	"""
	results_pr = obtain_results_from_result(paths[0])
	results_rd = obtain_results_from_result(paths[1])

	indx_1 = 0
	indx_3 = 1
	indx_5 = 2
	indx_10= 3
	indx_20= 4

	projs1 = ['rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
	# boolean projects
	projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']
	projs = projs1 + projs2
	proj_id = 0

	print("Rank differences on 36 scenarios using RDTie and RD*\n")
	print("1) We calculate the rank difference of each scenario in cases of | Top-1 || Top-3 || Top-5 || Top-10 |.")
	print("2) We compare the maximum, minimum, mean, stand devition of two measurements in | max min mean std |.\n")

	while indx_1 < len(results_pr):

		print("| %s |" % projs[proj_id], end=" ")
		print(" %.3f  %.3f  %.3f  %.3f | %.3f  %.3f  %.3f  %.3f ||" % (np.min(results_pr[indx_1]), np.max(results_pr[indx_1]), 
				np.mean(results_pr[indx_1]), np.std(results_pr[indx_1]), np.min(results_rd[indx_1]), np.max(results_rd[indx_1]), np.mean(results_rd[indx_1]), np.std(results_rd[indx_1])), end=" ")
		print(" %.3f  %.3f  %.3f  %.3f | %.3f  %.3f  %.3f  %.3f ||" % (np.min(results_pr[indx_3]), np.max(results_pr[indx_3]), 
				np.mean(results_pr[indx_3]), np.std(results_pr[indx_3]), np.min(results_rd[indx_3]), np.max(results_rd[indx_3]), np.mean(results_rd[indx_3]), np.std(results_rd[indx_3])), end=" ")
		print(" %.3f  %.3f  %.3f  %.3f | %.3f  %.3f  %.3f  %.3f ||" % (np.min(results_pr[indx_5]), np.max(results_pr[indx_5]), 
				np.mean(results_pr[indx_5]), np.std(results_pr[indx_5]), np.min(results_rd[indx_5]), np.max(results_rd[indx_5]), np.mean(results_rd[indx_5]), np.std(results_rd[indx_5])), end=" ")
		print(" %.3f  %.3f  %.3f  %.3f | %.3f  %.3f  %.3f  %.3f ||" % (np.min(results_pr[indx_10]), np.max(results_pr[indx_10]), 
				np.mean(results_pr[indx_10]), np.std(results_pr[indx_10]), np.min(results_rd[indx_10]), np.max(results_rd[indx_10]), np.mean(results_rd[indx_10]), np.std(results_rd[indx_10])))

		# Tex format
		# print("%s &" % projs[proj_id], end=" ")
		# print(" %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f &" % (np.min(results_pr[indx_1]), np.max(results_pr[indx_1]), 
		# 		np.mean(results_pr[indx_1]), np.std(results_pr[indx_1]), 
		# 		np.min(results_rd[indx_1]), np.max(results_rd[indx_1]), np.mean(results_rd[indx_1]), np.std(results_rd[indx_1])), end=" ")
		# print("%.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f &" % (np.min(results_pr[indx_3]), np.max(results_pr[indx_3]), 
		# 		np.mean(results_pr[indx_3]), np.std(results_pr[indx_3]), 
		# 		np.min(results_rd[indx_3]), np.max(results_rd[indx_3]), np.mean(results_rd[indx_3]), np.std(results_rd[indx_3])), end=" ")
		# print("%.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f &" % (np.min(results_pr[indx_5]), np.max(results_pr[indx_5]), 
		# 		np.mean(results_pr[indx_5]), np.std(results_pr[indx_5]), 
		# 		np.min(results_rd[indx_5]), np.max(results_rd[indx_5]), np.mean(results_rd[indx_5]), np.std(results_rd[indx_5])), end=" ")
		# print("%.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (np.min(results_pr[indx_10]), np.max(results_pr[indx_10]), 
		# 		np.mean(results_pr[indx_10]), np.std(results_pr[indx_10]), 
		# 		np.min(results_rd[indx_10]), np.max(results_rd[indx_10]), np.mean(results_rd[indx_10]), np.std(results_rd[indx_10])))
		# print("\\hline")	

		indx_1 += 5
		indx_3 += 5
		indx_5 += 5
		indx_10 += 5
		indx_20 += 5
		proj_id += 1


def compare_result_fraction_parameters(paths):
	"""
	compare rank differences with different filter ratios, e.g, paths= [ReConfig-90, ReConfig-80, ReConfig-70, ReConfig-60]
	"""
	results_cfgrank_9 = obtain_results_from_result(paths[0])
	results_cfgrank_8 = obtain_results_from_result(paths[1])
	results_cfgrank_7 = obtain_results_from_result(paths[2])
	results_cfgrank_6 = obtain_results_from_result(paths[3])

	indx_1 = 0
	indx_3 = 1
	indx_5 = 2
	indx_10= 3
	indx_20= 4

	projs1 = ['rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
	# boolean projects
	projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']
	projs = projs1 + projs2
	proj_id = 0

	print("Rank differences using ReConfig with different filter ratios (90%, 80%, 70%, 60%)\n")
	print("1) We calculate the rank difference of each scenario in cases of | Top-1 | Top-3 | Top-5 | Top-10 |.")
	print("2) The order of filter ratios is | 90%  80%  70% |.\n")

	while indx_1 < len(results_cfgrank_9):
		print("| %-17s |" % projs[proj_id], end=" ")
		print(" %8.3f  %8.3f  %8.3f |" % (np.mean(results_cfgrank_9[indx_1]), np.mean(results_cfgrank_8[indx_1]), np.mean(results_cfgrank_7[indx_1])), end=" ")
		print(" %8.3f  %8.3f  %8.3f |" % (np.mean(results_cfgrank_9[indx_3]), np.mean(results_cfgrank_8[indx_3]), np.mean(results_cfgrank_7[indx_3])), end=" ")
		print(" %8.3f  %8.3f  %8.3f |" % (np.mean(results_cfgrank_9[indx_5]), np.mean(results_cfgrank_8[indx_5]), np.mean(results_cfgrank_7[indx_5])), end=" ")
		print(" %8.3f  %8.3f  %8.3f |" % (np.mean(results_cfgrank_9[indx_10]), np.mean(results_cfgrank_8[indx_10]), np.mean(results_cfgrank_7[indx_10])))
		
		indx_1 += 5
		indx_3 += 5
		indx_5 += 5
		indx_10 += 5
		indx_20 += 5
		proj_id += 1

	# Tex format 
	# while indx_1 < len(results_cfgrank_9):
	# 	print("%s &" % projs[proj_id], end=" ")
	# 	print(" %.3f & %.3f & %.3f & %.3f &" % (np.mean(results_cfgrank_9[indx_1]), np.mean(results_cfgrank_8[indx_1]), np.mean(results_cfgrank_7[indx_1]), np.mean(results_cfgrank_6[indx_1])), end=" ")
	# 	print(" %.3f & %.3f & %.3f & %.3f &" % (np.mean(results_cfgrank_9[indx_3]), np.mean(results_cfgrank_8[indx_3]), np.mean(results_cfgrank_7[indx_3]), np.mean(results_cfgrank_6[indx_3])), end=" ")
	# 	print(" %.3f & %.3f & %.3f & %.3f &" % (np.mean(results_cfgrank_9[indx_5]), np.mean(results_cfgrank_8[indx_5]), np.mean(results_cfgrank_7[indx_5]), np.mean(results_cfgrank_6[indx_5])), end=" ")
	# 	print(" %.3f & %.3f & %.3f & %.3f \\\\" % (np.mean(results_cfgrank_9[indx_10]), np.mean(results_cfgrank_8[indx_10]), np.mean(results_cfgrank_7[indx_10]), np.mean(results_cfgrank_6[indx_10])))
	# 	print("\\hline")
		
	# 	indx_1 += 5
	# 	indx_3 += 5
	# 	indx_5 += 5
	# 	indx_10 += 5
	# 	indx_20 += 5
	# 	proj_id += 1


def caluculate_option_ratio(proj, value=0):
	"""
	to calculate the average value ratio of each project
	"""
	project = "../raw_data/" + proj + ".csv"
	pdcontent = pd.read_csv(project)
	
	attr = pdcontent.columns
	feas = attr[:-1]
	perf = attr[-1]
	sortedcontent = pdcontent.sort_values(perf)
	# print(sortedcontent.iloc[0][perf])

	best_config = []
	count = 1

	best_perf = sortedcontent.iloc[0][perf]
	best_config.append(sortedcontent.iloc[0])
	print(">>", sortedcontent.iloc[0][feas].tolist(), ":", sortedcontent.iloc[0][perf])
	# print(best_perf)

	while sortedcontent.iloc[count][perf] == best_perf:
		best_config.append(sortedcontent.iloc[count])
		config = sortedcontent.iloc[count]
		print(">>", config[feas].tolist(), ":", config[perf])
		count += 1

	zero_lst = []
	
	for feature in best_config:

		zero_count=feature[feas].tolist().count(value)					
		zero_lst.append(zero_count)
	
	zero_ratio = np.mean(zero_lst)/len(feas)

	return zero_ratio


def add_by_elems(lst1, lst2):
	"""
	synthesize a new list which is the sum of each elems in 2 lists
	"""
	if len(lst1) != len(lst2):
		print("[ERROR]: Two lists should be in same size.")
		return []

	synthesize_lst = []
	for i in range(len(lst1)):
		synthesize_lst.append(lst1[i] + lst2[i])

	return synthesize_lst


def transform_to_list(str_elem):
	"""
	to transform the string elems into number list, e.g.,
	INPUT  : '[1, 3, 4, 5, 6]'
	OUTPUT : [1, 3, 4, 5, 6]
	"""
	elems = []

	str_data = str_elem[1:-1]
	# print(str_data)

	datas = str_data.split(", ")

	for data in datas:
		elems.append(float(data))

	return elems


def count_zero_in_elem(str_elem, value=0.0):
	"""
	to count the zero count in string elems, e.g.,
	INPUT  : '[0.0, 3.0, 2.0, 0.0]'
	OUTPUT : 0.5
	"""
	count = 0
	elems = transform_to_list(str_elem)
	for data in elems:
		if data == value:
			count += 1
	return count


def calculate_train_size():
	"""
	to calculate the sub_train_set of each project
	"""
	import os
	# projs = ["Apache", "Hipacc", "SQL", "rs-6d-c3-obj1", "wc-6d-c1-obj1"]
	# numeric projects
	projs1 = ['rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
	# boolean projects
	projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']

	ave_sub_train = []
	projs = projs1 + projs2
	for proj in projs:
		root_dir = "../experiment/parse_data/sub_train/" + proj + "/"
		lst_name = [i for i in os.listdir(root_dir)]
		train_set = []
		for csv in lst_name:
			pdcontent = pd.read_csv(root_dir + csv)
			train_set.append(len(pdcontent))
		print(train_set, ":", np.mean(train_set))
		# print(np.mean(train_set))
		ave_sub_train.append(np.mean(train_set))
	print(ave_sub_train)
	#[39.100000000000001, 38.280000000000001, 37.219999999999999, 45.140000000000001, 26.84, 27.52, 24.82, 28.359999999999999, 28.039999999999999, 24.66, 34.0, 33.700000000000003, 38.18, 39.299999999999997, 34.579999999999998, 41.280000000000001, 34.280000000000001, 38.039999999999999, 35.280000000000001, 33.619999999999997, 26.859999999999999, 27.699999999999999, 57.840000000000003, 26.48, 39.939999999999998, 29.960000000000001, 26.620000000000001, 33.219999999999999, 39.799999999999997, 33.5, 31.219999999999999, 34.359999999999999, 29.84, 27.559999999999999, 42.740000000000002, 30.440000000000001]
	return ave_sub_train


def find_negative_tags():
	"""
	to find ratio of samples labelled as "negative" in validation set
	"""
	projs1 = ['rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
	projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']

	projs = projs1 + projs2
	# projs = ["Apache"]
	ave_anomaly_list = []
	for proj in projs:
		anomaly_list = []
		for i in range(50):
			csv_file = "../experiments/classification/validation_tags/" + proj + "/rank_based" + str(i) + ".csv"
			pdcontent = pd.read_csv(csv_file)
			anomaly_indexes = [i for i in range(len(pdcontent)) if pdcontent.iloc[i]["isAnomaly"] == -1]
			anomaly_size = len(anomaly_indexes)

			print(anomaly_size, end=",")
			if anomaly_size > 0:
				anomaly_list.append(anomaly_size/len(pdcontent))

		print("\n>>",np.mean(anomaly_list))
		ave_anomaly_list.append(np.mean(anomaly_list))

	print(ave_anomaly_list)


def find_outlier_median_rank():
	"""
	to find average median ranks that predicted as "outlier" in testing set
	"""
	projs1 = ['rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
	projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']

	projs = projs1 + projs2
	proj_anomaly = []
	for proj in projs:
		anomaly_list = []
		for i in range(50):
			csv_file = "../experiment/outlier_detection/" + proj + "/rank_based" + str(i) + ".csv"
			pdcontent = pd.read_csv(csv_file)
			anomaly_indexes = [i for i in range(len(pdcontent)) if pdcontent.iloc[i]["isAnomaly"] == -1]
			# anomaly_size = len(anomaly_indexes)
			
			if len(anomaly_indexes) == 0:
				continue

			median_anomaly = anomaly_indexes[0]+1
			anomaly_list.append(median_anomaly)

		print(np.mean(anomaly_list), len(anomaly_list))
		proj_anomaly.append(np.mean(anomaly_list))

	print(proj_anomaly)


def get_performance_distribution(projs, proj_id):
	"""
	to get the distribution of performance in project {$proj_id}
	"""
	import matplotlib.pyplot as plt

	new_projs = []

	for proj in projs:
		proj_path = "../raw_data/" + proj + ".csv"
		if os.path.exists(proj_path):
			new_projs.append(proj)

	if proj_id >= len(new_projs):
		print("[IndexOutOfBounds Error]: %d is bigger than the size of project set."%proj_id)
		return
	file_path = "../raw_data/" + new_projs[proj_id] + ".csv"

	pdcontent = pd.read_csv(file_path)
	print(new_projs[proj_id])

	perfs = pdcontent.iloc[:,-1].tolist()
	perfs = sorted(perfs)
	x = range(len(perfs))
	# print(perfs)

	### line plot
	plt.figure(figsize=(8,4))
	plt.plot(x, perfs, color="black", linewidth=1)
	plt.scatter(x[0], perfs[0], color="#3a923c", marker="o", linewidth=5)
	plt.scatter(x[-1], perfs[-1], color="#c8393f", marker="o", linewidth=5)
	plt.title("performance distribution of project " + new_projs[proj_id])
	plt.ylabel("Performance \n(XX - in XX)", fontsize=12)  # performance type
	plt.xlabel("Configurations (sorted by performance)", fontsize=12)
	# plt.ylim(0,60000)  # limitation of y-axis and x-axis
	# plt.xlim(0,2500)

	str_a = "A ("+str(perfs[0])+")"
	str_b = "B ("+str(perfs[-1])+")"
	# str_a = "A (4422.172)"
	# str_b = "B (58091.45)"	
	plt.text(x[0], perfs[0], str_a, fontdict={"color":"#3a923c", "weight":"bold"})  # position of A and B
	plt.text(x[-1], perfs[-1], str_b, fontdict={"color":"#c8393f", "weight":"bold"})
	plt.show()

	### hist plot

	# plt.hist(perfs,len(x)//100)
	# plt.title("performance distribution of project " + projs[proj_id])
	# plt.ylabel("occurence")
	# plt.xlabel("performance")
	# plt.show()


def return_all_ranks(results, proj_id):

	top_1 = results[proj_id * 5]
	top_3 = results[proj_id * 5 + 1]
	top_5 = results[proj_id * 5 + 2]
	top_10 = results[proj_id * 5 + 3]
	top_20 = results[proj_id * 5 + 4]

	return [top_1, top_3, top_5, top_10, top_20]


def comparison_with_other_approach(proj_id):
	"""
	to get the comprarison experiemnt with other 6 approaches in one project
	"""
	dir_confrank = "../experiment/results/LTR_RDs.txt"
	dir_origin = "../experiment/results/Origin_RDs.txt"

	dir_ocs = "../experiment/results/OCS_RDs.txt"
	dir_rf = "../experiment/results/RF_RDs.txt"
	dir_rd = "../experiment/results/Random_RDs.txt"
	dir_l2r = "../experiment/results/LTR_DIR_RDs.txt"

	confrank_results = obtain_results_from_result(dir_confrank)
	confrank_data = return_all_ranks(confrank_results, proj_id)

	origin_results = obtain_results_from_result(dir_origin)
	origin_data = return_all_ranks(origin_results, proj_id)

	ocs_results = obtain_results_from_result(dir_ocs)
	ocs_data = return_all_ranks(ocs_results, proj_id)

	rf_results = obtain_results_from_result(dir_rf)
	rf_data = return_all_ranks(rf_results, proj_id)

	rd_results = obtain_results_from_result(dir_rd)
	rd_data = return_all_ranks(rd_results, proj_id)

	l2r_results = obtain_results_from_result(dir_l2r)
	l2r_data = return_all_ranks(l2r_results, proj_id)

	import matplotlib.pyplot as plt
	import seaborn as sns

	datas=[confrank_data, origin_data, ocs_data, rf_data, rd_data, l2r_data]

	# sns.boxplot(data=datas, linewidth=2.5)
	approaches = ["(a) ReConfig", "(b) Rank-based", "(c) Outlier detection", "(d) Classification", "(e) Random deletion", "(f) Direct LTR"]
	# approaches = ["CfgRank", "Rank-based", "OD", "CF", "RD", "DL"]
	grouped_data = pd.DataFrame(columns=["approaches", "top_k", "rank_difference"], index=None)
	# print(grouped_data)
	tops = [1,3,5,10]
	count = 0
	for app in range(len(approaches)):
		for k in range(len(tops)):
			for i in range(50):
				grouped_data.loc[count] = [approaches[app], tops[k], datas[app][k][i]]
				count +=1

	# print(grouped_data)
	plt.figure(figsize=(12,6))
	# plt.subplots_adjust(wspace=0.12, bottom=0.15)
	# sns.set_style("ticks", {"legend.frameon":True})
	sns.set_style("ticks")
	sns.boxplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_data, orient="v", width=0.8)
	plt.yscale("log")
	plt.xlabel("", fontsize=14)
	plt.xticks([0,1,2,3], tuple(["\nTop-1","\nTop-3","\nTop-5","\nTop-10"]), fontsize=14)
	plt.ylabel("RDTie", fontsize=14)
	plt.legend(ncol=3, bbox_to_anchor=(1.02, 1.15), fontsize=14)

	plt.figtext(0.150, 0.07, "(a)", fontsize=14)
	plt.figtext(0.175, 0.07, "(b)", fontsize=14)
	plt.figtext(0.200, 0.07, "(c)", fontsize=14)
	plt.figtext(0.225, 0.07, "(d)", fontsize=14)
	plt.figtext(0.250, 0.07, "(e)", fontsize=14)
	plt.figtext(0.276, 0.07, "(f)", fontsize=14)

	plt.figtext(0.345, 0.07, "(a)", fontsize=14)
	plt.figtext(0.370, 0.07, "(b)", fontsize=14)
	plt.figtext(0.395, 0.07, "(c)", fontsize=14)
	plt.figtext(0.422, 0.07, "(d)", fontsize=14)
	plt.figtext(0.447, 0.07, "(e)", fontsize=14)
	plt.figtext(0.472, 0.07, "(f)", fontsize=14)

	plt.figtext(0.540, 0.07, "(a)", fontsize=14)
	plt.figtext(0.565, 0.07, "(b)", fontsize=14)
	plt.figtext(0.590, 0.07, "(c)", fontsize=14)
	plt.figtext(0.617, 0.07, "(d)", fontsize=14)
	plt.figtext(0.642, 0.07, "(e)", fontsize=14)
	plt.figtext(0.667, 0.07, "(f)", fontsize=14)

	plt.figtext(0.730, 0.07, "(a)", fontsize=14)
	plt.figtext(0.755, 0.07, "(b)", fontsize=14)
	plt.figtext(0.780, 0.07, "(c)", fontsize=14)
	plt.figtext(0.807, 0.07, "(d)", fontsize=14)
	plt.figtext(0.832, 0.07, "(e)", fontsize=14)
	plt.figtext(0.857, 0.07, "(f)", fontsize=14)

	# plt.show() # show the figures

	if not os.path.exists("img/cmp6app/"):
		os.makedirs("img/cmp6app/")
	plt.savefig("img/cmp6app/"+projs[proj_id]+".jpg", format="jpg") # save the figures


def comparison_with_other_RDMetric(proj_id):
	"""
	to get the comprarison experiemnt with RD metric in one project
	"""
	dir_origin = "../experiment/results/Origin_RDs.txt" # RankDirrPr
	dir_nair = "../experiment/results/Nair_RDs.txt" # RD metric by Nair et al.


	nair_results = obtain_results_from_result(dir_nair)
	nair_data = return_all_ranks(nair_results, proj_id)

	origin_results = obtain_results_from_result(dir_origin)
	origin_data = return_all_ranks(origin_results, proj_id)

	datas=[origin_data, nair_data]

	approaches = ["RDTie", "RD"]
	grouped_data = pd.DataFrame(columns=["approaches", "top_k", "rank_difference"], index=None)
	# print(grouped_data)
	tops = [1,3,5,10]
	count = 0
	for app in range(len(approaches)):
		for k in range(len(tops)):
			for i in range(50):
				grouped_data.loc[count] = [approaches[app], tops[k], datas[app][k][i]]
				count +=1

	return grouped_data


def draw_sub_boxplot_rdpr(grouped_datas):

	import matplotlib.pyplot as plt
	import seaborn as sns

	# print(grouped_data)
	plt.figure(figsize=(12,6))
	plt.subplots_adjust(wspace=0.12, bottom=0.15)
	sns.set_style("ticks", {"legend.frameon":False})
	plt.subplot(121)
	# sns.set_style("ticks")
	#boxenplot, violinplot, catplot, stripplot
	sns.boxplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_datas[0], orient="v", width=0.6) # boxplot
	# sns.stripplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_data, orient="v", dodge=True) # stripplot
	plt.yscale("log")
	plt.xlabel("", fontsize=14)
	plt.xticks([0,1,2,3], tuple(["Top-1","Top-3","Top-5","Top-10"]), fontsize=14)
	plt.ylabel("", fontsize=14)
	plt.xlabel("\n(a) Ajstats")
	plt.legend([])

	plt.subplot(122)
	# sns.set_style("ticks")
	#boxenplot, violinplot, catplot, stripplot
	sns.boxplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_datas[1], orient="v", width=0.6) # boxplot
	# sns.stripplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_data, orient="v", dodge=True) # stripplot
	plt.yscale("log")
	plt.xlabel("", fontsize=14)
	plt.xticks([0,1,2,3], tuple(["Top-1","Top-3","Top-5","Top-10"]), fontsize=14)
	plt.ylabel("", fontsize=14)
	plt.xlabel("\n(b) wc-6d-c1-obj1")
	plt.legend(ncol=6, bbox_to_anchor=(1.02, 1.1), fontsize=12)

	# plt.tight_layout()

	plt.show()


def draw_sig_boxplot_rdpr(grouped_data, proj):

	import matplotlib.pyplot as plt
	import seaborn as sns

	# print(grouped_data)
	plt.figure(figsize=(12,6))
	plt.subplots_adjust(wspace=0.12, bottom=0.15)
	sns.set_style("ticks", {"legend.frameon":False})

	# sns.set_style("ticks")
	#boxenplot, violinplot, catplot, stripplot
	sns.boxplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_data, orient="v", width=0.6) # boxplot
	# sns.stripplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_data, orient="v", dodge=True) # stripplot
	plt.yscale("log")
	plt.xlabel("", fontsize=14)
	plt.xticks([0,1,2,3], tuple(["Top-1","Top-3","Top-5","Top-10"]), fontsize=14)
	plt.ylabel("", fontsize=14)
	plt.xlabel("\n" + proj)
	plt.legend(ncol=6, bbox_to_anchor=(1.02, 1.1), fontsize=12)
	# plt.tight_layout()
	# plt.show()

	if not os.path.exists("img/cmp2rd/"):
		os.makedirs("img/cmp2rd/")	
	plt.savefig("img/cmp2rd/"+proj+".jpg", format="jpg") # save the figures


def comparison_with_other_parameter(proj_id):
	"""
	to get the comprarison experiemnt with RD metric in one project
	"""
	dir_cfgrank_9 = "../experiment/results/LTR_RDs.txt" # RankDirrPr 90%
	dir_cfgrank_8 = "../experiment/results/LTR_eight_RDs.txt" # RankDirrPr 80%
	dir_cfgrank_7 = "../experiment/results/LTR_seven_RDs.txt" # RankDirrPr 70%
	
	cfgrank_9_results = obtain_results_from_result(dir_cfgrank_9)
	cfgrank_9_data = return_all_ranks(cfgrank_9_results, proj_id)

	cfgrank_8_results = obtain_results_from_result(dir_cfgrank_8)
	cfgrank_8_data = return_all_ranks(cfgrank_8_results, proj_id)

	cfgrank_7_results = obtain_results_from_result(dir_cfgrank_7)
	cfgrank_7_data = return_all_ranks(cfgrank_7_results, proj_id)

	datas=[cfgrank_9_data, cfgrank_8_data, cfgrank_7_data]

	approaches = ["Removing 90%", "Removing 80%", "Removing 70%"]
	grouped_data = pd.DataFrame(columns=["approaches", "top_k", "rank_difference"], index=None)
	# print(grouped_data)
	tops = [1,3,5,10]
	count = 0
	for app in range(len(approaches)):
		for k in range(len(tops)):
			for i in range(50):
				grouped_data.loc[count] = [approaches[app], tops[k], datas[app][k][i]]
				count +=1

	return grouped_data


def draw_sig_boxpot_ratios(grouped_data, proj):

	import matplotlib.pyplot as plt
	import seaborn as sns

	plt.figure(figsize=(12,6))

	plt.subplots_adjust(wspace=0.12, bottom=0.15)
	sns.set_style("ticks", {"legend.frameon":False})

	#boxenplot, violinplot, catplot, stripplot
	sns.boxplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_data, orient="v", width=0.8) # boxplot
	plt.yscale("log")
	plt.xlabel("", fontsize=14)
	plt.xticks([0,1,2,3], tuple(["Top-1","Top-3","Top-5","Top-10"]), fontsize=14)
	plt.ylabel("RDTie", fontsize=14)
	plt.xlabel("\n"+proj)
	plt.legend(ncol=6, bbox_to_anchor=(1.01, 1.1), fontsize=14)

	# plt.show() # show the figure

	if not os.path.exists("img/cmp3ratio/"):
		os.makedirs("img/cmp3ratio/")
	plt.savefig("img/cmp3ratio/"+proj+".jpg", format="jpg")


def draw_sub_boxpot_ratios(grouped_datas):

	import matplotlib.pyplot as plt
	import seaborn as sns

	plt.figure(figsize=(12,6))

	plt.subplots_adjust(wspace=0.12, bottom=0.15)
	sns.set_style("ticks", {"legend.frameon":False})

	plt.subplot(121)
	#boxenplot, violinplot, catplot, stripplot
	sns.boxplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_datas[0], orient="v", width=0.8) # boxplot
	# sns.stripplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_data, orient="v", dodge=True) # stripplot
	plt.yscale("log")
	plt.xlabel("", fontsize=14)
	plt.xticks([0,1,2,3], tuple(["Top-1","Top-3","Top-5","Top-10"]), fontsize=14)
	plt.ylabel("RDTie", fontsize=14)
	plt.xlabel("\n(a) Ajstats")
	plt.legend([])

	plt.subplot(122)
	#boxenplot, violinplot, catplot, stripplot
	sns.boxplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_datas[1], orient="v", width=0.8) # boxplot
	# sns.stripplot(x="top_k", y="rank_difference", hue="approaches", data=grouped_data, orient="v", dodge=True) # stripplot
	plt.yscale("log")
	plt.xlabel("", fontsize=14)
	plt.xticks([0,1,2,3], tuple(["Top-1","Top-3","Top-5","Top-10"]), fontsize=14)
	# plt.ylabel("RDTie", fontsize=14)
	plt.ylabel("")
	plt.xlabel("\n(b) wc-6d-c1-obj1")
	plt.legend(ncol=6, bbox_to_anchor=(1.01, 1.1), fontsize=14)

	plt.show()


def get_basic_info(projs):
	"""
	to show the basic information in each dataset, including datasets name, options, configs, performance
	"""
	proj_size = 0

	sum_option = 0
	sum_config = 0

	print("TABLE 1: Basic Information in each dataset")
	print("-------------------------------------------------------------------")
	print("| %-17s | %-6s | %-9s | %-23s |"%("Dataset", "# Opts", "# Configs", "Performance"))
	print("-------------------------------------------------------------------")

	for proj in projs:
		proj_dir = "../raw_data/" + proj + ".csv"

		if os.path.exists(proj_dir) == False:  # if some project is not exist
			continue

		proj_size += 1
		pdcontent = pd.read_csv(proj_dir)
		attrs = pdcontent.columns

		options = [attr for attr in attrs if "$<" not in attr]
		perfs = [attr for attr in attrs if "$<" in attr]
		perf = perfs[-1][2:]

		sum_option += len(options)
		sum_config += len(pdcontent)

		# print(proj, "&", len(options), "&", len(pdcontent), "&", perf, "\\\\")
		# print("\hline")
		print("| %-17s | %-6d | %-9d | %-23s |"%(proj, len(options), len(pdcontent), perf))
		
	print("-------------------------------------------------------------------")
	print("| %-17s | %-6d | %-9d | %-23s |"%("TOTAL", sum_option, sum_config, "-"))
	print("| %-17s | %-6.2f | %-9.2f | %-23s |"%("AVERAGE", sum_option/proj_size, sum_config/proj_size, "-"))


def get_tied_top_1(projs):
	"""
	to get how many configurations predicted as top-1 in each project
	"""
	datas = []
	for proj in projs:
		top_1 = calculate_same_top1(proj)
		if top_1 > 0:
			datas.append(top_1)
	# datas=[65.200000000000003, 56.899999999999999, 47.520000000000003, 37.880000000000003, 4.1600000000000001, 4.1600000000000001, 5.5, 3.7799999999999998, 4.7999999999999998, 3.8399999999999999, 15.98, 14.34, 22.920000000000002, 22.0, 46.780000000000001, 53.060000000000002, 26.120000000000001, 24.379999999999999, 22.66, 31.16, 729.63999999999999, 5.7400000000000002, 35.039999999999999, 4.6200000000000001, 14.82, 51.659999999999997, 288.18000000000001, 56.5, 16.600000000000001, 6.3600000000000003, 24.640000000000001, 383.07999999999998, 82.939999999999998, 4.4400000000000004, 36.240000000000002, 20.5]
	draw_same_top1_barplot(projs, datas)


def calculate_same_top1(proj):
	"""
	return the size of configurations that predicted as top 1
	"""
	same_rank = []

	if os.path.exists("../experiment/rank_based/" + proj) == False:
		# print("[Error]: ")
		return -1

	for index in range(50): #for each csv

		path = "../experiment/rank_based/" + proj + "/rank_based" + str(index) + ".csv"
		# print(path)
		pdcontent = pd.read_csv(path)

		predicted_1 = pdcontent.iloc[0]["pre_performance"]
		
		for i in range(len(pdcontent)):
			if pdcontent.iloc[i]["pre_performance"] != predicted_1:
				same_rank.append(i)
				break

	return np.mean(same_rank)


def draw_same_top1_barplot(projs, datas):
	"""
	to draw the tied top-1 size of each project
	"""
	import matplotlib.pyplot as plt

	x = range(len(datas))
	print(datas)

	new_projs = []
	for proj in projs:
		if os.path.exists("../experiment/rank_based/" + proj):
			new_projs.append(proj)

	plt.figure(figsize=(12,8))
	for pos, data in zip(x, datas):
		if data < 10:
			plt.bar(pos, data, log=True, width=0.9, color='#e4fde0', alpha=0.9, edgecolor='black', hatch="/", label="n ≤ 10", linewidth=1.3)
		elif 10 < data < 50:
			plt.bar(pos, data, log=True, width=0.9, color='#466991', alpha=0.9, edgecolor='black', hatch="/", label="10 < n ≤ 50", linewidth=1.3)
		else:
			plt.bar(pos, data, log=True, width=0.9, color='#f35b68', alpha=0.9, edgecolor='black', hatch="/", label="n > 50", linewidth=1.3)

	# plt.legend()
	plt.title("Number of tied configurations in each dataset using the rank-based method")
	plt.ylabel("Number of tied configurations \n (in log scale)", fontsize=14)
	# plt.xticks(x, tuple(new_projs), rotation=90, fontsize=14)
	plt.xticks(x, tuple(projs), rotation=90, fontsize=14)
	plt.ylim(1, 1000)
	plt.xlim(-1, len(x))

	y = [i for i in datas]
	for xx, yy in zip(x, y):
		plt.text(xx-0.3, yy*1.4, str(round(yy,1)), rotation=90)
	# plt.yscale("log")
	plt.tight_layout()	
	# plt.xlim(1, )
	plt.show()

	top_10 = [i for i in datas if i>=10]
	top_100 =[i for i in datas if i>=100]

	print("tied numbers more than 10 :", len(top_10))
	print("tied numbers more than 100:", len(top_100))


def calculate_rdtie_of_project(projs):
	"""
	to calculate the RDTie of each dataset using different approaches,
	:: these results will be saved in txt file under the director "../experiment/results/"
	"""
	new_projs = []
	for proj in projs:
		if os.path.exists("../raw_data/"+proj+".csv"):
			new_projs.append(proj)
	print(new_projs)
	print("-------------------------")

	results_path = "../experiment/results/" # create directory "experiment/results" to save the results of each approach
	if not os.path.exists(results_path):
		os.makedirs(results_path)

	# ## using original predicted results (rank-based approach)
	# Do not remove any configurations
	print("Rank difference using Rank-based")
	for proj in new_projs:
		results = calculate_RDTie(proj)
		line = proj+":"+str(results)+"\n"
		# print(line)
		result_text = "../experiment/results/rank_based_RDTie.txt"
		with open(result_text, "a") as f:
			f.write(line)

	# ## using Classification (Random Forest, method=0)
	# Remove 90% tied configurations
	print("Rank difference using Classification (random forest)")
	for proj in new_projs:
		results = calculate_RDTie_filter(proj, method=0, frac=0.9)
		line = proj+":"+str(results)+"\n"
		# print(line)
		result_text = "../experiment/results/classification_RDTie.txt"
		with open(result_text, "a") as f:
			f.write(line)
    
	# ## using Random
	# Remove 90% tied configurations
	print("Rank difference using Random Deletion")
	for proj in new_projs:
		results = calculate_RDTie_filter(proj, method=1, frac=0.9)
		line = proj+":"+str(results)+"\n"
		# print(line)
		result_text = "../experiment/results/random_rank_RDTie.txt"
		with open(result_text, "a") as f:
			f.write(line)

	# # ## using Direct LTR
	# print("Rank difference using direct LTR")
	# for proj in new_projs:
	# 	results = calculate_RDTie_filter(proj, method=2, frac=0.9)
	# 	line = proj+":"+str(results)+"\n"
	# 	# print(line)
	# 	result_text = "../experiment/results/direct_ltr_RDTie.txt"
	# 	with open(result_text, "a") as f:
	# 		f.write(line)

	# ## using ReConfig (method=1)
	# Remove 90% tied configurations
	print("Rank difference using ReConfig")
	for proj in new_projs:
		results = calculate_RDTie_filter(proj, method=3, frac=0.9)
		line = proj+":"+str(results)+"\n"
		# print(line)
		result_text = "../experiment/results/reconfig_RDTie.txt"
		with open(result_text, "a") as f:
			f.write(line)
    
	# # ## using Outlier detection (One class svm)
	# # Remove the configurations predicted as -1 (outlier)
	# print("Rank difference using Outlier Detection (one class svm)")
	# for proj in new_projs:
	# 	results = calculate_RDTie_outlier(proj)
	# 	line = proj+":"+str(results)+"\n"
	# 	# print(line)
	# 	result_text = "../experiment/results/outlier_detection_RDTie.txt"
	# 	with open(result_text, "a") as f:
	# 		f.write(line)



	################################################################################
	"""
	# ## using RD* (Nair et al.)
	print("Rank difference using RD* (Nair et al.)")
	for proj in new_projs:
		results = calculate_min_R(proj)
		line = proj+":"+str(results)+"\n"
		# print(line)
		result_text = "../experiment/results/Nair_RDs.txt"
		with open(result_text, "a") as f:
			f.write(line)

	# ## using reConfig with different filter ratios
	print("Rank difference using reConfig with the filter ratio of 80%")
	for proj in new_projs:
		results = calculate_rds_proba_filter(proj, method=1, frac=0.8)
		line = proj+":"+str(results)+"\n"
		# print(line)
		result_text = "../experiment/results/LTR_eight_RDs.txt"
		with open(result_text, "a") as f:
			f.write(line)

	print("Rank difference using reConfig with the filter ratio of 70%")
	for proj in new_projs:
		results = calculate_rds_proba_filter(proj, method=1, frac=0.7)
		line = proj+":"+str(results)+"\n"
		# print(line)
		result_text = "../experiment/results/LTR_seven_RDs.txt"
		with open(result_text, "a") as f:
			f.write(line)

	print("Rank difference using ReConfig with the filter ratio of 60%")
	for proj in new_projs:
		results = calculate_rds_proba_filter(proj, method=1, frac=0.6)
		line = proj+":"+str(results)+"\n"
		# print(line)
		result_text = "../experiment/results/LTR_six_RDs.txt"
		with open(result_text, "a") as f:
			f.write(line)
	"""
	#################################################################################


def answering_rq_1(projs):

	print("\nRQ1: Can ReConfig find better configurations than the rank-based approach?")
	print("--------------------")
	# detailed results of RQ1 (Table III)
	compare_result_bet_methods(projs, '../experiment/results/rank_based_RDTie.txt', '../experiment/results/reconfig_RDTie.txt')


def answering_rq_2(projs):

	print("\nRQ2: Can the learning-to-rank method in ReConfig outperform comparative methods in finding configurations?")
	print("--------------------")
	# detailed results of RQ2.
	# paths = ['../experiment/results/LTR_RDs.txt', '../experiment/results/Origin_RDs.txt', '../experiment/results/OCS_RDs.txt', 
	# '../experiment/results/RF_RDs.txt', '../experiment/results/Random_RDs.txt', '../experiment/results/LTR_DIR_RDs.txt']
	# compare_result_six_methods(paths)

	paths = ['../experiment/results/reconfig_RDTie.txt', '../experiment/results/rank_based_RDTie.txt', 
			 '../experiment/results/classification_RDTie.txt', '../experiment/results/random_rank_RDTie.txt']
	compare_results_of_four_methods(paths)

	# results visiualized in RQ2
	# for i in range(len(projs)):
	# 	comparison_with_other_approach(i) # save figures in specific folder


def answering_rq_3(projs):

	print("\nRQ3: How many tied configurations should be filtered out in ReConfig?")
	print("--------------------")
	# detailed results of RQ3
	compare_result_fraction_parameters(['../experiment/results/LTR_RDs.txt', '../experiment/results/LTR_eight_RDs.txt', 
										'../experiment/results/LTR_seven_RDs.txt', '../experiment/results/LTR_six_RDs.txt'])

	# results visiualized in RQ3
	# for i in range(len(projs)):
	# 	grouped_data = comparison_with_other_parameter(i)
	# 	draw_sig_boxpot_ratios(grouped_data, projs[i]) # save figures in specific folder


def answering_rq_4(projs):

	print("\nRQ4: Is RDTie stable for evaluating the tied prediction?")
	print("--------------------")
	# detailed results of RQ4
	compare_result_two_metrics(['../experiment/results/Origin_RDs.txt', '../experiment/results/Nair_RDs.txt'])

	# results visiualized in RQ4
	# for i in range(len(projs)):
	# 	grouped_data = comparison_with_other_RDMetric(i)
	# 	draw_sig_boxplot_rdpr(grouped_data, projs[i])


def show_argv():
	cmd_shortcuts = ["projInfo", "projDistr {$index}", "tiedNums", "calRDTie", "vsRankBased", "vsOthers", "removeRatio", "vsRD"]
	cmd_descripts = ["Showing the basic information (e.g., options and dataset size) in each dataset.", 
					 "Drawing the performance distribution of specific dataset.", 
					 "Drawing the number of tied configuretions in each datasets using the rank-based method.",
					 "Calculating the RDTie of each dataset using different methods.",
					 "RQ-1: Can ReConfig find better configurations than the rank-based approach?",
					 "RQ-2: Can the learning-to-rank method in ReConfig outperform comparative methods in finding configurations?",
					 "RQ-3: How many tied configurations should be filtered out in ReConfig?",
					 "RQ-4: Is RDTie stable for evaluating the tied prediction?"]

	print("\n--------------------------------------")
	print("%-20s | %s"%("Argument", "Description"))
	print("--------------------------------------")
	for cmd, des in zip(cmd_shortcuts, cmd_descripts):
		print("%-20s | %s"%(cmd, des))


if __name__ == "__main__":
	"""
	Note: If you do not want to obtain the intermediate results, just comment the code in STEP-1. STEP-2 will give the detailed results of RQ1-RQ4.
	"""
	# # numeric projects
	# projs1 = ['rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
	# # boolean projects
	# projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']
	# projs = projs1 + projs2

	projs = [f[:-4] for f in os.listdir("../raw_data/")]

	print(projs)
	print("-------------------------")

	cmd_shortcuts = ["projInfo", "projDistr", "tiedNums", "calRDTie", "vsRankBased", "vsOthers", "removeRatio", "vsRD"]

	if len(sys.argv) <= 1:
		print("[Arguments Format Error]: Try to use the following arguments.")
		show_argv()
		sys.exit(0)


	if sys.argv[1] == cmd_shortcuts[0]:
		# projInfo: Calculating the basic information in each dataset
		get_basic_info(projs)
	elif sys.argv[1] == cmd_shortcuts[1]: 
		# projDistr {$index}: Calculating the performance distribution of the specific project {:0-35}
		if len(sys.argv)==3 and sys.argv[2].isdigit:
			get_performance_distribution(projs, int(sys.argv[2]))
		else:
			print("[Arguments Format Error]: You should add a project index after the argument \"projDistr\".")
	elif sys.argv[1] == cmd_shortcuts[2]:
		# tiedNums: Calculating the number of tied configuretions in each datasets using the rank-based method
		get_tied_top_1(projs)
	elif sys.argv[1] == cmd_shortcuts[3]:
		# calRDTie: Calculating the RDTie of each approach using different approaches
		calculate_rdtie_of_project(projs) 
	elif sys.argv[1] == cmd_shortcuts[4]:
		# >>>>>> RQ-1: Can ReConfig find better configurations than the rank-based approach?
		answering_rq_1(projs)
	elif sys.argv[1] == cmd_shortcuts[5]:
		# >>>>>> RQ-2: Can the learning-to-rank method in ReConfig outperform comparative methods in finding configurations?
		answering_rq_2(projs)
	elif sys.argv[1] == cmd_shortcuts[6]:
		# >>>>>> RQ-3: How many tied configurations should be filtered out in ReConfig?
		answering_rq_3(projs)
	elif sys.argv[1] == cmd_shortcuts[7]:
		# >>>>>> RQ-4: Is RDTie stable for evaluating the tied prediction?
		answering_rq_4(projs)
	else:
		print("[Arguments Format Error]: Try to use the following arguments.")
		show_argv()

	# #//\\//\\//\\//\\   STEP-1: Obtaining the intermediate results  //\\//\\//\\//\\

	# FUNCTION-1: Calculating the basic information in each dataset
	# get_basic_info(projs)

	# FUNCTION-2: Calculating the performance distribution of the specific project {:0-35}
	# get_performance_distribution(projs, 25)

	# FUNCTION-3: Calculating the number of tied configuretions in each datasets using the rank-based method
	# get_tied_top_1(projs)

	# FUNCTION-4: Calculating the RDTie of each approach using different approaches
	# calculate_rdtie_of_project(projs) # comment it if you already have these results, otherwise you will cover the results


	# #//\\//\\//\\//\\  STEP-2: Answering the research questions  //\\//\\//\\//\\

	# # >>>>>> RQ-1: Can ReConfig find better configurations than the rank-based approach?
	# answering_rq_1()

	# # >>>>>> RQ-2: Can the learning-to-rank method in ReConfig outperform comparative methods in finding configurations?
	# answering_rq_2(projs)	

	# # >>>>>> RQ-3: How many tied configurations should be filtered out in ReConfig?
	# answering_rq_3(projs)

	# # >>>>>> RQ-4: Is RDTie stable for evaluating the tied prediction?
	# answering_rq_4(projs)
