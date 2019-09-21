#!/usr/bin/python3
# coding=utf-8

"""
@description:
running script of step-1 to step-4.

@author  : Yongfeng
@reviewer: Yongfeng
"""

import os

if __name__ == "__main__":

	print("Happy Starting ~")
	# step-1: prepare the datasets in "raw-data/"

	# step-2: obtain the ranking list using the rank-based approach
	rank_cmd = "python rank_based.py"
	os.system(rank_cmd)

	# step-3: obtain the ranking lists using different approaches
	methods = ["classification.py", "outlier_detection.py", "random_rank.py", "reconfig.py", "direct_ltr.py"]
	for method in methods:
		if not os.path.exists(method):
			print(method, "is not found!")
			continue
		exec_cmd = "python " + method
		os.system(exec_cmd)

	# step-4: calculate the RDTie of ranking lists
	cal_rdtie_cmd = "python experiment.py calRDTie"
	os.system(cal_rdtie_cmd)
	
	print("Happy ending ~")
	
