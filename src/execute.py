#!/usr/bin/python3
# coding=utf-8

import os

if __name__ == "__main__":
	methods = ["classification.py", "outlier_detection.py", "random_rank.py", "reconfig.py", "direct_ltr.py"]
	for method in methods:
		if not os.path.exists(method):
			print(method, "is not found!")
			continue
		exec_cmd = "python " + method
		os.system(exec_cmd)