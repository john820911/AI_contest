# -*- coding: utf-8 -*-
from yield_data import *

def gatherer(data_type):
	line_list = list()
	for root_dir_path, dir_path_list, file_path_list in os.walk("./data/train/cut/" + data_type + "/"):
		for file_path in file_path_list:
			line_list.extend(codecs.open(root_dir_path + file_path, "r").read().decode("utf-8").split("\n")[:-1])
	return line_list

def writer(all_line_list):
	file = codecs.open("./data/train/all.txt", "w")
	for line in all_line_list:
		file.write(line.encode("utf-8") + "\n")

if __name__ == "__main__":
	all_line_list = list()
	for data_type in type_list:
		all_line_list.extend(gatherer(data_type))
	writer(all_line_list)
