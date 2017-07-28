# -*- coding: utf-8 -*-
import jieba
from yield_data import *

def cutter(data_type):
	in_dir_path, out_dir_path = "./data/train/clean/" + data_type + "/", "./data/train/cut/" + data_type + "/"	
	if not os.path.exists(out_dir_path):
		os.mkdir(out_dir_path)
	for root_dir_path, dir_path_list, file_path_list in os.walk(in_dir_path):
		for file_path in file_path_list:
			line_list = codecs.open(root_dir_path + file_path, "r").read().decode("utf-8").split("\n")
			file = codecs.open(out_dir_path + file_path, "w")
			for line in line_list:
				if line == "":
					continue
				file.write(" ".join([ word for word in jieba.cut(line) if word != " " and word != "," and word != "..." ]).encode("utf-8") + "\n")	

if __name__ == "__main__":
	jieba.load_userdict("./dict/dict.txt.big")
	for data_type in type_list:
		cutter(data_type)
