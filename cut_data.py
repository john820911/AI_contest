# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
from collections import Counter
from yield_data import *

def cutter(data_type):
	in_dir_path, out_dir_path = "./data/train/clean/" + data_type + "/", "./data/train/cut/" + data_type + "/"	
	if not os.path.exists(out_dir_path):
		os.mkdir(out_dir_path)
	len_list = []	
	for root_dir_path, dir_path_list, file_path_list in os.walk(in_dir_path):
		for file_path in file_path_list:
			line_list = codecs.open(root_dir_path + file_path, "r").read().decode("utf-8").split("\n")
			file = codecs.open(out_dir_path + file_path, "w")
			for line in line_list:
				if line == "":
					continue
				word_list = [ word for word in jieba.cut(line) if word != " " and word != "," and word != "..." ]
				if len(word_list) > 0 and len(word_list) < 30:
					file.write(" ".join(word_list).encode("utf-8") + "\n")
					len_list.append(len(word_list))
	print sorted(Counter(len_list).most_common(), key=lambda x: x[0])

if __name__ == "__main__":
	jieba.load_userdict("./dict/vocab.txt")
	jieba.analyse.set_idf_path("./dict/idf.txt.big")
	for data_type in type_list:
		cutter(data_type)
