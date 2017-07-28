# -*- coding: utf-8 -*-
import codecs
import jieba

def cutter():
	in_file_path, out_file_path = "./data/test/clean.txt", "./data/test/cut.txt"
	line_list = codecs.open(in_file_path, "r").read().split("\n")
	file = codecs.open(out_file_path, "w")
	for line in line_list:
		if line == "":
			continue
		file.write(" ".join([ word for word in jieba.cut(line) if word != " " ]).encode("utf-8") + "\n")

if __name__ == "__main__":
	jieba.load_userdict("./dict/dict.txt.big")
	cutter()
