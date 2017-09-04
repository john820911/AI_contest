# -*- coding: utf-8 -*-
import codecs
import jieba
import jieba.analyse
from collections import Counter

def cutter():
	in_file_path, out_file_path = "./data/test/clean.txt", "./data/test/cut.txt"
	line_list = codecs.open(in_file_path, "r").read().decode("utf-8").split("\n")
	file = codecs.open(out_file_path, "w")
	len_list = []
	for i, line in enumerate(line_list):
		if line == "":
			continue
		word_list = [ word for word in jieba.cut(line) if word != " " and word != "," and word != "..." ]
		if len(word_list) >= 30:
			if i % 7 == 0:
				sentence_list = line.split(" ")
				while len([ word for word in jieba.cut(" ".join(sentence_list)) if word != " " and word != "," and word != "..." ]) >= 30:
					sentence_list.pop(0)
				word_list = [ word for word in jieba.cut(" ".join(sentence_list)) if word != " " and word != "," and word != "..." ]
			else:
				sentence_list = line.split(" ")
				while len([ word for word in jieba.cut(" ".join(sentence_list)) if word != " " and word != "," and word != "..." ]) >= 30:
					sentence_list.pop(-1)				
				word_list = [ word for word in jieba.cut(" ".join(sentence_list)) if word != " " and word != "," and word != "..." ]
		file.write(" ".join(word_list).encode("utf-8") + "\n")
		len_list.append(len(word_list))			
	print sorted(Counter(len_list).most_common(), key=lambda x: x[0])			

if __name__ == "__main__":
	jieba.load_userdict("./dict/vocab.txt")
	jieba.analyse.set_idf_path("./dict/idf.txt.big")
	cutter()
