# -*- coding: utf-8 -*-
import codecs
from collections import Counter

def writer(file_path, word_list):
	file = codecs.open(file_path, "w")
	file.write("_PAD\n_GO\n_EOS\n_UNK\n")
	for word, count in word_list:
		file.write(word.encode("utf-8") + "\n")

a = [(5, 4), (7, 5), (10, 7), (30, 30)]

def maker():
	train_enc_line_list = codecs.open("./data/train/train.enc", "r").read().decode("utf-8").split("\n")[:-1]
	train_dec_line_list = codecs.open("./data/train/train.dec", "r").read().decode("utf-8").split("\n")[:-1]
	dev_enc_line_list = codecs.open("./data/test/dev.enc", "r").read().decode("utf-8").split("\n")[:-1]
	dev_dec_line_list = codecs.open("./data/test/dev.dec", "r").read().decode("utf-8").split("\n")[:-1]
	train_line_list = train_enc_line_list + train_dec_line_list + dev_enc_line_list + dev_dec_line_list
	test_line_list = codecs.open("./data/test/cut.txt", "r").read().decode("utf-8").split("\n")[:-1]
	all_word_list = list()
	train_word_list = list()
	test_word_list = list()
	all_len_list = list()
	train_len_list = list()
	test_len_list = list()
	for line in train_line_list:
		all_word_list.extend(line.split(" "))
		train_word_list.extend(line.split(" "))
		all_len_list.append(len(line.split(" ")))
		train_len_list.append(len(line.split(" ")))
	for line in test_line_list:
		all_word_list.extend(line.split(" "))
		test_word_list.extend(line.split(" "))
		all_len_list.append(len(line.split(" ")))
		test_len_list.append(len(line.split(" ")))
	all_new_word_list = Counter(all_word_list).most_common(24996)
	train_new_word_list = Counter(train_word_list).most_common()
	test_new_word_list = Counter(test_word_list).most_common()
	all_new_len_list = sorted(Counter(all_len_list).most_common(), key=lambda x: x[0])
	train_new_len_list = sorted(Counter(train_len_list).most_common(), key=lambda x: x[0])
	test_new_len_list = sorted(Counter(test_len_list).most_common(), key=lambda x: x[0])
	writer("./dict/vocab25000.enc", all_new_word_list)
	writer("./dict/vocab25000.dec", all_new_word_list)
	print "all_len_list:", all_new_len_list
	print "train_new_len_list:", train_new_len_list
	print "test_new_len_list:", test_new_len_list
	try_list = []
	for enc_line, dec_line in zip(train_enc_line_list, train_dec_line_list):
		for i, (b_1, b_2) in enumerate(a):
			if(len(enc_line.split(" ")) <= b_1 and len(dec_line.split(" ")) <= b_2):
				try_list.append(i)
				break
	print "bucket:", Counter(try_list)


if __name__ == "__main__":
	maker()	
