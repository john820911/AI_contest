# -*- coding: utf-8 -*-
import codecs
import random

def writer(file_path, line_list):
	file = codecs.open(file_path, "w")
	for line in line_list:
		file.write(line.encode("utf-8") + "\n")

def maker():
	train_line_list = codecs.open("./data/train/all.txt", "r").read().decode("utf-8").split("\n")[:-1]
	test_line_list = codecs.open("./data/test/cut.txt", "r").read().decode("utf-8").split("\n")[:-1]
	train_enc_line_list, train_dec_line_list = train_line_list[:-1], train_line_list[1:]
	new_line_list = [ (enc_line, dec_line) for enc_line, dec_line in zip(train_enc_line_list, train_dec_line_list) if(len(enc_line.split(" ")) != 1 and len(enc_line.split(" ")) != 2) ]
	train_enc_line_list = [ pair[0] for pair in new_line_list ]
	train_dec_line_list = [ pair[1] for pair in new_line_list ]
	writer("./data/train/train.enc", [ train_enc_line_list[i] for i in range(len(train_enc_line_list)) if i % 10 != 9 ])	
	writer("./data/train/train.dec", [ train_dec_line_list[i] for i in range(len(train_dec_line_list)) if i % 10 != 9 ])
	writer("./data/test/dev.enc", [ train_enc_line_list[i] for i in range(len(train_enc_line_list)) if i % 10 == 9 ])
	writer("./data/test/dev.dec", [ train_dec_line_list[i] for i in range(len(train_dec_line_list)) if i % 10 == 9 ])
	writer("./data/test/test.enc", [ test_line_list[i] for i in range(len(test_line_list)) if i % 7 == 0 ])
	writer("./data/test/test_0.dec", [ test_line_list[i] for i in range(len(test_line_list)) if i % 7 == 1 ])
	writer("./data/test/test_1.dec", [ test_line_list[i] for i in range(len(test_line_list)) if i % 7 == 2 ])
	writer("./data/test/test_2.dec", [ test_line_list[i] for i in range(len(test_line_list)) if i % 7 == 3 ])
	writer("./data/test/test_3.dec", [ test_line_list[i] for i in range(len(test_line_list)) if i % 7 == 4 ])
	writer("./data/test/test_4.dec", [ test_line_list[i] for i in range(len(test_line_list)) if i % 7 == 5 ])
	writer("./data/test/test_5.dec", [ test_line_list[i] for i in range(len(test_line_list)) if i % 7 == 6 ])

if __name__ == "__main__":
	maker()
