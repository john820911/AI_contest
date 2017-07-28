# -*- coding: utf-8 -*-
import codecs

def cleaner():
	in_file_path, qa_file_path, choice_file_path = "./data/test/raw.txt", "./data/test/clean.txt", "./data/test/choice.txt"
	line_list = codecs.open(in_file_path, "r").read().split("\n")
	question_list, answer_list, choice_list = list(), list(), list()
	for line in line_list:
		if line == "":
			continue
		temp_list = list()
		for temp in line.split(",")[1].split("\t"):
			temp_list.append(temp.decode("utf-8"))
		question_list.append(" ".join(temp_list))
		answer_list.extend(line.split(",")[2].split("\t"))
		choice_list.append(line.split(",")[3])
	qa_file, choice_file = codecs.open(qa_file_path, "w"), codecs.open(choice_file_path, "w")
	for i in range(len(question_list)):
		qa_file.write(question_list[i].encode("utf-8") + "\n")
		qa_file.write(answer_list[6 * i + 0] + "\n")
		qa_file.write(answer_list[6 * i + 1] + "\n")
		qa_file.write(answer_list[6 * i + 2] + "\n")
		qa_file.write(answer_list[6 * i + 3] + "\n")
		qa_file.write(answer_list[6 * i + 4] + "\n")
		qa_file.write(answer_list[6 * i + 5] + "\n")
		choice_file.write(choice_list[i] + "\n")

if __name__ == "__main__":
	cleaner()
