# -*- coding: utf-8 -*-
import codecs

file_1 = open("./result/result.txt")
file_2 = open("./result/choice.txt")

lines_1 = [ line.strip("\n") for line in file_1.readlines() ]
lines_2 = [ line.strip("\n") for line in file_2.readlines() ]

not_equal_list = []
count = 0
for i, (line_1, line_2) in enumerate(zip(lines_1, lines_2)):
	if int(line_1) == int(line_2):
		count += 1
	else:
		not_equal_list.append(i)

print count, len(lines_1)
