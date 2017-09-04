# -*- coding: utf-8 -*-
import os
import codecs

def yielder_1(dir_path):
	return

def yielder_2(dir_path):
	return

def yielder_3(dir_path):
	return

def yielder_4(dir_path):
	return

def yielder_5(dir_path):
	for root_dir_path, dir_path_list, file_path_list in os.walk(dir_path):
		for file_path in file_path_list:
			line_list = codecs.open(root_dir_path + file_path, "r").read().decode("utf-8").split("\n")
			new_line_list = list()
			if u"我的這一班" in file_path.decode("utf-8"): # other 85 files
				for i, line in enumerate(line_list):
					if line == "":
						continue
					if i == 0:
						continue
					if u"：" in line:
						if line.split(u"：")[1] == "":
							continue
						new_line_list.append(line.split(u"：")[1])
						continue
					if "\t" in line:
						if line.split("\t")[-1] == "":
							continue
						new_line_list.append(line.split("\t")[-1])
						continue
					new_line_list.append(line)
				yield file_path, new_line_list
			else:
				for i, line in enumerate(line_list[::-1]):
					if line == "":
						continue
					if u"己所不欲他勿施於人" == line or u"就選擇遺忘" == line or u"就擁有自由夢想的天堂" == line or i == len(line_list) - 1: # index 1 ~ 91, 92 ~ 134, index 135 ~ 250, index 251 ~ 378
						break
					new_line_list.append(line)
				yield file_path, new_line_list[::-1]

def yielder_6(dir_path):
	return

def yielder_7(dir_path):
	return

def yielder_8(dir_path):
	return
	
# type_list = [
# 	"我的這一班"
# ]

type_list = [
	"下課花路米", "人生劇展", "公視藝文大道", "成語賽恩思", 
	"我的這一班", "流言追追追", "聽聽看", "誰來晚餐"
]

# type_dict = { 
# 	"我的這一班": yielder_5
# }

type_dict = {
	"下課花路米": yielder_1,
	"人生劇展": yielder_2,
	"公視藝文大道": yielder_3, 
	"成語賽恩思": yielder_4, 
	"我的這一班": yielder_5, 
	"流言追追追": yielder_6, 
	"聽聽看": yielder_7, 
	"誰來晚餐": yielder_8
}
