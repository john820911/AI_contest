# -*- coding: utf-8 -*-
from yield_data import *

def cleaner(data_type, yielder_type):
	in_dir_path, out_dir_path = "./data/train/raw/" + data_type + "/", "./data/train/clean/" + data_type + "/"
	yielder = yielder_type(in_dir_path)
	if not os.path.exists(out_dir_path):
		os.mkdir(out_dir_path)
	for file_path, new_line_list in yielder:
		file = codecs.open(out_dir_path + file_path, "w")
		for line in new_line_list:
			file.write(line.encode("utf-8") + "\n")	

if __name__ == "__main__":
	for data_type, yielder_type in type_dict.iteritems():
		cleaner(data_type, yielder_type)
