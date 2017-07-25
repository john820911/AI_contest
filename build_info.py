import os
import sys
import csv
import json
import nltk
import codecs

#argv
label_json_path = "./data/" + sys.argv[1] + "/label.json"
label_dir = "./data/" + sys.argv[1] + "/label/"
info_csv_path = "./data/" + sys.argv[1] + "/info.csv"
label_taken = sys.argv[2]
length_upper_bound = sys.argv[3]

#remove
for file in os.listdir(label_dir):
    os.remove(label_dir + file)

#open
output_label = json.load(codecs.open(label_json_path, "r", "utf-8"))

#preprocess
id_caption_list_of_list = []
id_caption_list_of_list.append(["feat_path", "label_path"])
for label in output_label:
    temp_sentences = [ nltk.word_tokenize(sentence) for sentence in label["caption"] ]
    punct_word_list = [ ",", ".", "!", "?", " " ]
    temp_regex_sentences = [ [  word for word in sentence if word not in punct_word_list ] for sentence in temp_sentences ]
    temp_strip_sentences = [ sentence for sentence in temp_regex_sentences if(len(sentence) <= int(length_upper_bound)) ]
    temp_sort_sentences = sorted(temp_strip_sentences, key=len)
    temp_join_sentences = [ " ".join(sentence) for sentence in temp_sort_sentences ]
    if(label_taken == "all"):
    	for index, sentence in enumerate(temp_join_sentences):
    		id_caption_list_of_list.append([ label["id"] + ".npy", label["id"] + "." + str(index) + ".json" ])
    		with open(label_dir + label["id"] + "." + str(index) + ".json", "w") as label_file:
    			json.dump(sentence, label_file, indent=4, sort_keys=True)
    elif(label_taken == "one"):
    	id_caption_list_of_list.append([ label["id"] + ".npy", label["id"] + ".json" ])
    	with open(label_dir + label["id"] + ".json", "w") as label_file:
    		json.dump(temp_join_sentences[-1], label_file, indent=4, sort_keys=True)
    else:
    	id_caption_list_of_list.append([ label["id"] + ".npy", label["id"] + ".json" ])
    	with open(label_dir + label["id"] + ".json", "w") as label_file:
    		json.dump(temp_join_sentences[-1], label_file, indent=4, sort_keys=True)		

#convert to utf-8
id_caption_list_of_list = [ [ unicode(w).encode("utf-8") for w in id_caption_list ] for id_caption_list in id_caption_list_of_list ]

#output
with open(info_csv_path, "w") as info_file:
    writer = csv.writer(info_file)
    writer.writerows(id_caption_list_of_list)
