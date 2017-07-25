#-*- coding: utf-8 -*-
import os
import operator
import collections
import pandas as pd
import numpy as np
import pickle
from parameter import  caption_size
from bleu import *

def getInfo(info_path):
    return pd.read_csv(info_path, sep=",")

def loadDic(word_dic_path, id_dic_path, init_bias_dic_path, embed_dic_path):
	word_id = pickle.load(open(word_dic_path))
	id_word = pickle.load(open(id_dic_path))
	init_bias_vector = pickle.load(open(init_bias_dic_path))
	embd = pickle.load(open(embed_dic_path))
	return word_id, id_word, init_bias_vector, embd

def arr2str(words):
	string = ''
	for word in words:
		if word == "<eos>":
			break
		string += word + ' '	
	return string

def bleu_score(labels, caption):
	score = []
	for label in labels:
		score.append(BLEU_2(caption,label[:-1]))
	score_mean = np.mean(score)
	print (score_mean)
	return score_mean
	
def inv_sigmoid(x, k):
	return k / (k + np.exp(float(x) / k))
