import Reader
import collections
import sys
import re
import argparse
from langconv import *
import numpy as np

def build_dictionary(sentences, label, vocabulary_size):
	words = [word for sublist in sentences for word in sublist]
	count = [('unknown', -1)]
	word_dict = {}
	count.extend(collections.Counter(words).most_common(vocabulary_size))
	#print(count[:10])
	#print(count[-50:])
	for word, word_count in count:
		word_dict[word] = len(word_dict)
	word_dict_rev = dict(zip(word_dict.values(), word_dict.keys()))
	return word_dict, word_dict_rev
	
def text2num(texts, word_dict):
	data = []
	for sentence in texts:
		sentence_data = []
		for word in sentence:
			if word in word_dict:
				word_ix = word_dict[word]
			else:
				word_ix = 0
			sentence_data.append(word_ix)
		data.append(sentence_data)
	return data
def append_padding(texts, max_sequence_len, padding = 0):
	sentences = []
	#sentence = []
	for text in texts:
		if len(text) < max_sequence_len:
			for id in range(max_sequence_len - len(text)):
				text.append(0)
		if len(text) > max_sequence_len:
			text = [word for word in text[:max_sequence_len]]
		sentences.append(text)
	return sentences	

def generate_num_label(labels):
	y = []
	for label in labels:
		if label == 'n':
			y.append(0)
		elif label == 'p':
			y.append(1)
		else:
			y.append(2)
	return y
	
	
def mergePostResponseLabel(post, response, label):
    post_response_label = []
    for elem in zip(post, response, label):
        post_response_label.append(elem)
    return post_response_label