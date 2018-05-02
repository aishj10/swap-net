from nltk.corpus import stopwords
import re
import gensim
from tensorflow.python.platform import gfile
from gensim.summarization import keywords
import sys
#sys.path.insert(0,'RAKE-tutorial')
#import rake
import operator
#stop = set(stopwords.words('english'))
import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2
import math
import pickle
import numpy as np

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_NUL = b"_NUL"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK,_NUL]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
NUL_ID=4

_DIGIT_RE = re.compile(br"\s\d+\s|\s\d+$")
SYB_RE=re.compile(b"([.,!?\"':;)(])|--")
model_path='dataset/finished_files/embed_model.bin'



SENTENCE_START={}
SENTENCE_END={}
SENTENCE_START['article'] = '<a>'
SENTENCE_END['article'] = '</a>'
SENTENCE_START['word'] = '<w>'
SENTENCE_END['word'] = '</w>'
SENTENCE_END['abstract'] = '<eos>'



def basic_tokenizer(sentence):
	
	words = []
	for space_separated_fragment in sentence.strip().split():
		words.extend(_WORD_SPLIT.split(space_separated_fragment))
	return [w for w in words if w]


def initialize_vocabulary(vocabulary_path):

	if gfile.Exists(vocabulary_path):
		rev_vocab = []
		with gfile.GFile(vocabulary_path, mode="rb") as f:
			rev_vocab.extend(f.readlines())
		rev_vocab = [line.strip() for line in rev_vocab]
		vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
		return vocab, rev_vocab
	else:
		raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def initialize_keywords(key_path):
	
	if gfile.Exists(key_path):
		key = []
		with gfile.GFile(key_path, mode="rb") as f:
			key.extend(f.readlines())
		key = [line.strip() for line in key]

		return key
	else:
		raise ValueError("keyword file %s not found.", key_path)


def sentence_to_token_ids(sentence, vocabulary,
													tokenizer=None, normalize_digits=True):


	if tokenizer:
		words = tokenizer(sentence)
	else:
		
		words = sentence

	if not normalize_digits:
		return [vocabulary.get(w, UNK_ID) for w in words]

	return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
											tokenizer=None, normalize_digits=True):

	if not gfile.Exists(target_path):
		print("Tokenizing data in %s" % data_path)
		vocab, _ = initialize_vocabulary(vocabulary_path)
		with gfile.GFile(data_path, mode="rb") as data_file:
			with gfile.GFile(target_path, mode="w") as tokens_file:
				counter = 0
				for line in data_file:
					counter += 1
					if counter % 100000 == 0:
						print("  tokenizing line %d" % counter)
					token_ids = sentence_to_token_ids(line, vocab, tokenizer,
																						normalize_digits)
					tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def get_embedding(vocab,embedding_size):
	
	
	sqrt3 = math.sqrt(3)
	#vocab, _ = initialize_vocabulary(vocabulary_path)
	size_vocab=len(vocab)
	embed=np.zeros((size_vocab,embedding_size))
	model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=True)  
	for (k, v) in vocab.iteritems() :
		if k in model.vocab:

			w=_DIGIT_RE.sub(b"0", k)
			embed[v]=model[w]
		else:
			embed[v]=np.random.uniform(-sqrt3,sqrt3,embedding_size)
	return embed




def article2sents(abstract,mode='article'):

	cur = 0
	sents = []
	while True:
		try:
			start_p = abstract.index(SENTENCE_START[mode], cur)
			end_p = abstract.index(SENTENCE_END[mode], start_p + 1)
			cur = end_p + len(SENTENCE_END[mode])
			sents.append(abstract[start_p+len(SENTENCE_START[mode]):end_p])
		except ValueError as e: # no more sentences
			#print("sents",sents)
			return sents


def text2labels(abstract,mode='word'):

	cur = 0
	
	labels=[]
	while True:
		try:
			sents = []
			start_p = abstract.index(SENTENCE_START[mode], cur)
			end_p = abstract.index(SENTENCE_END[mode], start_p + 1)
			cur = end_p + len(SENTENCE_END[mode])
			sents=(abstract[start_p+len(SENTENCE_START[mode]):end_p])
			#print("sents1" ,sents)
			sents=[int(s) for s in sents.strip().split(' ')]
			#print("sents2" ,sents)
			labels.append(sents)
		except ValueError as e: # no more sentences
			return labels

def abstract2sents(abstract,mode='abstract'):

        cur = 0
        sents = []
        while True:
                try:
                        start_p =cur
                        end_p = abstract.index(SENTENCE_END[mode], start_p + 1)
                        cur = end_p + len(SENTENCE_END[mode])
                        sents.append(abstract[start_p:end_p])
                except ValueError as e: # no more sentences
                        #print("sents",sents)
                        return sents.append(abstract[cur:])

def example_generator(data_path, single_pass):

	
	while True:
		filelist = glob.glob(data_path) # get the list of datafiles
		assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
		if single_pass:
			filelist = sorted(filelist)
		else:
			random.shuffle(filelist)
		for f in filelist:
			reader = open(f, 'rb')
			while True:
				len_bytes = reader.read(8)
				if not len_bytes: break # finished reading this file
				str_len = struct.unpack('q', len_bytes)[0]
				example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
				#print(example_str)
				yield example_pb2.Example.FromString(example_str)

		if single_pass:
			print "example_generator completed reading all datafiles. No more data."
			break

