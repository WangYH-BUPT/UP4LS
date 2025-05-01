import numpy as np
import collections
from tqdm import tqdm
import torch
import random
from sklearn.model_selection import train_test_split

PAD, CLS = '[PAD]', '[CLS]'


class DataHelper(object):
	def __init__(self, raw, word_drop=5, ratio=0.8, use_label=False, use_length=False):

		
		assert (use_label and (len(raw) == 2)) or ((not use_label) and (len(raw) == 1))
		self._word_drop = word_drop
		self.use_label = use_label
		self.use_length = use_length
		self.train = None
		self.train_org = None
		self.train_num = 0
		self.test = None
		self.test_org = None
		self.test_num = 0
		if self.use_label:
			self.label_train = None
			self.label_test = None
		if self.use_length:
			self.train_length = None
			self.test_length = None
			self.max_sentence_length = 0
			self.min_sentence_length = 0
		self.vocab_size = 0
		self.vocab_size_raw = 0
		self.sentence_num = 0
		self.word_num = 0
		self.w2i = {}
		self.i2w = {}
		sentences = []
		for _ in raw:
			sentences += _
		self._build_vocabulary(sentences)
		corpus_length = None
		label = None
		if self.use_length:
			corpus, corpus_length = self._build_corpus(sentences)
		else:
			corpus = self._build_corpus(sentences)
		if self.use_label:
			label = self._build_label(raw)
		# self._split(corpus, ratio, corpus_length=corpus_length, label=label)
		self._split(corpus, ratio, corpus_length=corpus_length, label=label, sentences=sentences)

	
	def _build_label(self, raw):
		label = [0]*len(raw[0]) + [1]*len(raw[1])
		return np.array(label)

	
	def _build_vocabulary(self, sentences):
		self.sentence_num = len(sentences)
		words = []
		for sentence in sentences:
			words += sentence.split(' ')
		self.word_num = len(words)
		word_distribution = sorted(collections.Counter(words).items(), key=lambda x: x[1], reverse=True)
		self.vocab_size_raw = len(word_distribution)
		self.w2i['_PAD'] = 0
		self.w2i['_UNK'] = 1
		self.w2i['_BOS'] = 2
		self.w2i['_EOS'] = 3
		self.i2w[0] = '_PAD'
		self.i2w[1] = '_UNK'
		self.i2w[2] = '_BOS'
		self.i2w[3] = '_EOS'

		for (word, value) in word_distribution:
			if value > self._word_drop:
				self.w2i[word] = len(self.w2i)
				self.i2w[len(self.i2w)] = word
		self.vocab_size = len(self.i2w)

	
	def _build_corpus(self, sentences):
		def _transfer(word):
			try:
				return self.w2i[word]
			except:
				return self.w2i['_UNK']
		corpus = [[self.w2i["_BOS"]] + list(map(_transfer, sentence.split(' '))) + [self.w2i["_EOS"]] for sentence in sentences]
		if self.use_length:
			corpus_length = np.array([len(i) for i in corpus])
			self.max_sentence_length = corpus_length.max()
			self.min_sentence_length = corpus_length.min()
			return np.array(corpus), np.array(corpus_length)
		else:
			return np.array(corpus)

	
	def _split(self, corpus, ratio, corpus_length=None, label=None, sentences=None):
		indices = list(range(self.sentence_num))
		np.random.shuffle(indices)
		self.train = corpus[indices[:int(self.sentence_num * ratio)]]
		self.train_num = len(self.train)
		self.test = corpus[indices[int(self.sentence_num * ratio):]]
		self.test_num = len(self.test)
		if sentences is not None:
			sentences = np.array(sentences)
			self.train_org = sentences[indices[:int(self.sentence_num * ratio)]]
			self.test_org = sentences[indices[int(self.sentence_num * ratio):]]
		if self.use_length:
			self.train_length = corpus_length[indices[:int(self.sentence_num * ratio)]]
			self.test_length = corpus_length[indices[int(self.sentence_num * ratio):]]
		if self.use_label:
			self.label_train = label[indices[:int(self.sentence_num * ratio)]]
			self.label_test = label[indices[int(self.sentence_num*ratio):]]

	def _padding(self, batch_data):
		max_length = max([len(i) for i in batch_data])
		for i in range(len(batch_data)):
			batch_data[i] += [self.w2i["_PAD"]] * (max_length - len(batch_data[i]))
		return np.array(list(batch_data))

	def train_generator(self, batch_size, shuffle=True):
		indices = list(range(self.train_num))
		if shuffle:
			np.random.shuffle(indices)
		while True:
			batch_indices = indices[0: batch_size]                
			indices = indices[batch_size:]                 
			if len(batch_indices) == 0:
				return True
			batch_data = self.train[batch_indices]
			batch_data = self._padding(batch_data)
			result = [batch_data]
			if self.use_length:
				batch_length = self.train_length[batch_indices]
				result.append(batch_length)
			if self.use_label:
				batch_label = self.label_train[batch_indices]
				result.append(batch_label)
			yield tuple(result)

	def test_generator(self, batch_size, shuffle=True):
		indices = list(range(self.test_num))
		if shuffle:
			np.random.shuffle(indices)
		while True:
			batch_indices = indices[0: batch_size]         
			indices = indices[batch_size:]        
			if len(batch_indices) == 0:
				return True
			batch_data = self.test[batch_indices]
			batch_data = self._padding(batch_data)
			result = [batch_data]
			if self.use_length:
				batch_length = self.test_length[batch_indices]
				result.append(batch_length)
			if self.use_label:
				batch_label = self.label_test[batch_indices]
				result.append(batch_label)
			yield tuple(result)
	pass


# def build_dataset(args):
# 	def load_dataset(paths, pad_size=100):
# 		contents = []
# 		for path in paths:
# 			with open(path, 'r', errors='ignore') as f:
# 				for line in tqdm(f):
# 					lines = []
# 					lin = line.strip()
# 					if 'cover' in path:
# 						label = 0
# 					else:
# 						label = 1
# 					lines.append(line)
# 					token = args.tokenizer.tokenize(lin)
# 					token = [CLS] + token
# 					seq_len = len(token)
# 					token_ids = args.tokenizer.convert_tokens_to_ids(token)
#
# 					if pad_size:
# 						if len(token) < pad_size:
# 							mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
# 							token_ids += ([0] * (pad_size - len(token)))
# 						else:
# 							mask = [1] * pad_size
# 							token_ids = token_ids[:pad_size]
# 							seq_len = pad_size
#
# 					token_ids_none_pad = [i for i in token_ids if i != 0]
# 					text2idx = dict(zip(token, token_ids_none_pad))
# 					idx2text = dict(zip(token_ids_none_pad, token))
# 					contents.append((token_ids, label, seq_len, mask, token, text2idx, idx2text, lines))
# 		random.shuffle(contents)
# 		return contents
#
# 	train_data = load_dataset([args.train_cover_dir, args.train_stego_dir])
# 	test_data = load_dataset([args.test_cover_dir, args.test_stego_dir])
# 	return train_data, test_data


def build_dataset(args):
	def load_dataset(path, label, pad_size=100):
		contents = []
		with open(path, 'r', errors='ignore') as f:
			for line in tqdm(f):
				lin = line.strip()
				token = args.tokenizer.tokenize(lin)
				token = [CLS] + token
				seq_len = len(token)
				token_ids = args.tokenizer.convert_tokens_to_ids(token)

				if pad_size:
					if len(token) < pad_size:
						mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
						token_ids += ([0] * (pad_size - len(token)))
					else:
						mask = [1] * pad_size
						token_ids = token_ids[:pad_size]
						seq_len = pad_size

				token_ids_none_pad = [i for i in token_ids if i != 0]
				text2idx = dict(zip(token, token_ids_none_pad))
				idx2text = dict(zip(token_ids_none_pad, token))
				# contents.append((token_ids, label, seq_len, mask, token, text2idx, idx2text, lines))
				contents.append((token_ids, label, seq_len, mask, token, text2idx, idx2text, [line]))
		# random.shuffle(contents)
		return contents

	cover_data = load_dataset(args.neg_filename, label=0)
	stego_data = load_dataset(args.pos_filename, label=1)

	combined_data = cover_data + stego_data
	if args.shuffle:
		random.shuffle(combined_data)

	train_data, test_data = train_test_split(combined_data, test_size=0.2)  # 例如，20%作为测试集
	return train_data, test_data

	# train_data = load_dataset([args.train_cover_dir, args.train_stego_dir])
	# test_data = load_dataset([args.test_cover_dir, args.test_stego_dir])
	# return train_data, test_data


class DatasetIterater(object):
	def __init__(self, batches, args):
		self.batch_size = args.batch_size
		self.batches = batches
		self.n_batches = len(batches) // self.batch_size
		self.residue = False
		if len(batches) % self.n_batches != 0:
			self.residue = True
		self.index = 0
		self.device = args.device

	def _to_tensor(self, datas):
		x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
		y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

		seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
		mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
		return (x, seq_len, mask), y

	def __next__(self):
		if self.residue and self.index == self.n_batches:
			batches = self.batches[self.index * self.batch_size:len(self.batches)]
			self.index += 1
			batches = self._to_tensor(batches)
			return batches

		elif self.index > self.n_batches:
			self.index = 0
			raise StopIteration

		else:
			batches = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
			self.index += 1
			batches = self._to_tensor(batches)
			return batches

	def __iter__(self):
		return self

	def __len__(self):
		if self.residue:
			return self.n_batches + 1
		else:
			return self.n_batches


def build_iterator(dataset, args):
	iters = DatasetIterater(dataset, args)
	return iters
