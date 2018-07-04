import pandas as pd
import tensorflow as tf
import string

def preprocess(filepath):
    data = pd.read_csv(filepath, sep='\t+', engine='python',
                       names=['question', 'passage', 'relation'])
    print(data.columns)
    print("Data Types {0}", data.dtypes)

def build_vocab(filepath):
    vocab = {'0': 0, '1': 1}
    index = 1
    with tf.gfile.Open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line.split('\t')
            question = line[0].split(' ').strip(string.punctuation)
            passage = line[1].split(' ').strip(string.punctuation)
            data = question + passage
            for word in data:
                if not word in vocab:
                    index += 1
                    vocab[word] = index
    return vocab, index+1

def sentence_to_word_ids(sentence, word_to_id):
  return [word_to_id[word] for word in sentence if word in word_to_id]

def read_raw_data(filepath):
    word_to_id = build_vocab(filepath)
    with tf.gfile.Open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line.split('\t')
            question = line[0].split(' ').strip(string.punctuation)
            passage = line[1].split(' ').strip(string.punctuation)
            label = line[2]
