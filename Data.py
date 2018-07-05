import pandas as pd
import tensorflow as tf
import string
import re
import numpy as np
import pickle as pkl
from os import path

class Data(object):
    def __init__(self, filepath, dssm_file_path, data_type, batch_size = 0, vocab = None):
        self.filepath = filepath
        self.dssm_file_path = dssm_file_path
        self.max_question_len = 180  # 176
        self.max_passage_len = 1000  # 989
        self.questions = []
        self.questions_len = []
        self.passages = []
        self.passages_len = []
        self.labels = []
        self.dssm_vectors = []
        self.batch_id = 0
        self.data_type = data_type
        if self.data_type == 'train':
            self.vocab = self.build_vocab()
            self.batch_size = batch_size
        else:
            self.vocab = vocab
            self.batch_size = 0
        self.read_raw_data()
        self.read_raw_data_with_dssm()
        print("Data loaded")

    def preprocess(self):
        data = pd.read_csv(self.filepath, sep='\t+', engine='python',
                        names=['question', 'passage', 'relation'])
        print(data.columns)
        print("Data Types {0}", data.dtypes)

    def build_vocab(self):
        print("Building Vocab")
        vocab = {'0': 0, '1': 1, 'UNK': 2, 'NUM': 3}
        index = 1
        if path.exists(self.data_type+'vocab.pkl'):
            return pkl.load(open(self.data_type+'vocab.pkl', 'rb'))
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                count+=1
                line = line.split('\t')
                question = re.sub('([.,!?()-])', r'  \1  ', line[0])
                question = question.split(' ')
                question = [word for word in question if word != '' ]
                passage = re.sub('([.,!?()-])', r'  \1  ', line[1])
                passage = passage.split(' ')
                passage = [word for word in passage if word != '' ]
                data = question + passage
                '''if count<100:
                    print(data)'''
                for word in data:
                    if word.isdigit():
                        continue
                    if not word in vocab:
                        index += 1
                        vocab[word.lower().replace(string.punctuation, '')] = index
        pkl.dump(vocab, open(self.data_type+'vocab.pkl',"wb"))
        return vocab

    def sentence_to_word_ids(self, sentence, padd_len):
        actual_sent = []
        for word in sentence:
            if word.isdigit():
                actual_sent.append(self.vocab['NUM'])
            elif word.lower().replace(string.punctuation, '') in self.vocab:
                actual_sent.append(self.vocab[word.lower().replace(string.punctuation, '')])
            else:
                print("Unknown Word:" + word)
                actual_sent.append(self.vocab['UNK'])
        actual_sent_len = len(actual_sent)
        if  actual_sent_len < padd_len:
            padded_sent = actual_sent + [self.vocab['UNK']] * (padd_len - actual_sent_len)
        return padded_sent, actual_sent_len

    def read_raw_data(self):
        print("Reading raw data")
        if self.load_data_from_dumps():
            return
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                if line is '\n':
                    continue
                count += 1
                line = line.split('\t')
                if self.data_type == 'dev':
                    line = line[1:]
                question = re.sub('([.,!?()-])', r'  \1  ', line[0])
                question = question.split(' ')
                question = [word for word in question if word != '' ]
                passage = re.sub('([.,!?()-])', r'  \1  ', line[1])
                passage = passage.split(' ')
                passage = [word for word in passage if word != '' ]
                label = line[2]
                if(count < 100):
                    print("************************")
                    print(question)
                    print(passage)
                    print(label)
                    print("************************")
                padded_question, question_len = self.sentence_to_word_ids(question, self.max_question_len)
                self.questions.append(padded_question)
                self.questions_len.append(question_len)
                padded_passage, passage_len = self.sentence_to_word_ids(passage, self.max_passage_len)
                self.passages.append(padded_passage)
                self.passages_len.append(passage_len)
                if int(label) == 0:
                    self.labels.append([1, 0])
                else:
                    self.labels.append([0, 1])
        pkl.dump((self.questions, self.questions_len, self.passages, self.passages_len, self.labels), open(self.data_type+'data.pkl', 'wb'))
        assert(len(self.questions) == len(self.passages) == len(self.labels))

    def get_next_batch(self, dssm=False):
        # print("Batch id"+ str(self.batch_id))
        if self.batch_id == len(self.questions):
            self.batch_id = 0
        batch_questions = self.questions[self.batch_id:min(self.batch_id + self.batch_size,
                                        len(self.questions))]
        batch_questions_len = self.questions_len[self.batch_id:min(self.batch_id + self.batch_size,
                                        len(self.questions_len))]
        batch_passages = self.passages[self.batch_id:min(self.batch_id + self.batch_size,
                                        len(self.passages))]
        batch_passages_len = self.passages_len[self.batch_id:min(self.batch_id + self.batch_size,
                                        len(self.passages_len))]
        batch_labels = self.labels[self.batch_id:min(self.batch_id  + self.batch_size,
                                   len(self.labels))]
        batch_dssm = self.dssm_vectors[self.batch_id:min(self.batch_id + self.batch_size,   
                                                len(self.dssm_vectors))]
        self.batch_id = min(self.batch_id  + self.batch_size, len(self.labels)) 
        
        if dssm:
            return batch_questions, batch_questions_len, batch_passages, batch_passages_len, np.asarray(batch_dssm), np.asarray(batch_labels)
        return batch_questions, batch_questions_len, batch_passages, batch_passages_len, batch_labels
    
    def read_raw_data_with_dssm(self):
        print("Read DSSM Vectors")
        if self.load_dssm_from_dumps():
            return
        with open(self.dssm_file_path, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                if line is '\n':
                    continue
                count += 1
                line = line.split('\t', maxsplit=3)
                final_line = line[3].strip('\n').split('\t')
                if len(final_line) == 302:
                    final_line = final_line[0:300] + [0.0]*300
                self.dssm_vectors.append(final_line)
        pkl.dump(self.dssm_vectors, open(self.data_type + 'dssm.pkl', 'wb'))
        assert(len(self.dssm_vectors) == len(self.questions))
    
    def load_data_from_dumps(self):
        if path.exists(self.data_type+'data.pkl'):
            data = pkl.load(open(self.data_type+'data.pkl', 'rb'))
            self.questions = data[0]
            self.questions_len = data[1]
            self.passages = data[2]
            self.passages_len = data[3]
            self.labels = data[4]
            assert(len(self.questions) == len(self.questions_len) == len(self.passages) == len(self.passages_len) == len(self.labels))
            return True
        return False
    
    def load_dssm_from_dumps(self):
        if path.exists(self.data_type+'dssm.pkl'):
            self.dssm_vectors = pkl.load(open(self.data_type+'dssm.pkl', 'rb'))
            assert(len(self.dssm_vectors) == len(self.questions))
            return True
        return False
    
    def get_data(self, dssm=False):
        if dssm: 
            return self.questions, self.questions_len, self.passages, self.passages_len, np.asarray(self.dssm_vectors), np.asarray(self.labels)
        return self.questions, self.questions_len, self.passages, self.passages_len, np.asarray(self.labels)