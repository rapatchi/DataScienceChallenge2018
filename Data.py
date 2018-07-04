import pandas as pd
import tensorflow as tf
import string
import re

class Data(object):
    def __init__(self, filepath, batch_size):
        self.filepath = filepath
        self.max_question_len = 180  # 176
        self.max_passage_len = 1000  # 989
        self.questions = []
        self.questions_len = []
        self.passages = []
        self.passages_len = []
        self.labels = []
        self.batch_id = 0
        self.batch_size = batch_size
        self.vocab = self.build_vocab()
        self.read_raw_data()

    def preprocess(self):
        data = pd.read_csv(self.filepath, sep='\t+', engine='python',
                        names=['question', 'passage', 'relation'])
        print(data.columns)
        print("Data Types {0}", data.dtypes)

    def build_vocab(self):
        vocab = {'0': 0, '1': 1, 'UNK': 2, 'NUM': 3}
        index = 1
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
                        vocab[word.lower()] = index
        return vocab

    def sentence_to_word_ids(self, sentence, padd_len):
        actual_sent = []
        for word in sentence:
            if word.isdigit():
                actual_sent.append(self.vocab['NUM'])
            elif word.lower() in self.vocab:
                actual_sent.append(self.vocab[word.lower()])
            else:
                print("Unknown Word: {0}", word)
                actual_sent.append(self.vocab['UNK'])
        actual_sent_len = len(actual_sent)
        if  actual_sent_len < padd_len:
            padded_sent = actual_sent + [self.vocab['UNK']] * (padd_len - actual_sent_len)
        return padded_sent, actual_sent_len

    def read_raw_data(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                count += 1
                line = line.split('\t')
                question = re.sub('([.,!?()-])', r'  \1  ', line[0])
                question = question.split(' ')
                question = [word for word in question if word != '' ]
                passage = re.sub('([.,!?()-])', r'  \1  ', line[1])
                passage = passage.split(' ')
                passage = [word for word in passage if word != '' ]
                label = line[2]
                if(count < 10):
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
                self.labels.append(label)

    def get_next_batch(self):
        if self.batch_id == len(self.questions) == len(self.passages) == len(self.labels):
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
        self.batch_id = min(self.batch_id  + self.batch_size, len(self.labels))
        return batch_questions, batch_questions_len, batch_passages, batch_passages_len, batch_labels
