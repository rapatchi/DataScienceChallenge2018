from Data import Data
from FFNN import train
import pickle as pkl
import json

if __name__ == "__main__":
    # ques_pass_data_train = Data("D:\\PycharmProjects\\DataScienceChallenge\\Data\\train.txt", "D:\\PycharmProjects\\DataScienceChallenge\\Data\\train_dssm_new.txt", data_type='train', batch_size = 10)
    # ques_pass_data_dev = Data("D:\\PycharmProjects\\DataScienceChallenge\\Data\\dev_w_id.txt", "D:\\PycharmProjects\\DataScienceChallenge\\Data\\dev_dssm_new.txt", data_type='dev', vocab=pkl.load(open('trainvocab.pkl', 'rb')))
    '''_, _, _, _, x, y = ques_pass_data_dev.get_data(True)
    for index in range(0, x.shape[0]):
        if not len(x[index]) == 600:
            print("Length not 600 for "+ str(index) + " actual length:" + str(len(x[index])))
            print(x[index])
    with open('vocab.txt', 'w') as f:
        f.write(json.dumps(ques_pass_data.vocab))
    for i in range(0, 10):
        batch_questions, batch_questions_len, batch_passages, batch_passages_len, batch_labels = ques_pass_data_train.get_next_batch()
        # print(batch_dssm.shape)
        # print(batch_dssm)
        # print(batch_questions, batch_questions_len, batch_passages, batch_passages_len)'''
    train()