from Data import Data
from FFNN import train
import json

if __name__ == "__main__":
    ques_pass_data_train = Data("D:\\PycharmProjects\\DataScienceChallenge\\Data\\train.txt", "D:\\PycharmProjects\\DataScienceChallenge\\Data\\train_dssm_new.txt", data_type='train', batch_size = 10)
    ques_pass_data_dev = Data("D:\\PycharmProjects\\DataScienceChallenge\\Data\\dev_w_id.txt", "D:\\PycharmProjects\\DataScienceChallenge\\Data\\dev_dssm_new", data_type='dev', vocab=ques_pass_data_train.vocab)
    '''with open('vocab.txt', 'w') as f:
        f.write(json.dumps(ques_pass_data.vocab))
    for i in range(0, 1):
        batch_questions, batch_questions_len, batch_passages, batch_passages_len, batch_dssm, batch_labels = ques_pass_data.get_next_batch(dssm=True)
        print(batch_dssm.shape)
        # print(batch_dssm)
        # print(batch_questions, batch_questions_len, batch_passages, batch_passages_len)'''
    train()