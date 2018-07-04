from Data import Data
import json

if __name__ == "__main__":
    ques_pass_data = Data("..\\Data\\train.txt", 1)
    with open('vocab.txt', 'w') as f:
        f.write(json.dumps(ques_pass_data.vocab))
    for i in range(0, 1):
        batch_questions, batch_questions_len, batch_passages, batch_passages_len, batch_labels = ques_pass_data.get_next_batch()
        # print(batch_questions, batch_questions_len, batch_passages, batch_passages_len)