import tensorflow as tf
from sklearn.neural_network import MLPClassifier

inp_vec=[]
label=[]
inp_vec_dev=[]
label_dev=[]

print("Read Training Data")
with open('../data/train_dssm_new.txt') as f:
    for line in f.readlines():
        data=line.split('\t')
        if(len(data[3:])<600):
                data=data[:-2]
                diff= 600 - len(data[3:])
                temp=[0]*diff
                data=data+temp
        inp_vec.append(data[3:])
        label.append(data[2])


print("Dev Data")
with open('../data/dev_dssm_new.txt') as f:
    for line in f.readlines():
        if len(line.strip())> 0:
            data = line.split('\t')
            if(len(data[3:])<600):
                data=data[:-2]
                diff= 600 - len(data[3:])
                temp=[0]*diff
                data=data+temp
            inp_vec_dev.append(data[3:])
            label_dev.append(data[2])

print("Train Model")
model = MLPClassifier(solver='adam', alpha=1e-5, random_state=1, batch_size= 512)
model.fit(inp_vec, label)

print("Test Model")
predicted = model.predict(inp_vec_dev)
correct=0
print("Calculate Accuracy")
for i in range(0,len(predicted)):
    if predicted[i] == label_dev[i]:
        correct+=1

print(correct)