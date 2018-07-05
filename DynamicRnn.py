from __future__ import print_function

import tensorflow as tf
import random
import numpy as np
from itertools import *
from Data import *

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_steps = 50
batch_size = 500
display_step = 5
embedding = 100


# Network Parameters
seq_ques_max_len = 180  # Sequence max length
seq_passage_max_len = 1000
n_lstm_hidden = 32  # hidden layer num of features
hid1_size = 64
n_classes = 2  # linear sequence or not
feed_inputs_len = 4 * n_lstm_hidden

trainset = Data("D:\\PycharmProjects\\DataScienceChallenge\\Data\\train.txt", "D:\\PycharmProjects\\DataScienceChallenge\\Data\\train_dssm_new.txt", data_type='train', batch_size = batch_size)
#testset = ToySequenceData(n_samples=500, max_seq_len=seq_ques_max_len)
vocab_size = len(trainset.vocab)
# tf Graph input
x_ques = tf.placeholder(tf.int32, [None, seq_ques_max_len])
x_passage = tf.placeholder(tf.int32, [None, seq_passage_max_len])
y = tf.placeholder(tf.int32, [None, n_classes])



# A placeholder for indicating each sequence length
seqlen_ques = tf.placeholder(tf.int32, [None])
seqlen_passage = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_lstm_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}
lookup = {
    'lookup': tf.Variable(tf.random_normal([vocab_size, embedding]))
} 

def DynamicBiRNNQues(x, seqlen, seq_max_len):


    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_lstm_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_lstm_hidden, forget_bias=1.0)
    print(type(x))
    print(x.shape)
    embed = tf.nn.embedding_lookup(lookup['lookup'], x)
    print(type(embed))
    print(embed.shape)
    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    with tf.variable_scope('question'):
        (encoder_outputs_fw, encoder_outputs_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, embed, dtype=tf.float32,sequence_length=seqlen)

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(encoder_outputs_fw)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    index_backward=tf.range(0, batch_size) * seq_max_len
    # Indexing
    outputs_fw = tf.gather(tf.reshape(encoder_outputs_fw, [-1, n_lstm_hidden]), index)
    outputs_bw = tf.gather(tf.reshape(encoder_outputs_bw, [-1, n_lstm_hidden]), index_backward)

    # return last vector in forward direct and first vector in backward direction
    return [outputs_fw,outputs_bw]

def DynamicBiRNNPassage(x, seqlen, seq_max_len):

    # Forward direction cell
    lstm_fw_cell_pass = tf.contrib.rnn.BasicLSTMCell(n_lstm_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell_pass = tf.contrib.rnn.BasicLSTMCell(n_lstm_hidden, forget_bias=1.0)
    print(type(x))
    print(x.shape)
    embed = tf.nn.embedding_lookup(lookup['lookup'], x)

    print(type(embed))
    print(embed.shape)
    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    (encoder_outputs_pass_fw, encoder_outputs_pass_bw), states_pass = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_pass,lstm_bw_cell_pass, embed, dtype=tf.float32,sequence_length=seqlen)

    batch_size = tf.shape(encoder_outputs_pass_fw)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    index_backward = tf.range(0, batch_size) * seq_max_len

    # Indexing
    outputs_ques_fw = tf.gather(tf.reshape(encoder_outputs_pass_fw, [-1, n_lstm_hidden]), index)
    outputs_ques_bw = tf.gather(tf.reshape(encoder_outputs_pass_bw, [-1, n_lstm_hidden]), index_backward)

    # return last vector in forward direct and first vector in backward direction
    return [outputs_ques_fw,outputs_ques_bw]



def feedForward(data):


    w1 = tf.Variable(tf.random_normal([hid1_size, feed_inputs_len], stddev=0.01), name='w1')
    b1 = tf.Variable(tf.random_normal([hid1_size, 1]), name='b1')
    y1 = tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(data)), b1))

    wo = tf.Variable(tf.random_normal([n_classes, hid1_size], stddev=0.01), name='wo')
    bo = tf.Variable(tf.random_normal([n_classes, 1]), name='bo')
    yo = tf.transpose(tf.add(tf.matmul(wo,y1), bo))

    return yo


passage_vectors = DynamicBiRNNPassage(x_passage, seqlen_passage, seq_passage_max_len)
ques_vectors = DynamicBiRNNQues(x_ques, seqlen_ques, seq_ques_max_len)
data=tf.concat([passage_vectors[0],passage_vectors[1],ques_vectors[0],ques_vectors[1]],axis=0)
data = tf.reshape(data,[-1,feed_inputs_len])
logits = feedForward(data)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_and_vars = optimizer.compute_gradients(cost)
global_step = tf.Variable(0, name="global_step", trainable=False)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    writer = tf.summary.FileWriter("output", sess.graph)
    for step in range(1, training_steps + 1):
        batch_x_ques,batch_seqlen_ques,batch_x_passage,batch_seqlen_pass, batch_y  = trainset.get_next_batch()
        batch_x_ques=np.asarray(batch_x_ques, dtype=int)
        batch_x_passage=np.asarray(batch_x_passage, dtype=int)
        # Run optimization op (backprop)
        sess.run([train_op, global_step, cost], feed_dict={x_passage: batch_x_passage, x_ques: batch_x_ques, y: batch_y,
                                       seqlen_ques: batch_seqlen_ques,seqlen_passage:batch_seqlen_pass})
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost],feed_dict={x_passage: batch_x_passage, x_ques: batch_x_ques, y: batch_y,
                                       seqlen_ques: batch_seqlen_ques,seqlen_passage:batch_seqlen_pass})
            print("Step " + str(step ) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    '''print("Optimization Finished!")
    # Calculate accuracy
    test_set = Data("D:\\PycharmProjects\\DataScienceChallenge\\Data\\dev_w_id.txt", "D:\\PycharmProjects\\DataScienceChallenge\\Data\\dev_dssm_new.txt", data_type='dev', vocab=trainset.vocab)
    ques, seqlen_ques, passage, seqlen_pass, _ = test_set.get_data()
    ques = np.asarray(ques).reshape([-1,ques,1])
    passage = np.asarray()
    acc, pred = sess.run([accuracy, prediction], feed_dict={x_passage: passage, x_ques: ques, y: y,
                                       seqlen_ques: seqlen_ques,seqlen_passage:seqlen_pass})
    print("Testing Accuracy:" + str(acc))
    with open('results.csv', 'w') as res:
        for i in range(0, pred.shape[0]):
            res.write(str(i+1) +"," + str(pred[i][1])+"\n")'''