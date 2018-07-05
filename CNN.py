""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.

This example is using the MNIST database of handwritten digits

(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien

Project: https://github.com/aymericdamien/TensorFlow-Examples/

"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from Data import Data
# Training Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 540
display_step = 10
# Network Parameters
num_input = 600 
num_classes = 2 
dropout = 0.75 # Dropout, probability to keep units
# Import Data
trainset = Data("D:\\PycharmProjects\\DataScienceChallenge\\Data\\train.txt", "D:\\PycharmProjects\\DataScienceChallenge\\Data\\train_dssm_new.txt", data_type='train', batch_size = batch_size)
# tf Graph input

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)# Create some wrappers for simplicity

def conv1d(x, F, b, stride=5):
   # Conv2D wrapper, with bias and relu activation
   x = tf.nn.conv1d(x, F, stride, padding='VALID')
   x = tf.nn.bias_add(x, b)
   return tf.nn.relu(x)
def maxpool1d(x, k=[1, 6, 1, 1]):
    # MaxPool1D wrapper
    return tf.nn.max_pool(x, ksize=k, strides=[1, 1, 1, 1],
                            padding='VALID')# Create model

def conv_net(x, weights, biases, dropout):
   # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
   # Reshape to match picture format [Height x Width x Channel]
   # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
   print("Before Reshape"+ str(x.shape))
   x = tf.reshape(x, shape=[-1, num_input, 1])   # Convolution Layer
   print("After Reshape"+ str(x.shape))
   conv1 = conv1d(x, weights['wc1'], biases['bc1'])
   print("conv1"+ str(conv1))
   # Max Pooling (down-sampling)
   conv1_r = tf.reshape(conv1, shape=[-1, 50, 1, 1])
   print("conv1 reshape"+ str(conv1_r))
   conv1_p = maxpool1d(conv1_r)   # Convolution Layer
   print("pool1"+ str(conv1_p))
   # conv2 = conv1d(conv1, weights['wc2'], biases['bc2'])
   # Max Pooling (down-sampling)
   # conv2 = maxpool1d(conv2, k=2)   # Fully connected layer
   # Reshape conv2 output to fit fully connected layer input
   fc1 = tf.reshape(conv1_p, [-1, weights['wd1'].get_shape().as_list()[0]])
   fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
   fc1 = tf.nn.relu(fc1)
   print("fc1"+ str(fc1))
   # Apply Dropout
   fc1 = tf.nn.dropout(fc1, dropout)   # Output, class prediction
   out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
   print("out"+ str(out))
   return out
# Store layers weight & bias

weights = {
   #  conv, 1 input, 32 outputs
   'wc1': tf.Variable(tf.random_normal([355, 1, 1])),
   # 5x5 conv, 32 inputs, 64 outputs
   # 'wc2': tf.Variable(tf.random_normal([64])),
   # fully connected, 7*7*64 inputs, 1024 outputs
   'wd1': tf.Variable(tf.random_normal([45, 32])),
   # 1024 inputs, 10 outputs (class prediction)
   'out': tf.Variable(tf.random_normal([32, num_classes]))
}
biases = {
   'bc1': tf.Variable(tf.random_normal([1])),
   # 'bc2': tf.Variable(tf.random_normal([64])),
   'bd1': tf.Variable(tf.random_normal([32])),
   'out': tf.Variable(tf.random_normal([num_classes]))

}
# Construct model

logits = conv_net(X, weights, biases, keep_prob)

prediction = tf.nn.softmax(logits)
# Define loss and optimizer

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
   logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)# Evaluate model

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initialize the variables (i.e. assign their default value)

init = tf.global_variables_initializer()
# Start training

with tf.Session() as sess:   # Run the initializer
   sess.run(init)   
   for step in range(1, num_steps+1):
       _, _, _, _, batch_x, batch_y = trainset.get_next_batch(True)
       # Run optimization op (backprop)
       sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
       if step % display_step == 0 or step == 1:
           # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                Y: batch_y,
                                                                keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                "{:.4f}".format(loss) + ", Training Accuracy= " + \
                "{:.3f}".format(acc))