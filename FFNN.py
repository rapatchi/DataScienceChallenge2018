import tensorflow as tf
from Data import Data

# Parameters
learning_rate = 0.1
num_steps = 300 
batch_size = 5000
display_step = 10
drop_out = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 600 # DSSM data input 
num_classes = 2 # Binary Classifier  

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Dropout on layer 1
    layer_1_dropout = tf.nn.dropout(layer_1, keep_prob = drop_out)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_dropout, weights['h2']), biases['b2']))
    # Dropout on layer 2
    layer_2_dropout = tf.nn.dropout(layer_2, keep_prob = drop_out)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1_dropout, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def train():
    ques_pass_data_train = Data("D:\\PycharmProjects\\DataScienceChallenge\\Data\\train.txt", "D:\\PycharmProjects\\DataScienceChallenge\\Data\\train_dssm_new.txt", data_type='train', batch_size = batch_size)
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        for step in range(1, num_steps+1):
            _, _, _, _, batch_x, batch_y = ques_pass_data_train.get_next_batch(True)
            # print(batch_x.shape)
            # print(batch_y.shape)
            # Run optimization op (backprop)
            try:
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            except:
                print("Exception occured")
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                    Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    "{:.3f}".format(acc))
        print("Optimization Finished!")
        '''_, _, _, _, x_train, y_train = ques_pass_data_train.get_data(True)
        print(x_train.shape)
        print(y_train.shape)
        print("Train Accuracy:", \
            sess.run(accuracy, feed_dict={X: x_train,
                                        Y: y_train}))'''
 
        # Calculate accuracy for MNIST test images
        ques_pass_data_dev = Data("D:\\PycharmProjects\\DataScienceChallenge\\Data\\dev_w_id.txt", "D:\\PycharmProjects\\DataScienceChallenge\\Data\\dev_dssm_new.txt", data_type='dev', vocab=ques_pass_data_train.vocab)
        _, _, _, _, x_dev, y_dev = ques_pass_data_dev.get_data(True)
        acc, pred =sess.run([accuracy, prediction], feed_dict={X: x_dev,
                                        Y: y_dev})
        print("Dev Accuracy:", acc)
        with open('results.csv', 'w') as res:
            for i in range(0, pred.shape[0]):
                res.write(str(i+1) +"," + str(pred[i][1])+"\n")
