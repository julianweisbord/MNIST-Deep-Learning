'''
Created on January 10th, 2018
author: Julian Weisbord
sources: https://www.tensorflow.org/get_started/mnist/pros, https://www.youtube.com/watch?v=mynJtLhhcXk
description: Convolutional Neural Network to classify handwritten digits using
                  the MNIST Dataset.
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

N_CLASSES = 10
BATCH_SIZE = 128
KEEP_RATE = 0.8
N_EPOCHS = 15

WEIGHTS = {
    'W_conv1':tf.Variable(tf.random_normal([5, 5, 1, 32])),  # Convolve 5 * 5, take 1 input, produce 32 output features
    'W_conv2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out':tf.Variable(tf.random_normal([1024, N_CLASSES]))}

BIASES = {
    'b_conv1':tf.Variable(tf.random_normal([32])),
    'b_conv2':tf.Variable(tf.random_normal([64])),
    'b_fc':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([N_CLASSES]))}

def conv2d(x, W):
    '''
    Convolves the input image with the weight matrix, one pixel at a time.
    '''
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    '''
    Maxpooling is used to simplify and take the extreme values. It slides a
    2*2 window 2 pixels at a time.
    '''
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x, keep_prob):
    '''
    Create model for inference
    '''
    x = tf.reshape(x, shape=[-1, 28, 28, 1])  # Reshape to a 28 *28 tensor

    conv1 = tf.nn.relu(conv2d(x, WEIGHTS['W_conv1']) + BIASES['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.relu(conv2d(conv1, WEIGHTS['W_conv2']) + BIASES['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, WEIGHTS['W_fc']) + BIASES['b_fc'])
    fc = tf.nn.dropout(fc, keep_prob)
    output = tf.matmul(fc, WEIGHTS['out']) + BIASES['out']

    return output

def train_neural_network(x, y, keep_prob):
    '''
    Compute loss, train the model, and print out the accuracy of the model.
    '''
    prediction = convolutional_neural_network(x, keep_prob)
    # Compute loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(N_EPOCHS):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / BATCH_SIZE)):
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: KEEP_RATE})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', N_EPOCHS, 'loss:', epoch_loss)
        # Compute the accuracy of the model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels, keep_prob: 1.}))

def main():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, shape=[None, 10])
    train_neural_network(x, y, keep_prob)

if __name__ == '__main__':
    main()
