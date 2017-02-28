import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import csv

batch_size    = 50
max_iters     = 10
learning_rate = 1e-5
# 0.01 originally

def binarize(images, threshold=0.1):
    return (threshold < images).astype('float32')

def getLinearLayer(X, dimIN, dimOUT):
    w = tf.Variable(tf.random_normal((dimIN,dimOUT), stddev = 0.01))
    b = tf.Variable(tf.random_normal((1,dimOUT) , stddev = 0.01))
    probY = tf.matmul(X,w)+b
    return probY

def getHiddenLayer(X, dimIN, dimOUT):
    # ReLU hidden layer
    w_h = tf.Variable(tf.random_normal([dimIN, dimOUT], stddev=0.01))
    b_h = tf.Variable(tf.random_normal([1, dimOUT], stddev=0.01))
    h = tf.nn.relu(tf.matmul(X, w_h)+b_h)
    output = h
    return output


def getLSTM(X, nunits):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(nunits, forget_bias=1.0)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)
    final_outputs = outputs[:, -1, :]

    print("X shape", X.get_shape())
    print("Outputs shape", outputs.get_shape())
    print("State shape", state)
    print("Final outputs shape", final_outputs.get_shape())

    return final_outputs


def getModel(X, dimIN, dimOUT):
    n_lstm_units = 128
    n_hidden_units = 100

    # RNN layer
    lstm_outputs = getLSTM(X, n_lstm_units)
    print("1. lstm outputs shape", lstm_outputs.get_shape())

    # 100 unit hidden layer with ReLu
    hidden_outputs = getHiddenLayer(lstm_outputs, n_lstm_units, n_hidden_units)
    print("2. hidden outputs shape", hidden_outputs.get_shape())

    # linear layer to produce 10 class output
    scoreY = getLinearLayer(hidden_outputs, n_hidden_units, dimOUT)
    print("3. scoreY shape", scoreY.get_shape())

    return scoreY



def main():

    print ("Loading the data......")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

    ntraindata = 20000
    ntestdata = 10000
    dimX = 784
    dimY = 10

    trainX, trainY, testX, testY = mnist.train.images[0:ntraindata], mnist.train.labels[0:ntraindata], \
                                            mnist.test.images[0:ntestdata], mnist.test.labels[0:ntestdata]

    trainX = binarize(trainX)
    testX = binarize(testX)
    print("Converted to binary")

    (nrTrainSamples, dimX) = trainX.shape
    (nrTestSamples, dimY)  = testY.shape

    print ("Nr of training samples:", nrTrainSamples)
    print ("Nr of testing  samples:", nrTestSamples)
    print ("Dimension of X:", dimX)
    print ("Dimension of Y:", dimY)

    print("Reshaping")
#    trainXreshape = np.reshape(trainX, (nrTrainSamples, dimX, 1))
#    testXreshape = np.reshape(testX, (nrTestSamples, dimX, 1))
    trainXreshape = np.reshape(trainX, (nrTrainSamples, 784, 1))
    testXreshape = np.reshape(testX, (nrTestSamples, 784, 1))

    print("Original", trainX.shape)
    print("Reshaped", trainXreshape.shape)



    # Build Model
    #X = tf.placeholder("float", [None, dimX, 1])
    #X = tf.placeholder("float", [None, 784, 1])
    #Y = tf.placeholder("float", [None, dimY])
    X = tf.placeholder("float", [batch_size, 784, 1])
    Y = tf.placeholder("float", [batch_size, dimY])

    modelscores = getModel(X, dimX, dimY)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(modelscores, Y))

    predict_op = tf.argmax(modelscores, 1)
    accuracy   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y,1), predict_op), tf.float32))

    # Train model
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


    with tf.Session() as sess:

        # initialize variables
        tf.global_variables_initializer().run()

        print ("Training started........")

        for indexIter in range(max_iters):

            print("Iteration", indexIter + 1)

            for startIndex, endIndex in zip(range(0,len(trainXreshape), batch_size), \
                                        range(batch_size, len(trainXreshape), batch_size)):

                if (startIndex) % 1000 == 0:
                    print("StartIndex", startIndex)

                sess.run(train_op, feed_dict={X: trainXreshape[startIndex:endIndex], Y: trainY[startIndex:endIndex]})


            #if (indexIter+1)%1==0:
                #acc_train = sess.run(accuracy, feed_dict={X:trainXreshape, Y:trainY})
                #print ("Accuracy(training)", acc_train)

                #acc_test  = sess.run(accuracy, feed_dict={X:testXreshape,  Y:testY})
                #print ("Accuracy(test)", acc_test)

                #loss_train = sess.run(loss, feed_dict={X: trainXreshape, Y: trainY})
                #print ("Training loss", loss_train)

        print ("Training finished.")



if __name__ == '__main__':
    tf.set_random_seed(123)
    main()
