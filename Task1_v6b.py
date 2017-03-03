import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import copy
import sys
import time

# Global parameters
## unchanging
nclasses        = 10
batch_size      = 256
relu_units      = 100
num_steps       = 784
step_depth      = 1

# master
what_to_do      = "train" # alternatives are: "optimise1", "optimise2", "train", "restore"
rnn_cell        = sys.argv[5] # alternatives are: "lstm", "stacked lstm", "gru", "stacked gru"
#rnn_cell        = "gru" # alternatives are: "lstm", "stacked lstm", "gru", "stacked gru"
rnn_units       = int(sys.argv[6])
#rnn_units       = 32
ntrain_data     = int(sys.argv[3]) # should be 10k for optimisation and 55k for training
ntest_data      = int(sys.argv[4]) # should be 10k for training
grad_clip       = True
batch_norm      = True

# optimisation
max_op_iters    = 5
lr_low          = 0.001
lr_high         = 0.1
# optimise_csv    = "C:/Users/Tariq/Desktop/UCL/GI13 Advanced/Assignment 2/T1_lstm_32units_optimised.csv"
optimise_csv    = "T1_gru_32units_optimised.csv"

# training
#lr_train        = 0.01
lr_train        = sys.argv[1]
max_iters       = int(sys.argv[2])
epsilon         = 1e-3
#train_ckpt      = "C:/Users/Tariq/Desktop/UCL/GI13 Advanced/Assignment 2/T1_trained.ckpt"
#train_csv       = "C:/Users/Tariq/Desktop/UCL/GI13 Advanced/Assignment 2/T1_trained.csv"

train_ckpt      = "T1_trained.ckpt"
train_csv       = "T1_trained.csv"


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

    if batch_norm == True:
        # Batch normalise
        z_bn = tf.matmul(X, w_h)
        batch_mean, batch_var = tf.nn.moments(z_bn, [0])
        z_hat = (z_bn - batch_mean) / tf.sqrt(batch_var + epsilon)

        scale_h = tf.Variable(tf.ones([dimOUT]))
        beta_h = tf.Variable(tf.zeros([dimOUT]))

        bn = scale_h * z_hat + beta_h
        h = tf.nn.relu(bn)
    else:
        b_h = tf.Variable(tf.random_normal([1, dimOUT], stddev=0.01))

        h = tf.nn.relu(tf.matmul(X, w_h)+b_h)
    output = h
    return output

def getRNN(X, nunits):
    if rnn_cell == "lstm":
        print ("Cell used: ", "lstm", nunits)
        cell = tf.nn.rnn_cell.BasicLSTMCell(nunits, forget_bias=1.0)
    elif rnn_cell == "gru":
        print ("Cell used: ", "gru", nunits)
        cell = tf.nn.rnn_cell.GRUCell(nunits)
    elif rnn_cell == "stacked lstm":
        print ("Cell used: ", "stacked lstm", nunits)
        onecell = tf.nn.rnn_cell.BasicLSTMCell(nunits, forget_bias=1.0)
        cell = tf.nn.rnn_cell.MultiRNNCell([onecell] * 3, state_is_tuple=True)
    elif rnn_cell == "stacked gru":
        print ("Cell used: ", "stacked gru", nunits)
        onecell = tf.nn.rnn_cell.GRUCell(nunits)
        cell = tf.nn.rnn_cell.MultiRNNCell([onecell] * 3, state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    final_outputs = outputs[:, -1, :]
    return final_outputs

def getModel(X, dimIN, dimOUT):
    # RNN layer
    lstm_outputs = getRNN(X, rnn_units)
    # 100 unit hidden layer with ReLu
    hidden_outputs = getHiddenLayer(lstm_outputs, rnn_units, relu_units)
    # linear layer to produce 10 class output
    scoreY = getLinearLayer(hidden_outputs, relu_units, dimOUT)
    return scoreY


def main():
    print ("Loading the data......")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    trainX, trainY, testX, testY = mnist.train.images[0:ntrain_data], mnist.train.labels[0:ntrain_data], \
                                            mnist.test.images[0:ntest_data], mnist.test.labels[0:ntest_data]
    trainX = binarize(trainX)
    testX = binarize(testX)
    print("Converted to binary")

    (nrTrainSamples, dimX) = trainX.shape
    (nrTestSamples, dimY)  = testY.shape

    trainXreshape = np.reshape(trainX, (ntrain_data, num_steps, step_depth))
    testXreshape = np.reshape(testX, (ntest_data, num_steps, step_depth))


    # Build Model
    X = tf.placeholder(tf.float32, [batch_size, num_steps, step_depth])
    Y = tf.placeholder(tf.float32, [batch_size, nclasses])
    learning_rate = tf.placeholder("float", shape=[])
    #X = tf.placeholder("float", [None, num_steps, step_depth])
    #Y = tf.placeholder("float", [None, nclasses])
    modelscores = getModel(X, num_steps, nclasses)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(modelscores, Y))

    if grad_clip==False:
        # No gradient clipping
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    else:
        # Gradient clipping
        optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gvs = optimiser.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimiser.apply_gradients(capped_gvs)

    predict_op = tf.argmax(modelscores, 1)
    accuracy   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y,1), predict_op), tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # initialize variables
        tf.global_variables_initializer().run()

        def calc_lossandacc(xdata, ylabels, datalength):
            acc = 0
            theloss = 0

            count = 0
            for startIndex, endIndex in zip(range(0, datalength, batch_size), \
                                            range(batch_size, datalength, batch_size)):
                count = count + 1
                # if (startIndex) % 5000 == 0:
                #     print("Performance calcs ... startIndex", startIndex)
                acc = acc + sess.run(accuracy,
                                     feed_dict={X: xdata[startIndex:endIndex], Y: ylabels[startIndex:endIndex]})
                theloss = theloss + sess.run(loss, feed_dict={X: xdata[startIndex:endIndex], Y: ylabels[startIndex:endIndex]})
            acc = acc / count
            theloss = theloss / count
            return acc, theloss

        print("What to do?.... ", what_to_do)

        if what_to_do == "optimise1":
            out = open(optimise_csv, 'w')
            out.write('%s;' % "Number training datapoints")
            out.write('%s;' % "Number test datapoints")
            out.write('%s;' % "Learning rate")
            out.write('%s;' % "Test accuracy")


            # Instability means just cycle through learning rates
            for lrIndex in range(0,7):

                this_lr = 10**(-lrIndex-1)

                # Batched training
                tf.global_variables_initializer().run()
                for startIndex, endIndex in zip(range(0, len(trainXreshape), batch_size), \
                                            range(batch_size, len(trainXreshape), batch_size)):
                    if (startIndex) % 5000 == 0:
                        print("Optimisation training ... startIndex", startIndex)
                    sess.run(train_op, feed_dict={X: trainXreshape[startIndex:endIndex], \
                                              Y: trainY[startIndex:endIndex], learning_rate: this_lr})
                opt_acc, opt_loss = calc_lossandacc(testXreshape, testY, len(testXreshape))

                print("this lr", this_lr, "accuracy", opt_acc)
                out.write('\n')
                out.write('%d;' % ntrain_data)
                out.write('%d;' % ntest_data)
                out.write('%f;' % this_lr)
                out.write('%f;' % opt_acc)

            out.close()

        elif what_to_do == "optimise2":
            out = open(optimise_csv, 'w')
            out.write('%s;' % "Number training datapoints")
            out.write('%s;' % "Number test datapoints")
            out.write('%s;' % "Learning rate")
            out.write('%s;' % "Test accuracy")

            op_iter = 0
            global lr_low, lr_high
            while (op_iter < max_op_iters):
                op_iter = op_iter + 1
                print("Optimisation iteration", op_iter)
                print("lr_low", lr_low)
                print("lr_high", lr_high)

                lr_mid = copy.deepcopy((lr_low + lr_high) / 2)

                # re-initialize variables
                tf.global_variables_initializer().run()

                # Batched training
                for startIndex, endIndex in zip(range(0,len(trainXreshape), batch_size), \
                                        range(batch_size, len(trainXreshape), batch_size)):
                    if (startIndex) % 5000 == 0:
                        print("Left training ... startIndex", startIndex)
                    sess.run(train_op, feed_dict={X: trainXreshape[startIndex:endIndex], \
                                                  Y: trainY[startIndex:endIndex],\
                                                  learning_rate: lr_mid * 0.98})
                # Batched accuracy calculation
                acc_left, loss_left = calc_lossandacc(testXreshape, testY, len(testXreshape))

                # re-initialize variables
                tf.global_variables_initializer().run()

                # Batched training
                for startIndex, endIndex in zip(range(0,len(trainXreshape), batch_size), \
                                        range(batch_size, len(trainXreshape), batch_size)):
                    if (startIndex) % 5000 == 0:
                        print("Right training ... startIndex", startIndex)
                    sess.run(train_op, feed_dict={X: trainXreshape[startIndex:endIndex], \
                                                  Y: trainY[startIndex:endIndex],\
                                                  learning_rate: lr_mid * 1.02})
                # Batched accuracy calculation
                acc_right, loss_right = calc_lossandacc(testXreshape, testY, len(testXreshape))
                print("Mid learning rate", lr_mid)
                print("Acc left", acc_left, "Acc right", acc_right)


                if acc_left > acc_right:
                    lr_low = copy.deepcopy(lr_low)
                    lr_high = copy.deepcopy(lr_mid)
                    this_acc = copy.deepcopy(acc_left)
                else:
                    lr_low = copy.deepcopy(lr_mid)
                    lr_high = copy.deepcopy(lr_high)
                    this_acc = copy.deepcopy(acc_right)

                out.write('\n')
                out.write('%d;' % op_iter)
                out.write('%d;' % ntrain_data)
                out.write('%d;' % ntest_data)
                out.write('%f;' % this_acc)
                out.write('%f;' % lr_mid)
            out.close()


        elif what_to_do == "train":
            print("Learning rate", lr_train)

            best_acc = 1e-3

            out = open(train_csv, 'w')
            out.write('%s;' % rnn_cell)
            out.write('%d;' % rnn_units)
            out.write('\n')
            out.write('%s;' % "Epoch")
            out.write('%s;' % "Training accuracy")
            out.write('%s;' % "Training loss")
            out.write('%s;' % "Test accuracy")
            out.write('%s;' % "Test loss")

            for indexIter in range(max_iters):
                # print("Iteration", indexIter + 1)
                start_time = time.time()

                # Batched training
                for startIndex, endIndex in zip(range(0,len(trainXreshape), batch_size), \
                                        range(batch_size, len(trainXreshape), batch_size)):
                    # if (startIndex) % 5000 == 0:
                    #     print("Training ... startIndex", startIndex)
                    sess.run(train_op, \
                             feed_dict={X: trainXreshape[startIndex:endIndex], \
                                        Y: trainY[startIndex:endIndex], \
                                        learning_rate: lr_train})

                # Only calculate training accuracy on ntestpoints of training data
                acc_train, loss_train = calc_lossandacc(trainXreshape, trainY, len(testXreshape))
                acc_test, loss_test = calc_lossandacc(testXreshape, testY, len(testXreshape))

                out.write('\n')
                out.write('%d;' % indexIter)
                out.write('%f;' % acc_train)
                out.write('%f;' % loss_train)
                out.write('%f;' % acc_test)
                out.write('%f;' % loss_test)

                if acc_test > best_acc:
                    best_acc = acc_test
                    saver.save(sess, train_ckpt)

                print("Iteration", indexIter+1, "Tr acc", acc_train, "Te acc", acc_test, \
                                                        "Best acc", best_acc, "Time", time.time()-start_time)


            print ("Training finished.")
            # saver.save(sess, train_ckpt)
            out.close()
            print ("Trained model saved.")

        else:
            saver.restore(sess, train_ckpt)

            acc_train, loss_train = calc_lossandacc(trainXreshape, trainY, len(testXreshape))
            print("Training accuracy", acc_train)
            print("Training loss", loss_train)
            acc_test, loss_test = calc_lossandacc(testXreshape, testY, len(testXreshape))
            print("Test accuracy", acc_test)
            print("Test loss", loss_test)

if __name__ == '__main__':
    tf.set_random_seed(123)
    main()


