from __future__ import print_function
import tensorflow as tf
import numpy as np

tf.reset_default_graph()


class LSTM:

    def __init__(self):
        self.num_periods = 20
        self.inputs = 1
        self.hidden = 100
        self.output = 1
        self.learning_rate = 0.001
        self.epochs = 50
        self.batch_size = 4000

        self.X = tf.placeholder(tf.float32, [None, self.num_periods, self.inputs])
        self.Y = tf.placeholder(tf.float32, [None, self.num_periods, self.output])

        self.basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.hidden, activation=tf.nn.relu)
        self.rnn_output, self.state = tf.nn.dynamic_rnn(self.basic_cell, self.X, dtype=tf.float32)

        self.stacked_rnn_output = tf.reshape(self.rnn_output, [-1, self.hidden])
        self.stacked_outputs = tf.layers.dense(self.stacked_rnn_output, self.output)
        self.outputs = tf.reshape(self.stacked_outputs, [-1, self.num_periods, self.output])

        self.loss = tf.reduce_sum(tf.square(self.outputs - self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()

    def fit(self, feature, label, valid_feature, valid_label):

        with tf.Session() as sess:
            self.init.run()
            for ep in range(self.epochs):
                total_batch = int(feature.shape[0]/self.batch_size)
                for i in range(total_batch):
                    if i * self.batch_size - 1 <= feature.shape[1]:
                        batch_x = feature[(i-1) * self.batch_size: i * self.batch_size - 1, :]
                        batch_y = label[(i - 1) * self.batch_size: i * self.batch_size - 1, :]
                    else:
                        batch_x = feature[(i - 1) * self.batch_size:, :]
                        batch_y = label[(i - 1) * self.batch_size:, :]
                    sess.run(self.training_op, feed_dict={self.X: batch_x, self.Y: batch_y})

                if ep % 100 == 0:
                    mse = self.loss.eval(feed_dict={self.X: feature, self.Y: label})
                    print('MSE for Epoch ', ep, ' is ', mse)

            preds = sess.run(self.outputs, feed_dict={self.X: valid_feature})
            square_error = 0
            skipped = 0
            total_vehicle = 0
            for i, pred in enumerate(preds):
                try:
                    pred_int = int(np.where(pred == 1)[0][0])
                except IndexError:
                    skipped += 1
                    continue
                label_int = int(np.where(valid_label[i] == 1)[0][0])
                square_error += (pred_int - label_int) ** 2
                total_vehicle += label_int

            print('Number of skipped features is', skipped)
            print('Number of predicted labels is ', len(preds))
            rmse = np.sqrt(square_error / (len(preds) - skipped))
            print('RMSE: ', rmse)
            print("Total vehicles are: ", total_vehicle)







