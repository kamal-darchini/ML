from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error


class MLP:

    def __init__(self, *args, **kwargs):
        # Parameters
        self.learning_rate = 0.001
        self.training_epochs = 50
        self.batch_size = 4000
        self.display_step = 1
        self.activation_function = tf.nn.tanh

        # Network Parameters
        self.n_input = None # data input
        self.n_hidden_1 = 150 # 1st layer number of neurons
        self.n_hidden_2 = 256 # 2nd layer number of neurons
        self.n_hidden_3 = 300
        self.n_hidden_4 = 400
        self.n_hidden_5 = 256
        self.n_classes = None # total classes

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [None, self.n_classes])

    # MLP model
    def multilayer_perceptron(self, x):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1']))
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3']))
        layer_4 = tf.nn.tanh(tf.add(tf.matmul(layer_3, self.weights['h4']), self.biases['b4']))
        layer_5 = tf.nn.tanh(tf.add(tf.matmul(layer_4, self.weights['h5']), self.biases['b5']))
        out_layer = tf.nn.relu(tf.matmul(layer_5, self.weights['out']) + self.biases['out'])
        return out_layer

    def fit(self, feature: np.ndarray, label: np.ndarray, valid_feature: np.ndarray, vali_label: np.ndarray, test_features):
        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3])),
            'h4': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_hidden_4])),
            'h5': tf.Variable(tf.random_normal([self.n_hidden_4, self.n_hidden_5])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_5, self.n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([self.n_hidden_3])),
            'b4': tf.Variable(tf.random_normal([self.n_hidden_4])),
            'b5': tf.Variable(tf.random_normal([self.n_hidden_5])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Construct model
        logits = self.multilayer_perceptron(self.X)

        # Define loss and optimizer
        # loss_op = tf.reduce_mean(tf.losses.mean_squared_error(predictions=logits, labels=self.Y))
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)
        # Initializing the variables
        init = tf.global_variables_initializer()

        print('started training ...')
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_batch = int(feature.shape[0]/self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    if i * self.batch_size - 1 <= feature.shape[1]:
                        batch_x = feature[(i-1) * self.batch_size: i * self.batch_size - 1, :]
                        batch_y = label[(i - 1) * self.batch_size: i * self.batch_size - 1, :]
                    else:
                        batch_x = feature[(i - 1) * self.batch_size:, :]
                        batch_y = label[(i - 1) * self.batch_size:, :]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict={self.X: batch_x,
                                                                    self.Y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Test model
                pred = tf.nn.softmax(logits)  # Apply softmax to logits
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print('=========================')
                print("Epoch:", '%04d' % (epoch+1), "Accuracy:", accuracy.eval({self.X: valid_feature, self.Y: vali_label}))
            print("Optimization Finished!")

            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print('****************************')
            print("Total Accuracy:", accuracy.eval({self.X: valid_feature, self.Y: vali_label}))
            print('+++++++++++++++++++++++++++++')
            preds = pred.eval({self.X: valid_feature, self.Y: vali_label})
            predictions = [int(np.argmax(pred)) for pred in preds]
            label_true = [int(np.where(x == 1)[0][0]) for x in vali_label]
            mse = mean_squared_error(label_true, predictions)
            rmse = np.sqrt(mse)

            print('Number of predicted labels is ', len(preds))
            print('RMSE: ', rmse)
            print("Total vehicles are: ", sum(label_true))

            pred_test = self.multilayer_perceptron(self.X)
            test_predictions = [int(np.argmax(pred)) for pred in pred_test.eval({self.X: test_features})]

        return test_predictions

