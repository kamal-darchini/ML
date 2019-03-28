import tensorflow as tf
import os
import shutil
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# ###READ DATA###

print("\nImporting the MNIST data")
mnist = read_data_sets("/tmp/data/", one_hot=True)

# ###SET THE CNN OPTIONS###


class CNN:
    def __init__(self,
                 n_outputs = 10,
                 image_x = 28,
                 image_y=28,
                 display_step=10,
                 training_epochs=200,
                 batch_size=50,
                 learning_rate=1e-4,
                 output_directory='/Users/kamaloddindarchinimaragheh/git/cnn_explore/app/mnist_TB_logs',
                 ):
        self.image_shape = [-1, image_x, image_y, 1]
        self.n_outputs = n_outputs
        self.image_x = image_x
        self.image_y = image_y
        self.display_step = display_step
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_directory = output_directory

        self._build_cnn()

    def _build_cnn(self):
        print('Building the CNN.')

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

        with tf.name_scope('input_reshape'):
            x_reshaped = tf.reshape(self.x, self.image_shape)
            tf.summary.image('input', x_reshaped, 10)

        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', self.keep_prob)

        # First conv+pool layer
        with tf.name_scope('conv1'):
            with tf.name_scope('weights'):
                W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
                with tf.name_scope('summaries'):
                    mean = tf.reduce_mean(W_conv1)
                    tf.summary.scalar('mean', mean)
                    with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv1 - mean)))
                    tf.summary.scalar('stddev', stddev)
                    tf.summary.scalar('max', tf.reduce_max(W_conv1))
                    tf.summary.scalar('min', tf.reduce_min(W_conv1))
                    tf.summary.histogram('histogram', W_conv1)

            with tf.name_scope('biases'):
                b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
                with tf.name_scope('summaries'):
                    mean = tf.reduce_mean(b_conv1)
                    tf.summary.scalar('mean', mean)
                    with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv1 - mean)))
                    tf.summary.scalar('stddev', stddev)
                    tf.summary.scalar('max', tf.reduce_max(b_conv1))
                    tf.summary.scalar('min', tf.reduce_min(b_conv1))
                    tf.summary.histogram('histogram', b_conv1)
            with tf.name_scope('Wx_plus_b'):
                preactivated1 = tf.nn.conv2d(x_reshaped, W_conv1,
                                             strides=[1, 1, 1, 1],
                                             padding='SAME') + b_conv1
                h_conv1 = tf.nn.relu(preactivated1)
                tf.summary.histogram('pre_activations', preactivated1)
                tf.summary.histogram('activations', h_conv1)
            with tf.name_scope('max_pool'):
                h_pool1 = tf.nn.max_pool(h_conv1,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
            # save output of conv layer to TensorBoard - first 16 filters
            with tf.name_scope('Image_output_conv1'):
                image = h_conv1[0:1, :, :, 0:16]
                image = tf.transpose(image, perm=[3, 1, 2, 0])
                tf.summary.image('Image_output_conv1', image)
            # save a visual representation of weights to TensorBoard
        with tf.name_scope('Visualise_weights_conv1'):
            # We concatenate the filters into one image of row size 8 images
            W_a = W_conv1  # i.e. [5, 5, 1, 32]
            W_b = tf.split(W_a, 32, 3)  # i.e. [32, 5, 5, 1, 1]
            rows = []
            for i in range(int(32 / 8)):
                x1 = i * 8
                x2 = (i + 1) * 8
                row = tf.concat(W_b[x1:x2], 0)
                rows.append(row)
            W_c = tf.concat(rows, 1)
            c_shape = W_c.get_shape().as_list()
            W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1])
            tf.summary.image("Visualize_kernels_conv1", W_d, 1024)

        # Second conv+pool layer
        with tf.name_scope('conv2'):
            with tf.name_scope('weights'):
                W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
                with tf.name_scope('summaries'):
                    mean = tf.reduce_mean(W_conv2)
                    tf.summary.scalar('mean', mean)
                    with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv2 - mean)))
                    tf.summary.scalar('stddev', stddev)
                    tf.summary.scalar('max', tf.reduce_max(W_conv2))
                    tf.summary.scalar('min', tf.reduce_min(W_conv2))
                    tf.summary.histogram('histogram', W_conv2)

            with tf.name_scope('biases'):
                b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
                with tf.name_scope('summaries'):
                    mean = tf.reduce_mean(b_conv2)
                    tf.summary.scalar('mean', mean)
                    with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv2 - mean)))
                    tf.summary.scalar('stddev', stddev)
                    tf.summary.scalar('max', tf.reduce_max(b_conv2))
                    tf.summary.scalar('min', tf.reduce_min(b_conv2))
                    tf.summary.histogram('histogram', b_conv2)
            with tf.name_scope('Wx_plus_b'):
                preactivated2 = tf.nn.conv2d(h_pool1, W_conv2,
                                             strides=[1, 1, 1, 1],
                                             padding='SAME') + b_conv2
                h_conv2 = tf.nn.relu(preactivated2)
                tf.summary.histogram('pre_activations', preactivated2)
                tf.summary.histogram('activations', h_conv2)
            with tf.name_scope('max_pool'):
                h_pool2 = tf.nn.max_pool(h_conv2,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
            # save output of conv layer to TensorBoard - first 16 filters
            with tf.name_scope('Image_output_conv2'):
                image = h_conv2[0:1, :, :, 0:16]
                image = tf.transpose(image, perm=[3, 1, 2, 0])
                tf.summary.image('Image_output_conv2', image)
            # save a visual representation of weights to TensorBoard
        with tf.name_scope('Visualise_weights_conv2'):
            # We concatenate the filters into one image of row size 8 images
            W_a = W_conv2
            W_b = tf.split(W_a, 64, 3)
            rows = []
            for i in range(int(64 / 8)):
                x1 = i * 8
                x2 = (i + 1) * 8
                row = tf.concat(W_b[x1:x2], 0)
                rows.append(row)
            W_c = tf.concat(rows, 1)
            c_shape = W_c.get_shape().as_list()
            W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1])
            tf.summary.image("Visualize_kernels_conv2", W_d, 1024)

        # Fully connected layer
        with tf.name_scope('Fully_Connected'):
            W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))  # 28/(2*2) = 7
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
            # Flatten the output of the second pool layer
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            # Dropout
            # keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=1.0)

        # Readout layer
        with tf.name_scope('Readout_Layer'):
            W_fc2 = tf.Variable(tf.truncated_normal([1024, self.n_outputs], stddev=0.1))
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.n_outputs]))
        # CNN output
        with tf.name_scope('Final_matmul'):
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Cross entropy functions
        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                           logits=y_conv)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

        # Optimiser
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        # Accuracy checks
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_conv, 1),
                                              tf.argmax(self.y_, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        print('CNN successfully built.')

    def _setup_folders(self):
        # Merge all the summaries and write them out to "mnist_logs"
        self.merged = tf.summary.merge_all()
        if not os.path.exists(self.output_directory):
            print('\nOutput directory does not exist - creating...')
            os.makedirs(self.output_directory)
            os.makedirs(self.output_directory + '/train')
            os.makedirs(self.output_directory + '/test')
            print('Output directory created.')
        else:
            print('\nOutput directory already exists - overwriting...')
            shutil.rmtree(self.output_directory, ignore_errors=True)
            os.makedirs(self.output_directory)
            os.makedirs(self.output_directory + '/train')
            os.makedirs(self.output_directory + '/test')
            print('Output directory overwitten.')

    def train(self):

        # ###START SESSSION AND COMMENCE TRAINING###

        # create session
        sess = tf.Session()
        # initalise variables
        sess.run(tf.global_variables_initializer())

        self._setup_folders()

        # prepare log writers
        train_writer = tf.summary.FileWriter(self.output_directory + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(self.output_directory + '/test')
        roc_writer = tf.summary.FileWriter(self.output_directory)

        # prepare checkpoint writer
        saver = tf.train.Saver()

        # Train
        print('\nTraining phase initiated.\n')
        for i in range(1, self.training_epochs + 1):
            batch_img, batch_lbl = mnist.train.next_batch(self.batch_size)
            testbatch = mnist.test.next_batch(self.batch_size)

            # run training step
            sess.run(self.train_step, feed_dict={self.x: batch_img,
                                            self.y_: batch_lbl,
                                            self.keep_prob: 1.0})

            # output the data into TensorBoard summaries every 10 steps
            if (i) % self.display_step == 0:
                train_summary, train_accuracy = sess.run([self.merged, self.accuracy], feed_dict={self.x: batch_img,
                                                                                        self.y_: batch_lbl,
                                                                                        self.keep_prob: 1.0})
                train_writer.add_summary(train_summary, i)
                print("step %d, training accuracy %g" % (i, train_accuracy))

                test_summary, test_accuracy = sess.run([self.merged, self.accuracy], feed_dict={self.x: testbatch[0],
                                                                                      self.y_: testbatch[1],
                                                                                      self.keep_prob: 0.9})
                test_writer.add_summary(test_summary, i)
                print("test accuracy %g" % test_accuracy)
            # output metadata every 100 epochs
            if i % 100 == 0 or i == self.training_epochs:
                print('\nAdding run metadata for epoch ' + str(i) + '\n')
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([self.merged, self.train_step],
                                      feed_dict={self.x: batch_img, self.y_: batch_lbl, self.keep_prob: 1.0},
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
            # save checkpoint at the end
            if i == self.training_epochs:
                print('\nSaving model at ' + str(i) + ' epochs.')
                saver.save(sess, self.output_directory + "/model_at_" + str(i) + "_epochs.ckpt",
                           global_step=i)

        # close writers
        train_writer.close()
        test_writer.close()

        # Evaluate model
        print('\nEvaluating final accuracy of the model (1/3)')
        train_accuracy = sess.run(self.accuracy, feed_dict={self.x: mnist.test.images,
                                                       self.y_: mnist.test.labels,
                                                       self.keep_prob: 1.0})
        print('Evaluating final accuracy of the model (2/3)')
        test_accuracy = sess.run(self.accuracy, feed_dict={self.x: mnist.train.images,
                                                      self.y_: mnist.train.labels,
                                                      self.keep_prob: 1.0})
        # print('Evaluating final accuracy of the model (3/3)')
        # val_accuracy = sess.run(self.accuracy, feed_dict={self.x: mnist.validation.images,
        #                                              self.y_: mnist.validation.labels,
        #                                              self.keep_prob: 1.0})

        # Output results
        print("\nTraining phase finished")
        print("\tAccuracy for the train set examples      = ", train_accuracy)
        print("\tAccuracy for the test set examples       = ", test_accuracy)
        # print("\tAccuracy for the validation set examples = ", val_accuracy)

        # print a statement to indicate where to find TensorBoard logs
        print('\nRun "tensorboard --logdir=' + self.output_directory + '" to see results on localhost:6006')