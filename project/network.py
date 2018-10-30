import tensorflow as tf
#import pandas as pd
import numpy as np
import os
import numpy as np
import sys

def input_parser(img_path, label):
    """
    Parse input images and labels.
    Set grayscale for every image and resize to 240x320.
    Generate one-hot representations of labels

    :args:relative path to image and label
    :return:img_decoded
    :return:one_hot
    """
    one_hot = tf.one_hot(label, 3)
    img_file = tf.read_file(img_path)
    #img_file = tf.read_file("/data2/center6/" + img_path + ".jpg")
    img_decoded = tf.image.decode_jpeg(img_file, channels=1)
    img_decoded = tf.image.resize_images(img_decoded, [240, 320])
    return img_decoded, one_hot

def _parse_line(line):
    """
    Parse line from csv file.
    Extract frame id which is the image path name
    Extract the corresponding label

    :args: line - single csv row
    :return:path (or filename)
    :return:label
    """
    fields = tf.decode_csv(line, [[""], [0.0], [0]])
    features = dict(zip(["frame_id", "steering_angle", "label"],fields))
    path = features.pop('frame_id')
    label = features.pop('label')
    return path, label

def cnn(x, n_classes, keep_prob):
    """

    """
    xavier_init = tf.keras.initializers.he_normal()
    weights = {
            'W_conv1': tf.Variable(xavier_init([5, 5, 1, 32])),
            'W_conv2': tf.Variable(xavier_init([5, 5, 32, 32])),
            'W_conv3': tf.Variable(xavier_init([5, 5, 32, 32])),
            'W_conv4': tf.Variable(xavier_init([5, 5, 32, 32])),
            'W_conv5': tf.Variable(xavier_init([5, 5, 32, 32])),
            'W_conv6': tf.Variable(xavier_init([5, 5, 32, 32])),
            'W_fc': tf.Variable(xavier_init([30 * 40 * 32, 1024])),
            'out': tf.Variable(xavier_init([1024, n_classes]))
    }


    biases = {
            'b_conv1': tf.Variable(xavier_init([32])),
            'b_conv2': tf.Variable(xavier_init([32])),
            'b_conv3': tf.Variable(xavier_init([32])),
            'b_conv4': tf.Variable(xavier_init([32])),
            'b_conv5': tf.Variable(xavier_init([32])),
            'b_conv6': tf.Variable(xavier_init([32])),
            'b_fc': tf.Variable(xavier_init([1024])),
            'out': tf.Variable(xavier_init([n_classes]))
    }

    conv1 = tf.nn.leaky_relu(conv2d(x, weights['W_conv1']) +  biases['b_conv1'])
    tf.summary.image("conv1_filter1", tf.reshape(conv1[:,:,:,1], [-1, 240, 320, 1]))
    tf.summary.image("conv1_filter2", tf.reshape(conv1[:,:,:,2], [-1, 240, 320, 1]))
    tf.summary.image("conv1_filter3", tf.reshape(conv1[:,:,:,3], [-1, 240, 320, 1]))
    tf.summary.image("conv1_filter4", tf.reshape(conv1[:,:,:,4], [-1, 240, 320, 1]))
    tf.summary.image("conv1_filter5", tf.reshape(conv1[:,:,:,5], [-1, 240, 320, 1]))
    conv2 = tf.nn.leaky_relu(conv2d(conv1, weights['W_conv2']) +  biases['b_conv2'])
    #tf.summary.image("conv2_filter1", tf.reshape(conv2[:,:,:,1], [-1, 240, 320, 1]))
    #tf.summary.image("conv2_filter2", tf.reshape(conv2[:,:,:,2], [-1, 240, 320, 1]))
    #tf.summary.image("conv2_filter3", tf.reshape(conv2[:,:,:,3], [-1, 240, 320, 1]))
    #tf.summary.image("conv2_filter4", tf.reshape(conv2[:,:,:,4], [-1, 240, 320, 1]))
    #tf.summary.image("conv2_filter5", tf.reshape(conv2[:,:,:,5], [-1, 240, 320, 1]))
    conv2 = maxpool2d(conv1)

    conv3 = tf.nn.leaky_relu(conv2d(conv2, weights['W_conv3'] + biases['b_conv3']))
    tf.summary.image("conv3_filter1", tf.reshape(conv3[:,:,:,1], [-1, 120, 160, 1]))
    tf.summary.image("conv3_filter2", tf.reshape(conv3[:,:,:,2], [-1, 120, 160, 1]))
    tf.summary.image("conv3_filter3", tf.reshape(conv3[:,:,:,3], [-1, 120, 160, 1]))
    tf.summary.image("conv3_filter4", tf.reshape(conv3[:,:,:,4], [-1, 120, 160, 1]))
    tf.summary.image("conv3_filter5", tf.reshape(conv3[:,:,:,5], [-1, 120, 160, 1]))
    conv4 = tf.nn.leaky_relu(conv2d(conv3, weights['W_conv4'] + biases['b_conv4']))
    #tf.summary.image("conv4_filter1", tf.reshape(conv4[:,:,:,1], [-1, 240, 320, 1]))
    #tf.summary.image("conv4_filter2", tf.reshape(conv4[:,:,:,2], [-1, 240, 320, 1]))
    #tf.summary.image("conv4_filter3", tf.reshape(conv4[:,:,:,3], [-1, 240, 320, 1]))
    #tf.summary.image("conv4_filter4", tf.reshape(conv4[:,:,:,4], [-1, 240, 320, 1]))
    #tf.summary.image("conv4_filter5", tf.reshape(conv4[:,:,:,5], [-1, 240, 320, 1]))
    conv4 = maxpool2d(conv2)

    conv5 = tf.nn.leaky_relu(conv2d(conv4, weights['W_conv5'] + biases['b_conv5']))
    tf.summary.image("conv5_filter1", tf.reshape(conv5[:,:,:,1], [-1, 60, 80, 1]))
    tf.summary.image("conv5_filter2", tf.reshape(conv5[:,:,:,2], [-1, 60, 80, 1]))
    tf.summary.image("conv5_filter3", tf.reshape(conv5[:,:,:,3], [-1, 60, 80, 1]))
    tf.summary.image("conv5_filter4", tf.reshape(conv5[:,:,:,4], [-1, 60, 80, 1]))
    tf.summary.image("conv5_filter5", tf.reshape(conv5[:,:,:,5], [-1, 60, 80, 1]))
    conv6 = tf.nn.leaky_relu(conv2d(conv5, weights['W_conv6'] + biases['b_conv6']))
    #tf.summary.image("conv6_filter1", tf.reshape(conv6[:,:,:,1], [-1, 240, 320, 1]))
    #tf.summary.image("conv6_filter2", tf.reshape(conv6[:,:,:,2], [-1, 240, 320, 1]))
    #tf.summary.image("conv6_filter3", tf.reshape(conv6[:,:,:,3], [-1, 240, 320, 1]))
    #tf.summary.image("conv6_filter4", tf.reshape(conv6[:,:,:,4], [-1, 240, 320, 1]))
    #tf.summary.image("conv6_filter5", tf.reshape(conv6[:,:,:,5], [-1, 240, 320, 1]))
    conv6 = maxpool2d(conv5)


    fc = tf.reshape(conv6, [-1, 30 * 40 * 32])
    fc = tf.nn.leaky_relu(tf.matmul(fc, weights['W_fc'] + biases['b_fc']))
    fc = tf.nn.dropout(fc, keep_prob)
    output = tf.matmul(fc, weights['out'] + biases['out'])
    return output

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def train_neural_network(y, optimizer, cost):
    #_cost_train = []
    #_cost_validation = []
    #_accuracy_train = []
    #_accuracy_validation = []

    #_out_loss_train = open("loss_train.out","w")
    #_out_loss_validation = open("loss_validation.out","w")

    #_out_acc_train = open("acc_train.out","w")
    #_out_acc_validation = open("acc_validation.out","w")

    _out_final_accuracy = open("final_acc_test.out", "w")
    print("Training starts.")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        validation_writer = tf.summary.FileWriter('./logs/validation', sess.graph)
        for epoch in range(hm_epochs):
            sess.run(ds_train_iterator.initializer)
            #epoch_loss = 0
            #count = 0
            epoch_accuracy = 0
            while True:
                try:
                    elem = sess.run(train_next_element)
                    """
                    sample_x = []
                    sample_y = []
                    for i, e in enumerate(elem[1]):
                        if e[0] == 1:
                            if np.random.random_sample() > 0.5:
                                sample_x.append(elem[0][i])
                                sample_y.append(e)
                        elif e[1] == 1:
                            if np.random.random_sample() > 0.5:
                                sample_x.append(elem[0][i])
                                sample_y.append(e)
                        else:
                            sample_x.append(elem[0][i])
                            sample_y.append(e)
                    """
                    merge = tf.summary.merge_all()
                    _, c = sess.run([optimizer, cost], feed_dict={x: elem[0], y: elem[1], keep_prob: 0.8})
                    _, epoch_accuracy, summary, mean_err = sess.run([avg_acc, avg_acc_update, merge, mean_error_update], feed_dict={x: sample_x, y: sample_y, keep_prob: 0.8})
                    train_writer.add_summary(summary, epoch)
                    #epoch_loss += c
                    #count += 1
                    print("loss: ", mean_err)
                    #print(epoch_loss)
                except tf.errors.OutOfRangeError:
                    s = 5
                    epoch_validation_accuracy = 0
                    #validation_cost = 0
                    sess.run(ds_validation_iterator.initializer)
                    for i in range(s):
                        try:
                            elem_validation = sess.run(validation_next_element)
                            merge = tf.summary.merge_all()
                            _, c = sess.run([optimizer, cost], feed_dict={x: elem_validation[0], y: elem_validation[1], keep_prob: 1.0})
                            validation_cost += c
                            _, epoch_validation_accuracy, summary, mean_err = sess.run([avg_acc, avg_acc_update, merge, mean_error_update], feed_dict={x: elem_validation[0], y: elem_validation[1], keep_prob: 1.0})
                            predictions = prediction.eval(feed_dict = {x: elem_validation[0], keep_prob: 1.0})
                            validation_writer.add_summary(summary, epoch)
                            if i == 0:
                                print(predictions)
                            #correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(elem_test[1], 1))
                            #acc = tf.reduce_mean(tf.cast(correct, 'float'))
                            #accuracy += acc.eval({x: elem_test[0], y: elem_test[1], keep_prob: 1.0})
                        except tf.errors.OutOfRangeError:
                            print('ERROR Accuracy:',accuracy/s)
                    print("End of epoch.")
                    print("Train accuracy; ", epoch_accuracy)
                    print("validation accuracy: ", epoch_validation_accuracy)


                    #_accuracy_train.append(epoch_accuracy)
                    #_accuracy_validation.append(epoch_validation_accuracy)

                    #_cost_train.append(epoch_loss/count)
                    #_cost_validation.append(validation_cost/s)

                    #_out_loss_train.write(str(epoch_loss) + "\n")
                    #_out_loss_validation.write(str(validation_cost/s) + "\n")

                    #_out_acc_train.write(str(epoch_accuracy/count) + "\n")
                    #_out_acc_validation.write(str(epoch_validation_accuracy) + "\n")

                    break
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            """
            try:
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:',accuracy.eval({x: elem_test[0], y: elem_test[1], keep_prob: 1.0 }))
            except tf.errors.OutOfRangeError:
                print("failed to get test data.")
                break
            """
        s = 10
        epoch_test_accuracy = 0
        for i in range(s):
            try:
                sess.run(ds_test_iterator.initializer)
                elem_test = sess.run(test_next_element)
                _, epoch_test_accuracy = sess.run([avg_acc, avg_acc_update], feed_dict={x: elem_test[0], y: elem_test[1], keep_prob: 1.0})

            except tf.errors.OutOfRangeError:
                print("OUT OF RANGE")
        print("TEST ACCURACY: ", epoch_test_accuracy)
        _out_final_accuracy.write(str(epoch_test_accuracy) + "\n")

    #print(_cost_train)
    #print(_cost_validation)
    #print(_accuracy_train)
    #print(_accuracy_validation)

"""
:: START OF EXECUTION ::
"""

"""
Initilize phase

    batch_size: size of consecutive elements into (batching)
    n_classes : left = 0, center = 1, right = 2.

"""
batch_size = 256
n_classes = 3
image_width = 320
image_height = 240
hm_epochs = 5
print("Setting up")

ds_test = tf.data.TextLineDataset("data/test.csv").skip(1)
ds_test = ds_test.map(_parse_line)
ds_test = ds_test.map(input_parser)
ds_test = ds_test.batch(batch_size)
ds_test_iterator = ds_test.make_initializable_iterator();
test_next_element = ds_test_iterator.get_next()
print("test data setup completed")

ds_validation = tf.data.TextLineDataset("data/validation.csv").skip(1)
ds_validation = ds_validation.map(_parse_line)
ds_validation = ds_validation.map(input_parser)
ds_validation = ds_validation.batch(batch_size)
ds_validation_iterator = ds_validation.make_initializable_iterator();
validation_next_element = ds_validation_iterator.get_next()
print("Validation data setup completed")

ds_train = tf.data.TextLineDataset("data/train.csv").skip(1)
ds_train = ds_train.map(_parse_line)
ds_train = ds_train.map(input_parser)
ds_train = ds_train.shuffle(buffer_size=1024)
ds_train = ds_train.repeat(1)
ds_train = ds_train.batch(batch_size)
ds_train_iterator = ds_train.make_initializable_iterator();
train_next_element = ds_train_iterator.get_next()
print("Train data setup completed")



keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder("float", [None, 240, 320, 1])
y = tf.placeholder("float", [None, n_classes])

prediction = cnn(x, n_classes, keep_prob)

# Compute softmax cross entropy between logits and labels.
# Mesures the probability error in descrete classification tasks in which
# the classes are mutually exclusive
# https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2
out = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y)
tf.summary.histogram("Predictions Histogram", out)

# LEGACY CODE
#tf.summary.scalar("Predictions Scalar", out)
#correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#accs = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
tf.metrics.accuracy(labels, predictions):
Calculates how often predictions matches labels (y)

https://www.tensorflow.org/api_docs/python/tf/metrics/accuracy
"""
avg_acc, avg_acc_update = tf.metrics.accuracy(tf.argmax(y,1), tf.argmax(prediction, 1))

# LEGACY CODE
#tf.summary.histogram("Accuracy Histogram", update_)

"""
tf.summary.scalar(name, real numeric tensor):
Outputs a Summary protocol buffer containing a single scalar value

https://www.tensorflow.org/api_docs/python/tf/summary/scalar
"""
tf.summary.scalar("Accuracy Scalar", avg_acc)
#class_weights = tf.constant([0.8, 0.8, 1.0])
#out = tf.nn.weighted_cross_entropy_with_logits(logits=prediction, targets=y, pos_weight=class_weights)'
"""
tf.reduce_mean(input tensor)
Computes the mean of elements across dimensions of a tensor.

https://www.tensorflow.org/api_docs/python/tf/reduce_mean
"""
cost = tf.reduce_mean(out)
mean_error, mean_error_update = tf.metrics.mean_absolute_error(tf.argmax(y,1), tf.argmax(prediction, 1))
tf.summary.scalar("mean loss", mean_error)
tf.summary.scalar("Loss", cost)



"""
Optimizer

We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm.
"""
optimizer = tf.train.AdamOptimizer().minimize(cost)
train_neural_network(y, optimizer, cost)
