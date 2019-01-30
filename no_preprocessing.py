from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os, glob
import cv2

tf.logging.set_verbosity (tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # LFW Dataset images are 60 * 60 pixels, and have one color channel
    input_layer = tf.reshape (features["x"], [-1, 60, 60, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height (60 * 60).
    # Input Tensor Shape: [batch_size, 60, 60, 1]
    # Output Tensor Shape: [batch_size, 60, 60, 32]
    conv1 = tf.layers.conv2d (inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 60, 60, 32]
    # Output Tensor Shape: [batch_size, 30, 30, 32]
    pool1 = tf.layers.max_pooling2d (inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 30, 30, 32]
    # Output Tensor Shape: [batch_size, 30, 30, 64]
    conv2 = tf.layers.conv2d (inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 3x3 filter and stride of 2
    # Input Tensor Shape: [batch_size, 30, 30, 64]
    # Output Tensor Shape: [batch_size, 10, 10, 64]
    pool2 = tf.layers.max_pooling2d (inputs=conv2, pool_size=[3, 3], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 10, 10, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 256]
    pool2_flat = tf.reshape (pool2, [-1, 7 * 7 * 256])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 256]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense (inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout (inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    logits = tf.layers.dense (inputs=dropout, units=2)

    predictions = {  # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax (input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax (logits, name="softmax_tensor")}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec (mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot (indices=tf.cast (labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy (onehot_labels=onehot_labels, logits=logits)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer (learning_rate=0.0001)
        train_op = optimizer.minimize (loss=loss, global_step=tf.train.get_global_step ())
        return tf.estimator.EstimatorSpec (mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy (labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec (mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    train_face = [[0] * 3600] * 10001
    train_face = np.array (train_face)
    train_non_face = [[0] * 3600] * 10001
    train_non_face = np.array (train_non_face)
    test_face = [[0] * 3600] * 1001
    test_face = np.array (test_face)
    test_non_face = [[0] * 3600] * 1001
    test_non_face = np.array (test_non_face)
    # Load training and eval data
    train_path = '/Users/Sharvari/Documents/NCSU/training_model/face'
    os.chdir (train_path)
    f = 1
    for face in glob.glob ('*.jpg'):
        image = cv2.imread (face, 0)
        image_flatten = image.flatten ()
        train_face[f, :] = image_flatten
        f = f + 1
    non_train_path = '/Users/Sharvari/Documents/NCSU/training_model/nonface'
    os.chdir (non_train_path)
    f = 1
    for non_face in glob.glob ('*.jpg'):
        image = cv2.imread (non_face, 0)
        image_flatten = image.flatten ()
        train_non_face[f, :] = image_flatten
        f = f + 1
    train_labels_face = np.asarray (np.zeros (len (train_face)), dtype=np.int32)
    train_labels_non_face = np.asarray (np.ones (len (train_non_face)), dtype=np.int32)
    train_data = np.concatenate ((train_face, train_non_face), axis=0)
    train_data = np.float32 (train_data)
    train_labels = np.concatenate ((train_labels_face, train_labels_non_face), axis=0)

    test_path = "/Users/Sharvari/Documents/NCSU/testing_model/face"
    os.chdir (test_path)
    nf = 1
    for face in glob.glob ('*.jpg'):
        image = cv2.imread (face, 0)
        image_flatten = image.flatten ()
        test_face[nf, :] = image_flatten
        nf = nf + 1
    non_test_path = "/Users/Sharvari/Documents/NCSU/testing_model/nonface"
    os.chdir (non_test_path)
    nf = 1
    for non_face in glob.glob ('*.jpg'):
        image = cv2.imread (non_face, 0)
        image_flatten = image.flatten ()
        test_non_face[nf, :] = image_flatten
        nf = nf + 1
    test_labels_face = np.asarray (np.zeros (len (test_face)), dtype=np.int32)
    test_labels_non_face = np.asarray (np.ones (len (test_non_face)), dtype=np.int32)
    eval_data = np.concatenate ((test_face, test_non_face), axis=0)
    eval_data = np.float32 (eval_data)
    eval_labels = np.concatenate ((test_labels_face, test_labels_non_face), axis=0)

    # Create the Estimator
    face_classifier = tf.estimator.Estimator (model_fn=cnn_model_fn)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook (tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn (x={"x": train_data}, y=train_labels, batch_size=100,
                                                         num_epochs=500, shuffle=True)
    face_classifier.train (input_fn=train_input_fn, steps=10000, hooks=[logging_hook])
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn (x={"x": eval_data}, y=eval_labels, num_epochs=None,
                                                        shuffle=False)
    eval_results = face_classifier.evaluate (input_fn=eval_input_fn)
