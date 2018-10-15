import tensorflow as tf
import numpy as np

"""
1. Define a graph
"""
# Defin a model
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
batch_norm_momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope("dnn"):
    he_init = tf.variance_scaling_initializer()

    hidden1 = tf.layers.dense(X, n_hidden1, kernel_initializer=he_init, name="hidden1")
    bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=batch_norm_momentum)
    bn1_act = tf.nn.elu(bn1)

    hidden2 = tf.layers.dense(bn1_act, n_hidden2, kernel_initializer=he_init, name="hidden2")
    bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=batch_norm_momentum)
    bn2_act = tf.nn.elu(bn2)

    logits_before_bn = tf.layers.dense(bn2_act, n_outputs, kernel_initializer=he_init, name="outputs")
    logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=batch_norm_momentum)


# Define a cost function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


# Define a optimizer
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)


# Definane a metric
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


"""
2. Train a model
"""
# Load datasets
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity("INFO")

mnist = input_data.read_data_sets("/tmp/data/")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_epochs = 20
batch_size = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run([training_op, extra_update_ops], feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Train accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    save_path = saver.save(sess, "/tmp/tf-model/my_model_final.ckpt")