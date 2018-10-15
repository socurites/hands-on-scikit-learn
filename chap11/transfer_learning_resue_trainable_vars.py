import tensorflow as tf
import numpy as np

saver = tf.train.import_meta_graph("/tmp/tf-model/my_model_final.ckpt.meta")

for op in tf.get_default_graph().get_operations():
    print(op.name)

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

hidden2 = tf.get_default_graph().get_tensor_by_name("dnn/hidden2/Relu:0")

n_hidden3 = 20  # new layer 3
n_outputs = 10  # new layer 3
new_hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="new_hidden3")
new_logits = tf.layers.dense(new_hidden3, n_outputs, name="new_outputs")


with tf.name_scope("new_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("new_eval"):
    correct = tf.nn.in_top_k(new_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01
with tf.name_scope("new_train"):
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="new_hidden[3]|new_outputs")

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss, var_list=train_vars)

init = tf.global_variables_initializer()
new_saver = tf.train.Saver()


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


n_epochs = 40
batch_size = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


with tf.Session() as sess:
    init.run()
    saver.restore(sess, "/tmp/tf-model/my_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = new_saver.save(sess, "/tmp/tf-model/my_new_model_final.ckpt")
