import tensorflow as tf
import numpy as np

# Load test datasets
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_test = y_test.astype(np.int32)
X_test_sample = X_test[:20]


# Load a graph
saver = tf.train.import_meta_graph("/tmp/tf-model/my_model_final.ckpt.meta")
X = tf.get_default_graph().get_tensor_by_name("X:0")
logits = tf.get_default_graph().get_tensor_by_name("dnn/outputs/logits:0")

# Load a trained model
with tf.Session() as sess:
    saver.restore(sess, "/tmp/tf-model/my_model_final.ckpt")

    Z = logits.eval(feed_dict={X: X_test_sample})
    y_pred = np.argmax(Z, axis=1)

    print("Predicted:", y_pred)
    print("Actual:", y_test[:20])