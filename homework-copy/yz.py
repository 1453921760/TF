import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
im = mnist.train.images[1]
img = im.reshape([28,28])

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal(([784,10])))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x,W) + b)
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,"saver/mnist.cpkt")
    print(sess.run(pred,feed_dict={x:[im]}))

plt.imshow(img)
plt.show()