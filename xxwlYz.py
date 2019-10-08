import tensorflow as tf 

x = tf.constant([[1.0,1]])
W = tf.Variable(tf.random_normal([2,3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x,W)+b)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,"saverTest/test.cpkt")
    print(sess.run(y))