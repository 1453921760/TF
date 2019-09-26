import tensorflow as tf 
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable([1.0])
z = X+Y*W 
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(50):
        sess.run(tf.train.GradientDescentOptimizer(0.01).minimize(tf.square(z)),feed_dict={X:2,Y:2})
    sess.run(tf.train.GradientDescentOptimizer(0.01).minimize(tf.square(z)),feed_dict={X:2,Y:2})
    print(sess.run(W))
    #print(sess.run(z,feed_dict={X:2,Y:2}))
