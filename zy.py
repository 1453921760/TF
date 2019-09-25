import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
train_x = np.linspace(-1,1,100)
train_y = 2*train_x + np.random.rand(100)*0.3
plt.plot(train_x,train_y,'ro')


X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.zeros([1]))
z = tf.multiply(X,W) + b 

cost = tf.reduce_mean(tf.square(Y-z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init =tf.global_variables_initializer()
training_epochs = 20 
display_step = 2
with tf.Session() as sess:
    sess.run(init)
    plotdata = {"batchsize":[],"loss":[]}
    for epoch in range(training_epochs):
        for x,y in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_x,Y:train_y})
            print("epoch:",epoch+1,"cost:",loss,"W:",sess.run(W),"b:",sess.run(b))
            if not (loss=="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print("Finished!")
    print("cost=",sess.run(cost,feed_dict={X:train_x,Y:train_y}),"W:",sess.run(W),"b:",sess.run(b))

#plt.plot(train_x,(W*train_x+b))

plt.show()