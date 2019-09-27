import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import cv2

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

# print("输入数据的shape",mnist.train.images.shape)

# import pylab 
im = mnist.train.images[1]
# im1 = mnist.train.images[2]
#print(im)
#im = im.reshape(28,28)
#im = im.reshape(-1,28)
# pylab.imshow(im)
# pylab.show()


tf.reset_default_graph()

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal(([784,10])))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x,W) + b)                #pred的shape为[None,10]，b会自动广播为[None,10]
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),\
    reduction_indices = 1))                                #cost为交叉熵,reduction_indices=1 表示对行向量求和

saver = tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(tf.matmul(x,W) + b,feed_dict={x:[im,im1]}))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)      #梯度下降最小化损失函数

training_epochs = 25
batch_size = 100                                #一次训练batch_size项数据
display_step = 1
with tf.Session() as sess:
    with tf.device("/gpu:0"):
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost],feed_dict \
                ={x:batch_xs,y:batch_ys})
                avg_cost += c/total_batch 
            #if(epoch+1)%display_step == 0:
                #print("Epoch:",'%04d'%(epoch + 1),"cost=","{:.9f}".format(avg_cost))
                
        
        saver.save(sess,"saver/mnist.cpkt")
        print("finished!")
        #print(sess.run(pred,feed_dict={x:[im]}))            #给出预测


