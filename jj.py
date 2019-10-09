import tensorflow as tf 
import numpy 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def biases_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = biases_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = biases_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = biases_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = biases_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

saver = tf.train.Saver()

training_epochs = 15
batch_size = 100                                #一次训练batch_size项数据
display_step = 1
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    # for epoch in range(training_epochs):
    #     avg_cost = 0.0
    #     total_batch = int(mnist.train.num_examples/batch_size)
    #     for i in range(total_batch):
    #         batch_xs,batch_ys = mnist.train.next_batch(batch_size)
    #         _,c = sess.run([train_step,cross_entropy],feed_dict \
    #         ={x:batch_xs,y_:batch_ys,keep_prob:0.9})
    #         avg_cost += c/total_batch
            
    #     if(epoch+1)%display_step == 0:
    #         print("Epoch:",'%04d'%(epoch + 1),\
    #             "accuracy=","{:.9f}".format(sess.run(cross_entropy,feed_dict={x:mnist.test.images[0],y_:mnist.test.labels[0],keep_prob:1.0})))
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = sess.run(accuracy,feed_dict={
                x:batch[0],y_ : batch[1],keep_prob:1.0
            })
            print(train_accuracy)
        sess.run(train_step,feed_dict={x:batch[0],y_ : batch[1],keep_prob:1.0}) 
    
    print(sess.run(accuracy,feed_dict={x: mnist.test.images[:1000], y_: mnist.test.labels[:1000], keep_prob: 1.0}))   


    saver.save(sess,'jjSaver/jj.cpkt')