import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
#import cv2

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)



learning_rate = 0.001

training_epochs = 25 
batch_size = 100 
display_step = 1 
n_hidden_1 = 256 
n_hidden_2 = 256 
n_input = 784 
n_classes = 10 

x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

def multilayer_perceptron(x,weights,biases):
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2,weights['out'])+biases['out']
    return out_layer

weights = {'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
            'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
            'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases = {'b1':tf.Variable(tf.random_normal([n_hidden_1])),
            'b2':tf.Variable(tf.random_normal([n_hidden_2])),
            'out':tf.Variable(tf.random_normal([n_classes]))
            }

pred = multilayer_perceptron(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver()

with tf.Session() as sess:
    with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
                avg_cost += c/total_batch
            saver.save(sess,"tmpSaver/tmp.cpkt")
            print("Epoch:",'%04d'%(epoch + 1),"cost=","{:.9f}".format(avg_cost))
