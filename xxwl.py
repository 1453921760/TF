import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from random import shuffle
import time

def generate(sample_size,mean,cov,diff,regression):
    num_classes = 3 
    sample_per_class = int(sample_size/2)
    x0 = np.random.multivariate_normal(mean,cov,sample_per_class)               #产生二维随机变量
    y0 = np.zeros(sample_per_class)                                             #用于标记作用（红or蓝） 用于检验
    for ci,d in enumerate(diff):
        x1 = np.random.multivariate_normal(mean+d,cov,sample_per_class)
        #print()
        y1 = (ci+1)*np.ones(sample_per_class)
        x0 = np.concatenate((x0,x1))
        y0 = np.concatenate((y0,y1))
    if regression==False:
        Y=[]
        #print(y0)
        for i in range(len(y0)):
            tmp = np.zeros([num_classes])
            
            tmp[int(y0[i])] = 1.0
            Y.append(tmp)
            
            
    
    tmp = list(zip(x0,Y))
    shuffle(tmp)
    x0,y0 = zip(*tmp)
    x0 = np.asarray(x0)
    y0 = np.asarray(y0)
    return x0,y0

np.random.seed(10)
num_classes=3
mean = np.random.randn(2)
cov = np.eye(2)
X,Y = generate(1000,mean,cov,[[3.0],[3.0,0]],False)
print(len(X))
aa = [np.argmax(l) for l in Y]
#print(X)
#print(Y)
colors = ['r' if l==0 else 'b' if l==1  else 'y'  for l in aa[:]]
plt.scatter(X[:,0],X[:,1],c=colors)
#plt.show()





def main():
    

    input_dim = 2
    lab_dim = num_classes
    input_features = tf.placeholder(tf.float32,[None,input_dim])
    input_lables = tf.placeholder(tf.float32,[None,lab_dim])

    W = tf.Variable(tf.random_normal([input_dim,lab_dim]))
    b = tf.Variable(tf.zeros([lab_dim]))
    with tf.device('/gpu:0'):
        output = tf.matmul(input_features,W)+b
        z = tf.nn.softmax(output)
        a1 = tf.argmax(z,axis = 1)
        b1 = tf.argmax(input_lables,1)

        cross_entropy = -(input_lables*tf.log(z)+(1-input_lables)*tf.log(1-z))
        ser = tf.square(a1-b1)
        loss = tf.reduce_mean(cross_entropy)
        err = tf.reduce_mean(ser)

        optimizer = tf.train.AdamOptimizer(0.04)
        train = optimizer.minimize(loss)
    saver = tf.train.Saver()
    maxEpochs = 25 
    minibatchSize = 25
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        #with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        for epoch in range(maxEpochs):
            sumerr = 0
            for i in range(np.int32(len(Y)/minibatchSize)):
                x1 = X[i*minibatchSize:(i+1)*minibatchSize,:]
                #print(x1)
                y1 = Y[i*minibatchSize:(i+1)*minibatchSize]
                
                _,lossval,outputval,errval = sess.run([train,loss,output,err],feed_dict={input_features:x1,input_lables:y1})
            sumerr = sumerr + errval
            #print("Epoch:","%04d"%(epoch+1),"cost=","{:.9f}".format(lossval),"err=",sumerr/minibatchSize)
            #print(sess.run(W))
        saver.save(sess,"saverTest/test.cpkt")


        x= np.linspace(-1,8,200)
        y=-x*(sess.run(W)[0][0]/sess.run(W[1][0]))-sess.run(b[0])/sess.run(W[1][0])
        plt.plot(x,y)
        y=-x*(sess.run(W)[0][1]/sess.run(W[1][1]))-sess.run(b[1])/sess.run(W[1][1])

        print(sess.run(W))
        plt.plot(x,y)
        y=-x*(sess.run(W)[0][2]/sess.run(W[1][2]))-sess.run(b[2])/sess.run(W[1][2])
        plt.plot(x,y)
        plt.legend()
        #plt.show()

a = time.time()

main()

print(time.time()-a)