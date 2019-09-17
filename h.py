import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
train_x = np.linspace(-1,1,100)
train_y = 2*train_x+np.random.rand(train_x.shape[0])*0.3
plt.plot(train_x,train_y,'ro')
plt.legend()
plt.show()