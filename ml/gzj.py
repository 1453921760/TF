import numpy as np 
import matplotlib.pyplot as plt 

class GZ(object):
    def __init__(self,data):
        self.data = data
        n = len(data[0][0])
        self.data = data
        self.w = np.zeros(n)
        self.b = 0

    def func(self,x,yi):
        self.w = self.w + yi*np.asarray(x) 
        self.b = self.b + yi
        # print(self.w)
        # print(self.b)

    def h(self):
        tmp = 0
        for n in range(300):
            if (np.inner(self.data[tmp][0],self.w) + self.b)*self.data[tmp][1] <=0:
                self.func(self.data[tmp][0],self.data[tmp][1])
                tmp = 0
            else :
                tmp = tmp + 1
            if tmp == len(data):
                break
        return (self.w,self.b)

    def show(self):
        self.xx,self.yy = tuple(self.h()[0])
        self.b = self.h()[1]
        self.x = np.arange(-10,10,2)
        self.y = (-self.x*self.xx-self.b)/self.yy 
        self.data1 = [i[0] for i in self.data if i[1] == 1]
        self.datax1 = [i[0] for i in self.data1]
        self.datay1 = [i[1] for i in self.data1]
        self.data2 = [i[0] for i in self.data if i[1] == -1]
        self.datax2 = [i[0] for i in self.data2]
        self.datay2 = [i[1] for i in self.data2]
        plt.plot(self.datax1,self.datay1,'ro')
        plt.plot(self.datax2,self.datay2,'bo')
        plt.plot(self.x,self.y)
        plt.plot()
        plt.show()
        


        

data = [[[3,3],1],[[4,3],1],[[1,1],-1],[[2,2],-1],[[2.9,2.5],1]]
      
        
a = GZ(data)
a.show()
# print(a.h())
# xx,yy = tuple(a.h()[0])
# b = a.h()[1]
# # print(b)
# # print(x,y)
# data1 = [i[0] for i in data]
# datax = [i[0] for i in data1]
# datay = [i[1] for i in data1]
# x = np.arange(-5,5,0.2)
# y= (-x*xx -b )/yy
# print(x)
# plt.plot(datax,datay,'ro')
# plt.plot(x,y)
# plt.plot()
# plt.show()
