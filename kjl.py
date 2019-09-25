import numpy as np 
import math
import matplotlib.pyplot as plt 
from Tree import Tree #Tree.Tree 才是类
import copy


# def fun(data):
#     #data = [[1,2],[2,5],[3,4],[5,6],[7,8],[8,2],[9,2],[10,3],[2,3],[100,2],[44,33],[556,4],[34,5]]
#     #data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
#     dz = []                 #分类分叉口所有的点的集合
#     tmpd = [data,0]         #0是标记
#     tmptmp = []             #临时变量
    
#     while True:
#         tmptmp = []
        
#         #print(tmpd)
#         if(tmpd[-1]==0):                    #如果标记是0对x维进行分类
#             tmpd = tmpd[:len(tmpd)-1]
#             for x in tmpd:
#                 #print(x)
#                 x = sorted(x,key=lambda y:y[0])     #按照x轴进行排序
#                 half = math.floor(len(x)/2)         #对半
#                 dz.append(x[half])                  #取出中间点
#                 if(x[:half]!=[]):
#                     tmptmp.append(x[:half])         #将前半部分放入tmptmp
#                 if (x[half+1:]!=[]):                   
#                     tmptmp.append(x[half+1:])       #将后半部分放入tmptmp 不包括中间点
#             tmpd = copy.deepcopy(tmptmp)            #将tmpd更新
#             #print(tmptmp)
#             tmpd.append(1)
#             #print(dz)
#             if 0 == len(tmptmp):                    #如果tmptmp长度是零也就是说所有点都加入到dz了，退出循环
#                 break

#             continue
#             #print(tmpd)
#         if(tmpd[-1]==1):
#             for x in tmpd[:len(tmpd)-1]:
#                 x = sorted(x,key=lambda y:y[1])
#                 half = math.floor(len(x)/2)
#                 half = math.floor(len(x)/2)
#                 dz.append(x[half])
#                 if(x[:half]!=[]):
#                     tmptmp.append(x[:half])
#                 if (x[half+1:]!=[]):
#                     tmptmp.append(x[half+1:])
#             tmpd = copy.deepcopy(tmptmp)
#             #print(tmptmp)
#             tmpd.append(1)  
#             if 0 == len(tmptmp):
#                 break
#     return dz

#print(fun([[1,2],[2,5],[3,4],[5,6],[7,8],[8,2],[9,2],[10,3],[2,3],[100,2],[44,33],[556,4],[34,5]]))

             
class gg(object):
    def __init__(self,data):
        print(data)

class kjl(Tree.Tree):
    def __init__(self,data):
        data = self.fun(data)
        super(kjl,self).__init__(data)

    def fun(self,data):
        dz = []                 #分类分叉口所有的点的集合
        tmpd = [data,0]         #0是标记
        tmptmp = []             #临时变量
        
        while True:
            tmptmp = []
            
            #print(tmpd)
            if(tmpd[-1]==0):                    #如果标记是0对x维进行分类
                tmpd = tmpd[:len(tmpd)-1]
                for x in tmpd:
                    #print(x)
                    x = sorted(x,key=lambda y:y[0])     #按照x轴进行排序
                    half = math.floor(len(x)/2)         #对半
                    dz.append(x[half])                 #取出中间点
                    plt.plot([x[half][0],x[half][0]],[0,10])
                    if(x[:half]!=[]):
                        tmptmp.append(x[:half])         #将前半部分放入tmptmp
                    if (x[half+1:]!=[]):                   
                        tmptmp.append(x[half+1:])       #将后半部分放入tmptmp 不包括中间点
                tmpd = copy.deepcopy(tmptmp)            #将tmpd更新
                #print(tmptmp)
                tmpd.append(1)
                #print(dz)
                if 0 == len(tmptmp):                    #如果tmptmp长度是零也就是说所有点都加入到dz了，退出循环
                    break

                continue
                #print(tmpd)
            if(tmpd[-1]==1):
                for x in tmpd[:len(tmpd)-1]:
                    x = sorted(x,key=lambda y:y[1])
                    half = math.floor(len(x)/2)
                    half = math.floor(len(x)/2)
                    dz.append(x[half])
                    plt.plot([0,10],[x[half][1],x[half][1]])
                    if(x[:half]!=[]):
                        tmptmp.append(x[:half])
                    if (x[half+1:]!=[]):
                        tmptmp.append(x[half+1:])
                tmpd = copy.deepcopy(tmptmp)
                #print(tmptmp)
                tmpd.append(0)  
                if 0 == len(tmptmp):
                    break
        return dz
data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
a = kjl(data)
for i in data :
    plt.plot(i[0],i[1],'ro')
a.zx() 
# for i in bb:
#     plt.plot([i[0],i[0]],[i[1]-5,i[1]+5])

plt.show()

#kjl([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])