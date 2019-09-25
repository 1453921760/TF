import numpy as np 
import math
import matplotlib.pyplot as plt 
from Tree import Tree 
import copy

# class kjl(project):
#     def __init__(self,data):
#         self.data = data  
#         self.lens = len(data)
#         self.tree = [None for _ in range(self.lens)]

#     def Tree(self):
#         self.List = [None for _ in range(self.lens)]
#         self.tmpList = sorted(self.data,key=lambda x : x[0])
#         self.half = math.floor(len(tmpList)/2)
#         self.left = tmpList[:self.half]
#         self.right = tmpList[self.half:]
# a = Tree.Tree([1,2,3,4,5,6,7])
# a.qx()




def fun():
    #data = [[1,2],[2,5],[3,4],[5,6],[7,8],[8,2],[9,2],[10,3]]
    data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    dz = []
    tmpd = [data,0]
    tmptmp = []
    
    while True:
        tmptmp = []
        
        #print(tmpd)
        if(tmpd[-1]==0):
            tmpd = tmpd[:len(tmpd)-1]
            for x in tmpd:
                #print(x)
                x = sorted(x,key=lambda y:y[0])
                half = math.floor(len(x)/2)
                dz.append(x[half])
                if(x[:half]!=[]):
                    tmptmp.append(x[:half])
               
                if (x[half+1:]!=[]):
                    tmptmp.append(x[half+1:])
            tmpd = copy.deepcopy(tmptmp)
            #print(tmptmp)
            tmpd.append(1)
            #print(dz)
            if 0 == len(tmptmp):
                break

            continue
            #print(tmpd)
        if(tmpd[-1]==1):
            for x in tmpd[:len(tmpd)-1]:
                x = sorted(x,key=lambda y:y[1])
                half = math.floor(len(x)/2)
                half = math.floor(len(x)/2)
                dz.append(x[half])
                if(x[:half]!=[]):
                    tmptmp.append(x[:half])
                if (x[half+1:]!=[]):
                    tmptmp.append(x[half+1:])
            tmpd = copy.deepcopy(tmptmp)
            #print(tmptmp)
            tmpd.append(1)  
            if 0 == len(tmptmp):
                break
    return dz

print(fun())
             

            












    



#kjl([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])