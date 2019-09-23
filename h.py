import numpy as np 
import math 
class Node(object):
    def __init__(self,data):
        self.left = None 
        self.right = None 
        self.value = data 

class Tree(object):
    def __init__(self,data):
        self.List = [None for i in range(len(data))]
        self.node = None
        for i in range(len(data)):
            if i == 0:
                self.List[i] = Node(data[i])
                self.node = self.List[i]

            elif (i+1)%2 == 0:
                self.List[i] = Node(data[i])
                self.List[(i-1)//2].left = self.List[i]

            elif i%2 == 0 :
                self.List[i]= Node(data[i])
                self.List[i//2-1].right = self.List[i]

a = Tree([1,2,3])
print(a.List[0].left.value)


                
    
    
