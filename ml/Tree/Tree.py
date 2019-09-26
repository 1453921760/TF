import numpy as np 
import math 
class Node(object):
    def __init__(self,data):
        self.left = None 
        self.right = None 
        self.value = data 
        
class SNode(object):
    def __init__(self,x):
        self.node = x 
        self.tag = 1


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
    
    def zx(self):
        stack = [SNode(self.node)]
        top = 0
        a = SNode(self.node)
        
        while True:
            while(a.node.left!=None):
                top=top+1
                a = SNode(a.node.left)
                stack.append(a)
               # print(a.node.value) 
                
            if(top!=-1 and stack[top].tag == 2):
                top = top - 1
                stack.pop()
            
            if(top!=-1 and stack[top].tag==1):
                if stack[top].node.right!=None:
                    print(stack[top].node.value)
                    stack[top].tag=2
                    a = SNode(stack[top].node.right)
                    stack.append(a)
                    top = top+1
                    #print("gg")
                    #print(a.node.value)
                else :
                    stack[top].tag=2
                    print(stack[top].node.value)
            
            if top==-1:
                break



# a = Tree([1,2,3,4,5,6,7])

# a.qx()




                
    
    
