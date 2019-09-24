import numpy as np 
import math
import matplotlib.pyplot as plt 

class kjl(project):
    def __init__(self,data):
        self.data = data  
        self.lens = len(data)
        self.tree = [None for _ in range(self.lens)]

    def Tree(self):
        self.List = [None for _ in range(self.lens)]
        self.tmpList = sorted(self.data,key=lambda x : x[0])
        self.half = math.floor(len(tmpList)/2)
        self.left = tmpList[:self.half]
        self.right = tmpList[self.half:]




    



kjl([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])