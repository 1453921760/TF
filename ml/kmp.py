class KMP():
    def __init__(self,des_string,sour_string):
        self.des_string = des_string 
        self.sour_string = sour_string 

    def fun_next(self):
        self.lens = len(self.sour_string)
        self.next = []
        
        
        for i in range(self.lens):
            total = 0
            for j in range(i):
                if(self.sour_string[0:j+1]==self.sour_string[i-j:i+1]):
                    total = j+1
                if j == i-1:
                    break
            
            self.next.append(total)
        print(self.next)

    def find(self):
        self.fun_next()
        i = 0
        while i < len(self.des_string):
            x0 = 0
            for j in range(len(self.sour_string)):
                if(self.des_string[i+x0] == self.sour_string[j]):
                    x0 = x0+1
                else:
                    if x0 == 0:
                        i = i+1
                        break 
                    else:
                        i =i + x0 - self.next[x0] 
                    break 
                if(x0 == len(self.sour_string)):
                    return i 
                if(i+len(self.sour_string)>=len(self.des_string)):
                    return -1
            
                

    
a = KMP("aaadafdasfBackground: python 3/win10d","pyth")
print(a.find())
print("aaadafdasfBackground: python 3/win10d".find("pyth"))
