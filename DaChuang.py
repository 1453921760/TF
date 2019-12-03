import cv2 
from matplotlib import pyplot as plt 
import numpy as np 
import itertools
#np.set_printoptions(threshold='nan-NAN')
def panduan(point,pic):
    point_x = point[0][1]
    point_y = point[0][0]
    #print(point_x,point_y)
    x_min  = max([0,point_x - 10])
    x_max = min([480,point_x + 10])
    y_min = max([0,point_y-10])
    y_max = min([640,point_y+10])
   
    for i in range(x_min,x_max):
        for j in range(y_min,y_max):
            if pic[i][j] >0:
                return True 
        # print("gg")

    return False 



def standard_led(point_dict_1):
    for i in point_dict_1:
        if (max([len(list(v)) for k,v in itertools.groupby(point_dict_1[i])]))>10:
            return list(point_dict_1[i])

def standard_all_led(point_dict_1,standardLed):           #此处要有个异常抛出
    if standardLed!=None:
        for i in standardLed:
            if i == 0:
                continue
            else:
                for j in point_dict_1:
                    point_dict_1[j][i] = (point_dict_1[j][i]+1)%2      #point_dict_1[j][i] = ~point_dict_1[j][i]
        return point_dict_1 
    else:
        return list(point_dict_1[0])
def get_id(point_dict_1):
    id_dict = {}
    for i in point_dict_1:
        stat = 0
        for j in point_dict_1[i]:
            if j == 0:
                stat += 1
            else:
                stat = 0
            if stat == 5:
                break 
        if j!= len(point_dict_1[i]):
            id_dict[i] = [point_dict_1[i][j+1],point_dict_1[i][j+2],point_dict_1[i][j+3]]
    return id_dict 




cap = cv2.VideoCapture(0)
n = 0
print(cap.get(cv2.CAP_PROP_EXPOSURE))
cap.set(cv2.CAP_PROP_EXPOSURE,-10)
print(cap.get(cv2.CAP_PROP_EXPOSURE))
# print(cap.get(cv2.CAP_PROP_FPS))
# cap.set(cv2.CAP_PROP_FPS,120)
# print(cap.get(cv2.CAP_PROP_FPS))
pic_list = []

while n<30:
    ret,frame = cap.read()          #获取100贞图片进行分析
    pic_list.append(frame)
    n = n + 1
cap.release()      #释放摄像头
ll = []
point_list = []
i = 1
for pic in pic_list:
    
    #plt.ion()                                       #连续输出图片
    xulie = []
    gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)    # 转换为灰度图片
    
    result = cv2.blur(gray,(5,5))                     #模糊处理
    
    ret,result = cv2.threshold(result,200,255,cv2.THRESH_BINARY) #二值化
    

    #print(result)
    arr,x = cv2.findContours(result,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #img = cv2.drawContours(gray,arr,-1,(0,0,255),3)           #显示边框
    arr_list = []
    for y in arr:
        rect = cv2.minAreaRect(y)
        box = np.int0(cv2.boxPoints(rect))
        arr_list.append(box)
        #print(arr_list)
    
    #img = cv2.drawContours(gray,arr_list,-1,(0,0,255),3)

    """有用"""
    # for y in arr_list:
    #     point_x = int((y[0][0] + y[1][0] + y[2][0] + y[3][0])/4)
    #     point_y = int((y[0][1] + y[1][1] + y[2][1] + y[3][1])/4)
    #     point = np.asarray([[point_x,point_y]])
    #     print(point)
    #     point_list.append(point)
    #     #print(arr)
    #     ll.append(i)
    for y in arr_list:
        point_x = int((y[0][0] + y[1][0] + y[2][0] + y[3][0])/4)
        point_y = int((y[0][1] + y[1][1] + y[2][1] + y[3][1])/4)
        point = np.asarray([[point_x,point_y]])
        n = 0
        length = len(point_list)
        ll.append(i)
        while n < length :
            #print(point_list)
            #print(length)
            if (point_x - point_list[n][0][0])**2 + (point_y - point_list[n][0][1])**2 <=200:
                break
            n += 1
        if n == length:
            point_list.append(point)
    
    i = i+1
    
    #cv2.imshow("",img)
    
    #cv2.waitKey(500)
    # plt.imshow(result)
    # plt.pause(0.1)
print(ll)
#cv2.destroyAllWindows()
print(point_list)

gray = 0
re = 0

point_dict = {}
point_dict_1 = {}
for i,tmp in enumerate(point_list):
    point_dict[i] = tmp 
    point_dict_1[i] = []

cap = cv2.VideoCapture(0)
pic_list = []
while n<50:
    ret,frame = cap.read()          #获取100贞图片进行分析
    pic_list.append(frame)
    n = n + 1
cap.release()      #释放摄像头
for pic in pic_list:
    
    gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)    # 转换为灰度图片
    
    result = cv2.blur(gray,(5,5))                     #模糊处理
    
    ret,result = cv2.threshold(result,127,255,cv2.THRESH_BINARY) #二值化
    re = result

    for i in point_dict:
        #print(point_dict[i])
        #print(panduan(point_dict[i],result),end=" ")
        if panduan(point_dict[i],result) == True:
            #print(point_dict_1[i])
            point_dict_1[i].append(1)
        else :
            #print(point_dict_1[i])
            point_dict_1[i].append(0)

print(point_dict_1)


# for x in range(40):
#     for y in range(40):
#         print(re[297+y-20][278+x-20],end=" ")

standardLED = standard_led(point_dict_1)
point_dict_1 = standard_all_led(point_dict_1,standardLED)
ID = get_id(point_dict_1)
print(ID)




plt.imshow(result)
plt.show()
    
