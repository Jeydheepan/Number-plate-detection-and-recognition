import cv2
import numpy as np
import operator
import os
import imutils

reqarea = 100
width = 20
height = 30

class plot():
    i = None
    boundingRect = None
    x = 0
    y = 0
    w = 0
    h = 0
    area = 0.0

    def calcxywh(self):
        x1,y1,w1,h1 = self.boundingRect
        self.x=x1
        self.y=y1
        self.w=w1
        self.h=h1

    def chkvalid(self):
        if self.area < reqarea:
            return False
        return True

        
TotalContours=[]
ValidContours=[]

classifications = np.loadtxt("classification.txt", np.float32)
flatimg = np.loadtxt("images.txt", np.float32)


classifications=classifications.reshape((classifications.size,1))

KNN = cv2.ml.KNearest_create()
KNN.train(flatimg, cv2.ml.ROW_SAMPLE, classifications)

img = cv2.imread("test1.png")
img =imutils.resize(img,width=400)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,2)
#edged = cv2.Canny(thresh , 170 ,200)
thresh1 = thresh.copy()

contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#count = 0
for i in contour:
    item = plot()
    item.i = i
    item.boundingRect = cv2.boundingRect(item.i)
    item.calcxywh()
    item.area = cv2.contourArea(item.i)
    TotalContours.append(item)

               
for j in TotalContours:
    if(j.chkvalid()): 
        ValidContours.append(j)


ValidContours.sort(key = operator.attrgetter("x"))
end = ""
print(len(ValidContours))

for item in ValidContours:
    cv2.rectangle(img,(item.x,item.y),(item.x+item.w,item.y+item.h),(0,255,0),2)
    crop = thresh[item.y:item.y+item.h, item.x:item.x+item.w]
##    print(item.y , item.x)
    newimg = cv2.resize(crop, (width,height))
    resize = newimg.reshape((1,width*height))
    resize = np.float32(resize)
    retval, results, neighbour, distance = KNN.findNearest(resize, k = 1)
    char = str(chr(int(results[0][0])))
    end = end + char
    #print(distance)
    

cv2.imshow("img",img)

print(end)
    


