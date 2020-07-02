import cv2
import os
import glob
import numpy as np
import imutils
from PIL import Image
import pytesseract

img=cv2.imread('E:/Miniproject2/Try/Grid 0 - Data Collection/Car Images/18.jpg')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray , 11, 17,17)

edged = cv2.Canny(gray, 170, 200)

img1=img.copy()

contours,hierarchy=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30]
plotted=cv2.drawContours(img1, contours , -1 ,(0,255,0) , 2)

print('Cropping contours')
index=0
path='E:/Miniproject2/Try/cropped images'
plate=None
for c in contours:
    peri = cv2.arcLength(c, True)
    edge = cv2.approxPolyDP(c, 0.02*peri , True)
    if(len(edge) == 4):
        plate=edge
        x,y,w,h = cv2.boundingRect(c)
        crop = gray[y:y+h , x:x+w]
        img2=img.copy()
        cv2.drawContours(img2, [plate], -1, (0,255,0), 3)
        cv2.imshow("Number plate Contour", img2)
        press = cv2.waitKey(0)
        if(press == 13):
            cv2.imwrite('cropped images/'+str(index)+ '.png', crop)
            index = index + 1
            break
        
final = cv2.imread('cropped images/0.png')
final =imutils.resize(final,width=500)
cv2.imshow("Cropped img", final)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Number plate Found!!!")


im = cv2.imread("E:/Miniproject2/Try/Grid 4 - Final Master/Cropped images/0.png")

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

gray =cv2.bilateralFilter(gray , 11, 17, 17)

ret , thresh = cv2.threshold(gray,125,125,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours , hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]

curve = cv2.drawContours(im.copy() , contours , -1 , (0,255,0) , 2)
cv2.imshow('a',curve)

test= pytesseract.image_to_string(curve, lang="eng")
print("Numbet plate:",test)

cv2.waitKey(0)
cv2.destroyAllWindows()

BadChars = ['!','@','#','$','%','^','&','*','(',')','/',',','.','|','[',']',';',':','{','}',' ',"'",'"']

for j in BadChars:
        test = test.replace(j,'')
print(test)
