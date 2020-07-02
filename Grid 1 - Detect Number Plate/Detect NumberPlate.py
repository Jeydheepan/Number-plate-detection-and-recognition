import cv2
import os
import glob
import numpy as np
import imutils
from PIL import Image
import pytesseract

img=cv2.imread('E:/Miniproject2/Try/Grid 0 - Data Collection/Car Images/18.jpg')

cv2.imshow('Original Image',img)
cv2.waitKey(0)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image',gray)
cv2.waitKey(0)

gray = cv2.bilateralFilter(gray , 11, 17,17)
cv2.imshow('Sharpen removed',gray)
cv2.waitKey(0)

edged = cv2.Canny(gray, 170, 200)
cv2.imshow('Canny',edged)
cv2.waitKey(0)

img1=img.copy()

contours,hierarchy=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30]
plotted=cv2.drawContours(img1, contours , -1 ,(0,255,0) , 2)
cv2.imshow('Contours',img1)
cv2.waitKey(0)

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
