import cv2
import numpy as np
import sys
import imutils

image=cv2.imread("training_chars.png")
#image =imutils.resize(image,width=400)


if(image is None):
    print("error")

gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray , (5,5), 0)
thresh = cv2.adaptiveThreshold(blur , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV, 11, 2)

cv2.imshow("Threshold" , thresh)

threshCopy = thresh.copy()

contours , hierarchy = cv2.findContours(threshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

flatimg1 = np.empty((0, 20*30))

classification = []

validchar = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

#print(cv2.contourArea(contours[0]))
for i in contours:
    if(cv2.contourArea(i) > 100):
        x,y,w,h = cv2.boundingRect(i)
        #cv2.rectange(image,(x,y),(x+w,y+h),(0,0,255),2)

        crop=thresh[y:y+h , x:x+w]
        resize = cv2.resize(crop , (20,30))
        #edged = cv2.Canny(resize , 170 ,200)

        cv2.imshow("a", crop)
        cv2.imshow("b", resize)
        cv2.imshow("training_numbers.png", image) 

        press=cv2.waitKey(0)

        if(press == 27):
            sys.exit()
        elif press in validchar:
            classification.append(press)

            flatimg=resize.reshape((1,20*30))
            flatimg1 = np.append(flatimg1, flatimg , 0)

classification1 = np.array(classification , np.float32)
final_classification = classification1.reshape((classification1.size,1))

np.savetxt("classification.txt",final_classification)
np.savetxt("images.txt", flatimg1)

cv2.destroyAllWindows()
