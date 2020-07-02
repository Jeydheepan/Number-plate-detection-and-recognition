from PIL import Image
import pytesseract
import imutils
import cv2

im = cv2.imread("E:/Miniproject2/Try/Grid 3 - Character Recogonition with OCR Tesseract/Plates/7.png")
#im =imutils.resize(im,width=500)
#cv2.imshow('plate',im)

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray',gray)

gray =cv2.bilateralFilter(gray , 11, 17, 17)
#cv2.imshow('Gray1',gray)

ret , thresh = cv2.threshold(gray,125,125,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow('plate1',thresh)

contours , hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
#print(len(contours))
curve = cv2.drawContours(im.copy() , contours , -1 , (0,255,0) , 2)
cv2.imshow('a',curve)




test= pytesseract.image_to_string(curve, lang="eng")
print("Numbet plate:",test)

cv2.waitKey(0)
cv2.destroyAllWindows()
