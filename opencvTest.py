from bz2 import compress
import numpy as np
import cv2
import multiprocessing
import sys
from scipy.special import expit

sys.path.append("C:\PythonProjects")
from NeuralNetwork import *

capture=cv2.VideoCapture(0)
cascade_classifier = cv2.CascadeClassifier("C:\\Users\\henry\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")
r,f = capture.read()
#height, width = f.shape[:2]
height = 480
width = 640
hiddenLayerLength = 4
compression_size = 28
nn = NeuralNetwork(10,4)

(centerX, centerY) = (width // 2, height // 2)

print('Image height: ', height)
print('Image width: ', width)
print('Center location: ', (centerY, centerX))
gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
gVal = gray[centerX,centerY]
print(gVal)

def translate(thresh):
    pixels = []
    for y in thresh:
        for x in y:
            pixels.append(x)
            
    return pixels

while True:
    
    ret,frame=capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray,int(gVal-10),int(gVal+10),cv2.THRESH_BINARY)
    gaussian = cv2.GaussianBlur(thresh, (5, 5), 10)
    
    contours,hierarchy =  cv2.findContours(gaussian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    largest = 0
    ind = 0
    
    

    if(len(contours) > 0):
        
        for x in contours:
            
            if(cv2.contourArea(contours[ind]) > cv2.contourArea(contours[largest])):
                largest = ind
            ind+=1
            
        
        cv2.drawContours(frame, contours[largest], -1, (255,0,0),5)
        left = contours[largest][0][0][0]
        right = contours[largest][0][0][0]
        top = contours[largest][0][0][1]
        bot = contours[largest][0][0][1]
        for x in contours[largest]:
            
            xPos = x[0][0]
            yPos = x[0][1]
            if(xPos > right):
                right = xPos
            if(xPos < left):
                left = xPos 
            if(yPos > top):
                top = yPos
            if(yPos < bot):
                bot = yPos
        cv2.rectangle(frame,(left,bot),(right,top),(0,255,0),3)
        frame = frame[bot:top,left:right]             #           bot    top   left  right
        threshRez = thresh[bot:top,left:right]
        dim = (compression_size,compression_size)
        frame = cv2.resize(frame,dim)
        threshRez = cv2.resize(threshRez,dim)
        pixels = translate(threshRez)
        nn.updateInput(pixels)
        #print(nn.feedforward())
        
        #q1 = multiprocessing.Process(target = translate, args = [(top+bot)/2,top,(right+left)/2,right,thresh]) #Top Right
        #q1.start()
        #q2 = multiprocessing.Process(target = translate, args = [(top+bot)/2,top,left,(right+left)/2]) #Top Left
       # q2.start()
       # q3 = multiprocessing.Process(target = translate, args = [bot,(top+bot)/2,left,(right+left)/2]) #Bot Left
       # q3.start()
       # q4 = multiprocessing.Process(target = translate, args = [bot,(top+bot)/2,(right+left)/2,right]) #Bot Right

       # q4.start()
       # q1.join()
       # q2.join()
       # q3.join()
       # q4.join()
        cv2.imshow('Color',threshRez)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    cv2.imshow('Filter',thresh)
   
capture.release()
cv2.destroyAllWindows()

#faces = cascade_classifier.detectMultiScale(image_grey, minSize=(30, 30))
 #for (x, y, w, h) in faces:q
    #    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)


