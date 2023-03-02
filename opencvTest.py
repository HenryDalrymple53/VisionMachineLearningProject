import numpy as np
import cv2
import sys

#Initializing variables
sys.path.append("C/PythonProjects")
from NeuralNetwork import *
capture=cv2.VideoCapture(0)
r,f = capture.read()
height = f.shape[0]
width = f.shape[1]
compression_size = 28


#nn.returnWeights()
(centerX, centerY) = (width // 2, height // 2)
print('Image height: ', height)
print('Image width: ', width)
print('Center location: ', (centerY, centerX))
gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
gVal = gray[centerX,centerY] # Gets the Color at the middle of the screen, helps with calibrating in different lighting
print(gVal)
#Initializing NeuralNetwork
nn = NeuralNetwork([500])
#Training fnn
nn.train(30000,.01)

def box(contours):
    left = contours[0][0][0]
    right = contours[0][0][0]
    top = contours[0][0][1]
    bot = contours[0][0][1]
    for x in contours:  
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
    return left, right, top, bot


def translate(thresh): # Translates pixels from 0 or 255 to 0 or 1
    pixels = []
    max = max(thresh)
    for y in thresh:
        for x in y:
            if(x>=max-5): ## TODO: FIGURE OUT A WAY TO OPTIMIZE THIS VALUE
                pixels.append(1)
            else:
                pixels.append(0)
            
    return pixels

def findLargestContour(contour):
    largest = 0
    ind = 0
    for _ in contour: #For all of the contours, find the largest and get it's index
            
        if(cv2.contourArea(contour[ind]) > cv2.contourArea(contour[largest])):
            largest = ind
        ind+=1
    return largest

while True:
    
    ret,frame=capture.read()#Gets camera video
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)#Gray scales video
    gVal = gray[centerX,centerY]
    ret, thresh = cv2.threshold(gray,int(gVal-10),int(gVal+10),cv2.THRESH_BINARY) #Thresholds based off previous value given
    gaussian = cv2.GaussianBlur(thresh, (5, 5), 10)# Blurs the image, removing some small artefacts
                                                   # That may disrupt imaging
    #CV2 function find contours finds the edges of objects that it detects. in this case,
    #it is the blurred gray scale image
    contours,hierarchy =  cv2.findContours(gaussian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if(len(contours) > 0):
        
        largest = findLargestContour(contours)
        
        #Draws the largest contour
        #cv2.drawContours(frame, contours[largest], -1, (255,0,0),5)
        
        # For all the points for the largest contour, find the outermost bounds that the contour goes
        left, right, top, bot = box(contours[largest])
        
        #cv2.rectangle(frame,(left,bot),(right,top),(0,255,0),3)
        frame = frame[bot:top,left:right]#Resizes a frame to be the dimensions of the tracked object    
        threshRez = thresh[bot:top,left:right]#Binary image scaled to these dimensions

        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)#Gray scales video
        ret, sThresh = cv2.threshold(gray,int(gVal-20),int(gVal+20),cv2.THRESH_BINARY_INV)
        
        smallContour,smallHierarchy =  cv2.findContours(threshRez,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #print(len(smallContour))
        if(len(smallContour)>0):
            numLargest = findLargestContour(smallContour)
            nLeft,nRight,nTop,nBot = box(smallContour[numLargest])
            threshRez = sThresh[nBot:nTop,nLeft:nRight]
            
            
            if(len(threshRez)>=28): 
                 
                
                threshRez = cv2.resize(threshRez, (28,28))
                print(threshRez)
                cv2.imshow('final',threshRez)
                pixels = translate(threshRez)#Highlights the negative space in the image, in this case hopefully the number
                
                nn.updateInput(np.reshape(pixels,(len(pixels),1))) #Reshape and update input for NeuralNetwork
                nn.guessNum()
                
                
    #Exits camera when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    cv2.imshow('Filter',thresh)
   
capture.release()
cv2.destroyAllWindows()

