import numpy as np
import cv2
import sys

#Initializing variables
sys.path.append("C/PythonProjects")
from betterNeuralNetwork import *
capture=cv2.VideoCapture(0)
r,f = capture.read()
height = 480
width = 640
compression_size = 28

data = pd.read_csv("C:\\PythonProjects\\DataFiles\\train.csv")
data = np.array(data)
#nn.returnWeights()hi
(centerX, centerY) = (width // 2, height // 2)
print('Image height: ', height)
print('Image width: ', width)
print('Center location: ', (centerY, centerX))
gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
gVal = gray[centerX,centerY] # Gets the Color at the middle of the screen, helps with calibrating in different lighting
print(gVal)
#Initializing NeuralNetwork
nn = NeuralNetwork([20,20])
#Training fnn
nn.train(40000,.25,1,10)


def translate(thresh): # Translates pixels from 0 or 255 to 0 or 1
    pixels = []
    for y in thresh:
        for x in y:
            if(x==255):
                pixels.append(0)
            else:
                pixels.append(1)
            
    return pixels

while True:
    
    ret,frame=capture.read()#Gets camera video
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)#Gray scales video
    ret, thresh = cv2.threshold(gray,int(gVal-10),int(gVal+10),cv2.THRESH_BINARY) #Thresholds based off previous value given
    gaussian = cv2.GaussianBlur(thresh, (5, 5), 10)# Blurs the image, removing some small artefacts
                                                   # That may disrupt imaging
    #CV2 function find contours finds the edges of objects that it detects. in this case,
    #it is the blurred gray scale image
    contours,hierarchy =  cv2.findContours(gaussian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    
    largest = 0
    ind = 0
    
    

    if(len(contours) > 0):
        
        for x in contours: #For all of the contours, find the largest and get it's index
            
            if(cv2.contourArea(contours[ind]) > cv2.contourArea(contours[largest])):
                largest = ind
            ind+=1
            
        #Draws the largest contour
        cv2.drawContours(frame, contours[largest], -1, (255,0,0),5)
        left = contours[largest][0][0][0]
        right = contours[largest][0][0][0]
        top = contours[largest][0][0][1]
        bot = contours[largest][0][0][1]
        # For all the points for the largest contour, find the outermost bounds that the contour goes
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
        
        frame = frame[bot:top,left:right]#Resizes a frame to be the dimensions of the tracked object    
        threshRez = thresh[bot:top,left:right]#Binary image scaled to these dimensions
        dim = (compression_size,compression_size)# Dimensions of the wanted compression, for MNIST 28x28
        frame = cv2.resize(frame,dim)# Resizes frame to the dimensions
        threshRez = cv2.resize(threshRez,dim)#Resizes the thresholded image
        pixels = translate(threshRez)#Highlights the negative space in the image, in this case hopefully the number
        nn.updateInput(np.reshape(pixels,(len(pixels),1))) #Reshape and update input for NeuralNetwork
        
         
        cv2.imshow('Color',threshRez)
    #Exits camera when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    cv2.imshow('Filter',thresh)
   
capture.release()
cv2.destroyAllWindows()


