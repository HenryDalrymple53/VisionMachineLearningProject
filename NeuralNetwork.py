
import numpy as np
import pandas as pd
import random
#Potentially use leaky relu for weights and bias, and use sigmoid for output normalization.
#Avoiding vanishing gradient while keeping outputs scaled between -1 and 1
class NeuralNetwork:
    #Initializing variables
    weights = []
    bias = []
    answers = []
    input = []
    hlVals = []
    hlSig = []
    output = []
    DCoW = []
    DCoB = []
    sumCW = []
    sumCB = []
    only123 = []
    options = []
    
    correct = 0
    #Used for measuring accuracy
    correctList = [0,0,0,0,0,0,0,0,0,0]
    incorrectList = [0,0,0,0,0,0,0,0,0,0]
    #Reading in Data and shuffling it, no partition between test and train sets, as train isn't perfectly
    #replicable for what I am working with
    data = pd.read_csv("C:\\PythonProjects\\DataFiles\\train.csv")
    data = np.array(data[0:50000])
    random.shuffle(data)
    i = 0
    #Returning answers for the associated array inputs
    for x in data:
        answers.append(x[0])
        if(x[0]==1 or x[0]==2 or x[0]==3 or x[0]==4):
            only123.append(i)
        i+=1
    
    outputLength = 10
    hiddenLayerLength = 0
    #Initialize weights and bias for dimensionality given
    def __init__(self, layerLengths):
        prev = 784
        
        for i in range(0,len(layerLengths)):
            self.weights.append((np.random.normal(size=(layerLengths[i], prev)) * np.sqrt(2/layerLengths[i])))
            prev = layerLengths[i]
            self.bias.append((np.random.normal(size=(prev, 1)) * np.sqrt(2/prev)))
        self.weights.append((np.random.normal(size=(self.outputLength, prev)) * np.sqrt(2/self.outputLength)))
        self.bias.append(np.random.normal(size=(self.outputLength, 1)) * np.sqrt(2/self.outputLength))
    
    def sigmoid(self, x): #Sigmoid squishes any number given to it
        sig = 1 / (1 + np.exp(-x))
        return sig
    def sigdir(self,x): #Sigmoid der
        return self.sigmoid(x) * (1- self.sigmoid(x))

    def leaky_Relu(self,x): #Unused leaky_relu
        count = 0
        for val in x:
             x[count] = val*0.01 if val < 0 else val
             count+=1
        return x

    def  leaky_Relu_Derivative(self,x): # Unused leaky_relu derv
        count = 0
        for val in x:
            x[count] = 0.01 if val < 0 else 1
            count+=1
        return x
    def forwardProp(self):# Run through the application, starting at the input set, using matrix multiplication

        #Start at input, multiply through weights, end at output. Similar structure to top
        self.hlVals.append(self.weights[0].dot(self.input) + self.bias[0])
        self.hlSig.append(self.sigmoid(self.hlVals[0]))
        for i in range(1,len(self.weights)):
            self.hlVals.append(self.weights[i].dot(self.hlSig[i-1]) + self.bias[i])
            self.hlSig.append(self.sigmoid(self.hlVals[i]))
            

    def backProp(self, ind):#Backpropigate through the values, finding gradient descent curves for each
                            #weight and bias
        #Cost of the function
        prev = 2*(self.hlSig[len(self.hlSig)-1]-self.responseCleaner(self.answers[ind]))
        for i in range(len(self.hlSig)-1,0,-1):
            
            DAoZn = self.sigdir(prev)   
            DZoWn = np.reshape(self.hlSig[i-1],(1,len(self.hlSig[i-1])))
            DZnA1 = np.reshape(self.weights[i],(len(self.hlSig[i-1]),len(self.hlSig[i])))
            dot = DAoZn*prev
            self.DCoW.append(dot*DZoWn)
            self.DCoB.append(dot)
            prev = DZnA1.dot(dot)
        
        DAoZn = self.sigdir(prev)   #5,1
        DZoWn = np.reshape(self.input,(1,len(self.input))) #1,784
        DZnA1 = np.reshape(self.weights[0],(len(self.input),len(self.hlSig[0]))) #784,5
        self.DCoW.append((DAoZn*prev)*DZoWn)
        
        self.DCoB.append(DAoZn*prev)
        
            
    def responseCleaner(self, ans): #Finds the most confident answer given, and creates a binary array 
                                    #With 1 at greatest value
        
        retArr = []
        for x in range(0,10):
            if(x==ans):
                retArr.append(1)
            else :
                retArr.append(0)
        guess = np.argmax(self.hlSig[len(self.hlSig)-1])
        
        if(guess==ans):
            self.correctList[ans]+=1
            self.correct +=1
        else :
            self.incorrectList[ans]+=1
        return np.reshape(retArr,(len(retArr),1))

    def update(self, alpha): #This updates the values after the length of the batch.
        x = len(self.weights)-1
        for i in range(0,len(self.weights)):
            
            self.weights[i] -= np.multiply(self.sumCW[x],alpha)
            x-=1
        y = len(self.weights)-1
        for i in range(0,len(self.weights)):
            
            self.bias[i] -= np.multiply(self.sumCB[y],alpha)
            y-=1
        self.sumCB.clear()
        self.sumCW.clear()
        

    def train(self,iterations, alpha, epoch, batch): #Putting it all together
        data = []
       
        accuracy = []
        for x in range(0,epoch):
            print(f"Start Val: {self.only123[0:5]}")
            print(self.weights[len(self.weights)-1])
            for i in range(0,iterations):
                
                self.trainInput(i)
                self.forwardProp()
                
                self.backProp(i)
                #The batch collects the weights over a certain # of iterations, then applying them at the end
                #of the batch.
                if(i%batch==0):
                    self.sumCW = self.DCoW #Zeroes out the changes
                    self.sumCB = self.DCoB 
                elif(i%batch==batch-1):
                    self.update(alpha) #Updates changes
                else:
                    self.summation(alpha) #Adds changes
                #At i%10==9,  
            
                self.clear()
                if(i%1000==0):
                    
                    accuracy.append(self.correct/1000)
                    print(self.correct/1000)
                    self.correct = 0
            print(f"Next Epoch ----------------------")
            print(self.weights[len(self.weights)-1])
        print(f"Correct Guesses: {self.correctList}")
        print(f"Incorrect Guesses: {self.incorrectList}")
        
    def updateInput(self, input):
        self.input = input
    def clear(self):
        self.hlVals.clear()
        self.hlSig.clear()
        self.DCoW.clear()
        self.DCoB.clear()
    def summation(self, alpha): #Is called throughout the length of the batch to add all of the weights and biases
        
        for i in range(0,len(self.weights)):
            
            self.sumCW[i] += np.multiply(self.DCoW[i],alpha)
            
        
        for i in range(0,len(self.weights)):
            
            self.sumCB[i] += np.multiply(self.DCoB[i],alpha)
            
    def trainInput(self, ind): # Changes values of Mnist from 0-255 to 0-1
        
        self.input =  np.reshape(self.data[ind][1:785],(784,1))
        for i in range(0,len(self.input)):
            val = self.input[i]
            if(val>=230):
                self.input[i]=1
            else :
                self.input[i]=0
    def guessNum(self, input): #Meant to be used with camera information
        self.forwardProp()
        guess = np.argmax(self.hlSig[len(self.hlSig)-1])
        print(guess)
