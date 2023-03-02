
import numpy as np
import pandas as pd
import random
#Potentially use leaky relu for weights and bias, and use sigmoid for output normalization.
#Avoiding vanishing gradient while keeping outputs scaled between -1 and 1
class NeuralNetwork:
    #Initializing variables
    answers = []
    weights = []
    bias = []
    input = []
    hlVals = []
    hlSig = []
    output = []
    DCoW = []
    DCoB = []
    sumCW = []
    sumCB = []
    options = []
    
    correct = 0
    #Used for measuring accuracy
    correctList = [0,0,0,0,0,0,0,0,0,0]
    incorrectList = [0,0,0,0,0,0,0,0,0,0]
    #Reading in Data and shuffling it, no partition between test and train sets, as train isn't perfectly
    #replicable for what I am working with
    data = pd.read_csv("C:\\PythonProjects\\DataFiles\\train.csv")
    testRange = 30000
    data = np.array(data[0:testRange])
    print(len(data))
    i = 0
    random.shuffle(data)
    #Returning answers for the associated array inputs
    for x in data:
        answers.append(x[0])
        
    
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
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    def softmax_grad(x): 
        # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
        # input s is softmax value of the original input x. 
        # s.shape = (1, n) 
        # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
        # initialize the 2-D jacobian matrix.
        jacobian_m = np.diag(x)
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = x[i] * (1-x[i])
                else: 
                    jacobian_m[i][j] = -x[i]*x[j]
        return jacobian_m
    def forwardProp(self):# Run through the application, starting at the input set, using matrix multiplication
        end = len(self.weights)-1
        #Start at input, multiply through weights, end at output. Similar structure to top
        self.hlVals.append(self.weights[0].dot(self.input) + self.bias[0])
        self.hlSig.append(self.sigmoid(self.hlVals[0]))
        for i in range(1,end):
            self.hlVals.append(self.weights[i].dot(self.hlSig[i-1]) + self.bias[i])
            self.hlSig.append(self.sigmoid(self.hlVals[i]))
            
        self.hlVals.append(self.weights[end].dot(self.hlSig[end-1]) + self.bias[end])
        self.hlSig.append(self.sigmoid(self.hlVals[end]))
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
        for i in range(len(self.weights)):
            
            self.weights[i] -= np.multiply(self.DCoW[x],alpha)
            x-=1
        y = len(self.weights)-1
        for i in range(len(self.weights)):
            
            self.bias[i] -= np.multiply(self.DCoB[y],alpha)
            y-=1
        self.sumCB.clear()
        self.sumCW.clear()
        

    def train(self,iterations, alpha): #Putting it all together
        
       
        accuracy = []
        
        for i in range(0,iterations):
            x = i#random.randint(0,self.testRange-1)
            
            self.trainInput(x)
            self.forwardProp()
            
            self.backProp(x)
            #The batch collects the weights over a certain # of iterations, then applying them at the end
            #of the batch.
            
           
            self.update(alpha) #Updates changes

                
            #At i%10==9,  
        
            
            if(i%1000==0):
                print(self.hlSig[len(self.hlSig)-1])
                accuracy.append(self.correct/1000)
                print(self.correct/1000)
                
                self.correct = 0
                print(f"Correct Guesses: {self.correctList}")
                print(f"Incorrect Guesses: {self.incorrectList}")
                self.correctList = [0]*len(self.correctList)
                self.incorrectList = [0]*len(self.correctList)
            self.clear()
            
        
    def updateInput(self, input):
        self.input = input
    def clear(self):
        self.hlVals.clear()
        self.hlSig.clear()
        self.DCoW.clear()
        self.DCoB.clear()
    
            
    def trainInput(self, ind): # Changes values of Mnist from 0-255 to 0-1
        
        self.input =  np.reshape(self.data[ind][1:785],(784,1))
        for i in range(0,len(self.input)):
            val = self.input[i]
            if(val>=230):
                self.input[i]=1
            else :
                self.input[i]=0
    def guessNum(self): #Meant to be used with camera information
        self.forwardProp()
        guess = np.argmax(self.hlSig[len(self.hlSig)-1])
        print(guess)
