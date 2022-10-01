#Framework for a single hidden layer NN, with variable neurons in hidden layer. 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
from scipy.special import expit
from keras.datasets import mnist


class NeuralNetwork:
    inputWeights = []
    outputWeights = []
    cellInfo = []
    input = []
    output = []
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    def __init__(self, outputLength,layerLength):
        print(f"Test: {self.train_X[0]}")
        # ** Initialize bias here as well
        output = self.output
        cellInfo = self.cellInfo
        outputWeights = self.outputWeights
        inputWeights = self.inputWeights
        input = self.input
        for o in range(outputLength):
            output.append(0)
        for cell in range(layerLength):
            
            # Cell info: [0] = value, [1] = bias
            cellInfo.append([0]*2)
            inputWeights.append([0]*len(input))
            for iw in range(len(input)):
                
                inputWeights[cell][iw] = (np.random.rand()*2)-1
        
        for cell in range(len(output)):
            outputWeights.append([0]*len(cellInfo))
            for ow in range(len(cellInfo)):
                
                outputWeights[cell][ow] = (np.random.rand()*2)-1
        
        self.inputWeights = inputWeights
        self.outputWeights = outputWeights
        self.cellInfo = cellInfo
        self.input = input
        self.output = output
        
        
    def feedforward(self):
        cellInfo = self.cellInfo
        output = self.output
        outputWeights = self.outputWeights
        inputWeights = self.inputWeights
        input = self.input
        ind = 0
        for cell in range(len(cellInfo)):
            # Cell info: [0] = value, [1] = bias
            sum = 0
            counter = 0
            for iw in inputWeights[cell]:
                sum += iw*input[counter]
                counter+=1
            cellInfo[cell][0] = 1/(1+np.exp(-sum+cellInfo[cell][1]))
            
        for c in range(len(output)):
            # Cell info: [0] = value, [1] = bias
            sum = 0
            counter = 0
            
            for ow in outputWeights[c]: 
                sum += ow*cellInfo[counter][0]
                counter+=1
            output[c] = (1/(1+np.exp(-sum)))
        largest = 0
        ind = 0
        for x in output:
            if x>output[largest]:
                largest = ind
            ind+=1
        return largest
    
    def updateInput(self, input):
        self.input = input
    def inputChecker(self):
        input = self.input
        zeroCount = 0
        oneCount = 1
        for x in input:
            
            if(x == 0):
                zeroCount += 1
            else :
                oneCount += 1
        print(f"zeroCount: {zeroCount}")
        print(F"oneCount: {oneCount}")
    