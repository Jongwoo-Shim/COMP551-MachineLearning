import operator
import numpy as np
import matplotlib.pyplot as plt
import csv

def main():
    trainData = prepData("communities.csv")
    train,test = genFiles(trainData)
    weights = MSELinear(train,test)
    with open("Weights", 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for line in weights:
            writer.writerow(line)

def MSELinear(train, test):
    values = []
    train = np.array(train)
    test = np.array(test)
    weights = []
    for i in train:
        y = []
        for row in i:
            y.append(row[127])
        i = removal(i)
        weights.append(calculateWeight(i,y))
    testSet = []
    y = []
    for i in test:
        temp = []
        for row in i:
            temp.append(row[127])
        y.append(temp)
        i = removal(i)
        testSet.append(i)
    for i in range(len(test)):
        prediction = predictY(testSet[i], weights[i])
        values.append(MSE(y[i], prediction))
    print (sum(values)/5)
    return weights

def MSEL2(train, test):
    values = []
    train = np.array(train)
    test = np.array(test)
    weights = []
    for i in train:
        y = []
        for row in i:
            y.append(row[127])
        i = removal(i)
        lam, weights = L2Regularization(i, y)
        weights.append(weights)
    print np.array(weights).shape
    testSet = []
    y = []
    for i in test:
        temp = []
        for row in i:
            temp.append(row[127])
        y.append(temp)
        i = removal(i)
        testSet.append(i)
    for i in range(len(test)):
        prediction = l2Predictions(test[i], y[i], weights[i])
        values.append(MSE(y[i], prediction))
    print (sum(values)/5)
    return weights
def removal(dataSet):
    dataSet = np.delete(dataSet, [0, len(dataSet)], axis = 1)
    dataSet = np.delete(dataSet, [0, len(dataSet)], axis = 1)
    dataSet = np.delete(dataSet, [0, len(dataSet)], axis = 1)
    dataSet = np.delete(dataSet, [0, len(dataSet)], axis = 1)
    dataSet = np.delete(dataSet, [0, len(dataSet)], axis = 1)
    dataSet = np.delete(dataSet, [0, len(dataSet)], axis = -1)
    return dataSet

def prepData(filename):
    trainData = np.genfromtxt(filename, delimiter=',')
    average = [0 for i in range(len(trainData[0]))]
    for i in range(len(trainData)):
        for j in range(len(trainData[i])):
            if(np.isnan(trainData[i][j]) == False):
                average[j] = (trainData[i][j]) + average[j] / 2
    trainData = np.genfromtxt(filename, delimiter=',')
    average = [0 for i in range(len(trainData[0]))]
    for i in range(len(trainData)):
        for j in range(len(trainData[i])):
            if(np.isnan(trainData[i][j]) == True):
                trainData[i][j] = average[j]
    return trainData

def genFiles(trainData):
    trainArray = []
    testArray = []
    for i in range(5):
        trainFileName = "CandC-train<"+str(i) + ">.csv"
        testFileName = "CandC-test<"+str(i)+ ">.csv"
        with open(trainFileName, 'w') as train, open(testFileName, 'w') as test:
            trainSet, testSet = splitData(trainData)
            trainArray.append(trainSet)
            testArray.append(testSet)  
            writer = csv.writer(train, delimiter=',')
            for line in trainSet:
                writer.writerow(line)
            writer = csv.writer(test, delimiter=',')
            for line in testSet:
                writer.writerow(line)
    return trainArray, testArray
            

def splitData(trainData):
    np.random.shuffle(trainData)
    slice = int(len(trainData) * .80)
    return (trainData[:slice],trainData[slice:])

def MSE(y,prediction):
    return sum([(prediction[i] - y[i]) ** 2 for i in range(len(y))])/len(y)

def calculateWeight(x, y):
    temp = (x.T).dot(x)
    temp = np.linalg.inv(np.array(temp))
    temp = temp.dot(x.T)
    temp = temp.dot(y)
    return temp

def predictY(x, weight):
    prediction  = []
    for i in range(len(x)):
        temp = []
        for j in range(len(weight)):
            temp.append(weight[j] * x[i][j])
        prediction.append(sum(temp))
    return prediction

def L2Regularization(x, y):
    weights = []
    lamList = []
    lam = 0.0001
    while lam < 1:
        temp = L2Calc(x,y, lam)
        weights.append(temp)
        lamList.append(lam)
        lam *= 0.0001
    return lamList, weights

def l2Predictions(x,y, weights):
    output = []
    for i in weights:
        output.append(MSE(y,predictY(x,i)))
    return output

main()