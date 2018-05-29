import operator
import numpy as np
import matplotlib.pyplot as plt

POWER = 20
def main():
    xTrain, yTrain = prepData("Dataset_1_train.csv")
    xTest, yTest = prepData("Dataset_1_test.csv")
    xValid, yValid = prepData("Dataset_1_valid.csv")
    generateGraph(xTrain, yTrain, xValid, yValid, calculateWeight(xTrain,yTrain, POWER))
    generateMSEGraph(xTrain,yTrain, xValid, yValid, xTest, yTest)

def generateGraph(xTrain, yTrain, xValid, yValid, weight):
    plt.figure(0)
    plt.xlabel('y Values')
    plt.ylabel('x Values')
    plt.plot(xTrain, yTrain, 'ro', label = 'Training')
    plt.plot(xValid, yValid, 'bo', label = "Validation")
    plt.legend(loc='upper right')
    plt.plot(xValid, predictY(xValid, weight))
    print(MSE(yTrain, predictY(xTrain, weight)))
    print(MSE(yValid, predictY(xValid, weight)))
    plt.axis([-1,1,-50, 50])
    plt.show()

def generateMSEGraph(xTrain, yTrain, xValid, yValid, xTest, yTest):
    plt.figure(1)
    plt.xlabel('Lamda')
    plt.ylabel('MSE')
    lam, weights = L2Regularization(xTrain, yTrain)
    lam = np.array(lam)
    print(lam)
    weights = np.array(weights)
    train = l2Predictions(xTrain, yTrain, weights)
    valid = l2Predictions(xValid, yValid, weights)
    plt.plot(lam, train , 'ro', label = 'Training')
    plt.plot(lam, valid, 'bo', label = "Validation")
    plt.plot(lam, train)
    plt.plot(lam, valid)
    plt.legend(loc = 'upper right')
    plt.show()
    plt.figure(2)
    plt.xlabel('x Values')
    plt.ylabel('y Values')
    print(lam[valid.index(min(valid))])
    plt.plot(xTest, yTest, 'ro', label = 'Test')
    plt.plot(xTest, predictY(xTest, weights[valid.index(min(valid))]))
    plt.legend(loc = 'upper right')
    plt.show()

def prepData(filename):
    trainData = np.genfromtxt(filename, delimiter=',')
    xData, yData = trainData[:,0], trainData[:, 1]
    L = sorted(zip(xData, yData), key = operator.itemgetter(0))
    newX, newY = zip(*L)
    return newX, newY   

def calculateWeight(xColumn, yColumn, POWER):
    x = np.array([[xColumn[i]**j for j in range(POWER+1)] for i in range(len(xColumn))])
    temp = np.linalg.inv((x.T).dot(x))
    temp = temp.dot(x.T)
    temp = temp.dot(yColumn)
    return temp

def predictY(x, weight):
    return np.array([sum([weight[j] * x[i] ** j for j in range(len(weight))]) for i in range(0, len(x))]).T

def MSE(y,prediction):
    return sum([(prediction[i] - y[i]) ** 2 for i in range(len(y))])/len(y)

def L2Calc(x, y, lam):
    return  np.linalg.inv(x.T.dot(x) + lam*np.identity(POWER + 1)).dot(x.T).dot(y)

def L2Regularization(x, y):
    weights = []
    lamList = []
    x = np.array([[x[i]**j for j in range(POWER+1)] for i in range(len(x))])
    lam = 0.0001
    while lam < 1:
        temp = L2Calc(x,y, lam)
        weights.append(temp)
        lamList.append(lam)
        lam += 0.0001
    return lamList, weights

def l2Predictions(x,y, weights):
    output = []
    for i in weights:
        output.append(MSE(y,predictY(x,i)))
    return output

main()