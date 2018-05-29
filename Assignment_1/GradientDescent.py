import operator
import numpy as np
import matplotlib.pyplot as plt

STEP_SIZE = .000001
coef = [0.0, 0.0]

def main():
    xTrain, yTrain = prepData("Dataset_2_train.csv")
    xTest, yTest = prepData("Dataset_2_test.csv")
    xValid, yValid = prepData("Dataset_2_valid.csv")
    # mseGraph(xValid, yValid,xTrain, yTrain)
    # bestStep(xTest, yTest)
    mseEvolve(xValid,yValid)

def mseGraph(xValid,yValid, xTrain, yTrain):
    values = sgdMSE(xValid,yValid)
    coef = [0.0, 0.0]
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    testValues = []
    validValues = []
    for i in range(10000):
        learningCurve(xTrain, yTrain)
        learningCurve(xValid, yValid)
        testValues.append(MSE(yTrain,[predictY(xTrain[i]) for i in range(len(xTrain))]))
        validValues.append(MSE(yValid,[predictY(xValid[i]) for i in range(len(xValid))]))
    plt.plot([i for i in range(10000)], validValues, 'bo', label = 'Validation')
    plt.legend(loc = 'upper right')
    plt.figure(2)
    plt.plot(xTrain, yTrain, 'ro', label = 'Training')
    plt.plot(xValid, yValid, 'bo', label = 'Validating')
    plt.plot(xValid, [predictY(xValid[i]) for i in range(len(xValid))])
    plt.legend(loc='upper right')
    plt.show()

def bestStep(xTest, yTest):
    for i in range(100000):
        learningCurve(xTest, yTest)
    lowest = MSE(yTest,[predictY(xTest[i]) for i in range(len(xTest))])
    current = STEP_SIZE
    loop = 1
    while(current < .01):
        coef  = [0.0, 0.0]
        for i in range(10000):
            learningCurve(xTest, yTest)
            if (lowest > MSE(yTest,[predictY(xTest[i]) for i in range(len(xTest))])):
                lowest = MSE(yTest,[predictY(xTest[i]) for i in range(len(xTest))])
                step = current
        current *= 1.05
    print step
    print lowest

def mseEvolve(xValid, yValid):
    coef = [0.0, 0.0]
    plt.xlabel('x Values')
    plt.ylabel('y Values')
    for i in range(5):
        learningCurve(xValid, yValid)
        learningCurve(xValid, yValid)
    plt.plot(xValid, yValid, 'ro')
    plt.plot(xValid, [predictY(xValid[i]) for i in range(len(xValid))])
    plt.legend(loc='upper right')
    plt.show()
    for i in range(100):
        learningCurve(xValid, yValid)
        learningCurve(xValid, yValid)
    plt.plot(xValid, yValid, 'ro')
    plt.plot(xValid, [predictY(xValid[i]) for i in range(len(xValid))])
    plt.legend(loc='upper right')
    plt.show()
    for i in range(200):
        learningCurve(xValid, yValid)
        learningCurve(xValid, yValid)
    plt.plot(xValid, yValid, 'ro')
    plt.plot(xValid, [predictY(xValid[i]) for i in range(len(xValid))])
    plt.legend(loc='upper right')
    plt.show()
    for i in range(300):
        learningCurve(xValid, yValid)
        learningCurve(xValid, yValid)
    plt.plot(xValid, yValid, 'ro')
    plt.plot(xValid, [predictY(xValid[i]) for i in range(len(xValid))])
    plt.legend(loc='upper right')
    plt.show()
    for i in range(100000):
        learningCurve(xValid, yValid)
        learningCurve(xValid, yValid)
    plt.plot(xValid, yValid, 'ro')
    plt.plot(xValid, [predictY(xValid[i]) for i in range(len(xValid))])
    plt.legend(loc='upper right')
    plt.show()

def prepData(filename):
    trainData = np.genfromtxt(filename, delimiter=',')
    xData, yData = trainData[:,0], trainData[:, 1]
    L = sorted(zip(xData, yData), key = operator.itemgetter(0))
    newX, newY = zip(*L)
    return newX, newY  

def sgdMSE(x,y):
    tempX = []
    tempY = []
    output = []
    for i in range(len(np.array(x))):
        tempX.append(x[i])
        tempY.append(y[i])
        SGD(x[i], y[i])
        yHat = [predictY(j) for j in tempX] 
        output.append(MSE(tempY, yHat))
    return output

def learningCurve(x,y):
    for i in range(len(np.array(x))):
        SGD(x[i], y[i])

def SGD(x,y):
        coef[0] = coef[0] - (STEP_SIZE * ((predictY(x)) - y))
        coef[1] = coef[1] - (STEP_SIZE * ((predictY(x)) - y) * x)

def predictY(x):
    return coef[0] + coef[1]*x

def MSE(y,prediction):
    return sum([(prediction[i] - y[i]) ** 2 for i in range(len(y))])/len(y)

main()