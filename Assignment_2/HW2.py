import numpy as np
import math, csv, random, operator
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    negTrain, negTest, posTrain, posTest = part1()
    part2(negTrain, negTest, posTrain, posTest)
    part5()

def part1():
    cov = np.genfromtxt("DS1_Cov.txt", delimiter=',')
    cov = cov[:,:-1]
    neg = np.genfromtxt("DS1_m_0.txt", delimiter=',')
    neg = neg[:-1]
    pos = np.genfromtxt("DS1_m_1.txt", delimiter=',')
    pos = pos[:-1]
    neg = appendSign(genGaus(cov, neg), -1)
    pos = appendSign(genGaus(cov, pos), 1)
    negTrain, negTest = splitData(neg)
    posTrain, posTest = splitData(pos)
    train = negTrain + posTrain
    test = negTest + posTest
    writeCSV(neg + pos, 'DS1.txt')
    return np.array(negTrain), np.array(negTest), np.array(posTrain), np.array(posTest)
    
def part2(negTrain, negTest, posTrain, posTest):
    w,w0 = LDA(negTrain[:,1:], posTrain[:,1:])
    posResult = []
    negResult = []
    w = w.flatten()
    for i in negTest[:,1:]:
        i = np.array(i)
        temp = np.dot(w, i) + w0
        negResult.append(sigmoid(temp))
    for i in posTest[:,1:]:
        i = np.array(i)
        temp = np.dot(w, i) + w0
        posResult.append(sigmoid(temp))
    negResult = [1 if x > .5 else 0 for x in negResult]
    posResult = [1 if x > .5 else 0 for x in posResult]
    truePos = (count(posResult, 1))
    falseNeg = len(posTrain) - truePos
    trueNeg = (count(negResult, 0 ))
    falsePos = len(negTest) - trueNeg
    accuracy = (truePos + trueNeg) / float(len(negTest) + len(posTest))
    precision = truePos / float(truePos + falsePos)
    recall = truePos/float(truePos + falseNeg)
    F1 = (2 * precision * recall) /float(precision + recall)
    print(accuracy)
    print(precision)
    print(recall)
    print(F1)
    print(w)
    print(w0)

def part3(negTrain, negTest, posTrain, posTest):
    acc = []
    prec = []
    rec = []
    F1 = []
    for k in tqdm(range(1,11)):
        negResult, posResult = kNeighbors(k,negTrain, negTest, posTrain, posTest)
        truePos = (count(posResult, 1))
        falseNeg = len(posTrain) - truePos
        trueNeg = (count(negResult, 0 ))
        falsePos = len(negTest) - trueNeg
        acc.append((truePos + trueNeg) / float(len(negTest) + len(posTest)))
        precision = truePos / float(truePos + falsePos)
        prec.append(precision)
        recall = truePos/float(truePos + falseNeg)
        rec.append(recall)
        F1.append((2 * precision * recall) /float(precision + recall))
    acc = np.array(acc)
    prec = np.array(prec)
    rec = np.array(rec)
    F1 = np.array(F1)
    k = np.array([i for i in range(1,11)])
    plt.figure(0)
    plt.xlabel('k Values')
    plt.ylabel('Accuracy')
    plt.plot(k, acc, 'ro', label = 'Accuracy')
    plt.legend(loc='upper right')
    plt.plot(k, acc)
    plt.show()
    plt.figure(1)
    plt.xlabel('k Values')
    plt.ylabel('Precision')
    plt.plot(k, prec, 'ro', label = 'Precision')
    plt.legend(loc='upper right')
    plt.plot(k, prec)
    plt.show()
    plt.figure(2)
    plt.xlabel('k Values')
    plt.ylabel('Recall')
    plt.plot(k, rec, 'ro', label = 'Recall')
    plt.legend(loc='upper right')
    plt.plot(k, rec)
    plt.show()
    plt.figure(3)
    plt.xlabel('k Values')
    plt.ylabel('F1')
    plt.plot(k, F1, 'ro', label = 'F1')
    plt.legend(loc='upper right')
    plt.plot(k, F1)
    plt.show()

def part4():
    pos1 = np.genfromtxt("DS2_c1_m1.txt", delimiter=',')
    pos1 = pos1[:-1]
    pos2 = np.genfromtxt("DS2_c1_m2.txt", delimiter=',')
    pos2 = pos2[:-1]
    pos3 = np.genfromtxt("DS2_c1_m3.txt", delimiter=',')
    pos3 = pos3[:-1]

    neg1 = np.genfromtxt("DS2_c2_m1.txt", delimiter=',')
    neg1 = neg1[:-1]
    neg2 = np.genfromtxt("DS2_c2_m2.txt", delimiter=',')
    neg2 = neg2[:-1]
    neg3 = np.genfromtxt("DS2_c2_m3.txt", delimiter=',')
    neg3 = neg3[:-1]

    cov1 = np.genfromtxt("DS2_Cov1.txt", delimiter=',')
    cov1 = cov1[:,:-1]
    cov2 = np.genfromtxt("DS2_Cov2.txt", delimiter=',')
    cov2 = cov2[:,:-1]
    cov3 = np.genfromtxt("DS2_Cov3.txt", delimiter=',')
    cov3 = cov3[:,:-1]

    neg = appendSign(DS2GenGaus(neg1, neg2, neg3, cov1, cov2, cov3), -1)
    pos = appendSign(DS2GenGaus(pos1, pos2, pos3, cov1, cov2, cov3), 1)
    negTrain, negTest = splitData(neg)
    posTrain, posTest = splitData(pos)
    train = negTrain + posTrain
    test = negTest + posTest
    writeCSV(neg + pos, 'DS2.txt')
    return np.array(negTrain), np.array(negTest), np.array(posTrain), np.array(posTest)

def part5():
    negTrain, negTest, posTrain, posTest = part4()
    part2(negTrain, negTest, posTrain, posTest)
    part3(negTrain, negTest, posTrain, posTest)
    
def genGaus(cov, mean):
    x = np.empty([2000,20])
    for i in range(2000):
        x[i] = np.random.multivariate_normal(mean, cov).T
    return x

def writeCSV(x, filename):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerows(x)

def appendSign(x, sign):
    x = x.tolist()
    for i in range(len(x)):
        x[i] = [sign] + x[i][:]
    return x

def splitData(data):
    np.random.shuffle(data)
    slice = int(len(data) * .70)
    return (data[:slice],data[slice:])

def LDA(neg, pos):
    x = neg + pos
    p0 = len(neg)/float(len(x))
    p1 = len(pos)/float(len(x))
    u0 = np.array(getMean(neg))
    u1 = np.array(getMean(pos))
    cov = getCov(neg, pos, u0, u1)
    cov = np.linalg.inv(cov)
    w = cov.dot(u0 - u1)
    w0 = math.log(p0/p1)
    w0 = w0 + -.5 * np.dot(np.dot(u0.T,cov),u0)
    w0 = w0 + .5 *np.dot(np.dot(u1.T,cov),u1)
    return w, w0

def sigmoid(a):
    return 1/(1 + float(math.exp(a)))

def getMean(data):
    return [sum(x)/len(x) for x in zip(*data)]

def getCov(neg, pos, u0, u1):
    neg = [i - u0 for i in neg]
    pos = [i - u0 for i in pos]
    neg = np.matrix(neg)
    pos = np.matrix(pos)
    S1 = 1/float(len(neg)) * np.dot(neg.T, neg)
    S2 = 1/float(len(pos)) * np.dot(pos.T,pos)
    S2 = S2 / float(len(neg))
    N = float(len(neg) + len(pos))
    cov = (len(neg)/N)*S1 + S2 * (len(pos)/N)
    return cov

def count(matrix, sign):
    count = 0
    for i in matrix:
        if(sign == 1):
            if(i == 1):
                count += 1
        elif(sign == 0):
            if(i == 0):
                count += 1 
    return count

def kNeighbors(k,negTrain, negTest, posTrain, posTest):
    trainSet = np.array(negTrain.tolist() +  posTrain.tolist())
    posResult = []
    negResult = []
    for i in posTest:
        neighbors = getNeighbors(trainSet[:,1:], i, k)
        temp = [trainSet[j][0][0] for j in neighbors]
        posResult.append(getVote(temp))
    for i in negTest:
        neighbors = getNeighbors(trainSet[:,1:], i, k)
        temp = [trainSet[j][0][0] for j in neighbors]
        negResult.append(getVote(temp))
    negResult = [1 if x > 0 else 0 for x in negResult]
    posResult = [1 if x > 0 else 0 for x in posResult]
    return negResult, posResult


def getNeighbors(data, testCase, k):
    distances = []
    for x in range(len(data)):
        distances.append([x, euclideanDistance(data[x], testCase)])
    distances.sort(key=operator.itemgetter(1))
    output = []
    for i in range(k):
        output.append([distances[i][0]])
    return output

def euclideanDistance(point1, point2):
    temp = [(x1 - x2)** 2 for x1 in point1 for x2 in point2]
    temp = float(sum(temp))
    return math.sqrt(temp)

def getVote(neighbors):
    return sum(neighbors)/float(len(neighbors))

def DS2GenGaus(m1,m2,m3,cov1,cov2,cov3):
    x = np.empty([2000,20])
    for i in range(2000):
        rand = random.randint(0,100)
        if(rand < 11):
            x[i] = np.random.multivariate_normal(m1, cov1).T
        elif(rand < 53):
            x[i]= np.random.multivariate_normal(m2, cov2).T
        else:
            x[i] = np.random.multivariate_normal(m3, cov3).T
    return x

main()