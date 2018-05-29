import numpy as np
from tqdm import tqdm
import csv, os, re, string, operator
import getch as m
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB 
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
import pandas as pd

pd.options.mode.chained_assignment = None

def main():
    IMDB()

def yelp():
    print("Prep Yelp")
    yelpTrain, yTrain = getData("yelp-train.txt")
    yelpValid, yValid = getData("yelp-valid.txt")
    yelpTest, yTest = getData("yelp-test.txt")
    yelpTrain = splitData(yelpTrain)
    yelpValid = splitData(yelpValid)
    yelpTest = splitData(yelpTest)
    topList = topWords(yelpTrain, "yelp-vocab.txt")
    print("Yelp Train")
    yelpTrain = convertData(yelpTrain, yTrain, topList, 'yelp-train.txt')
    print("Yelp Valid")
    yelpValid = convertData(yelpValid, yValid, topList, 'yelp-valid.txt')
    print("Yelp Test")
    yelpTest = convertData(yelpTest, yTest, topList, 'yelp-test.txt')

    print("Yelp Dummy - Binary")
    yelpDummy = trainDummy(binaryTable(yelpTrain), yTrain)
    yelpValidDummy = f1_score(yValid, predictDummy(yelpDummy, binaryTable(yelpValid)),average = 'micro')
    yelpTestDummy = f1_score(yTest,predictDummy(yelpDummy, binaryTable(yelpTest)),average = 'micro')
    print("ValidBayes - Binary  F1 = "+ str(yelpValidDummy))
    print("TestBayes - Binary F1 = " + str(yelpTestDummy) + "\n")

    print("Yelp Dummy - Frequency")
    yelpDummy = trainDummy(frequencyTable(yelpTrain), yTrain)
    yelpValidDummy = f1_score(yValid, predictDummy(yelpDummy, frequencyTable(yelpValid)),average = 'micro')
    yelpTestDummy = f1_score(yTest,predictDummy(yelpDummy, frequencyTable(yelpTest)),average = 'micro')
    print("ValidBayes - Frequency  F1 = "+ str(yelpValidDummy))
    print("TestBayes - Frequency F1 = " + str(yelpTestDummy) + "\n")

    print("Yelp Vote - Binary")
    yelpVote = trainVote(binaryTable(yelpTrain), yTrain)
    yelpValidVote = f1_score(yValid, predictDummy(yelpVote, binaryTable(yelpValid)),average = 'micro')
    yelpTestVote = f1_score(yTest,predictDummy(yelpVote, binaryTable(yelpTest)),average = 'micro')
    print("ValidBayes - Binary  F1 = "+ str(yelpValidVote))
    print("TestBayes - Binary F1 = " + str(yelpTestVote) + "\n")

    print("Yelp Vote - Frequency")
    yelpVote = trainVote(frequencyTable(yelpTrain), yTrain)
    yelpValidVote = f1_score(yValid, predictDummy(yelpVote, frequencyTable(yelpValid)),average = 'micro')
    yelpTestVote = f1_score(yTest,predictDummy(yelpVote, frequencyTable(yelpTest)),average = 'micro')
    print("ValidBayes - Frequency  F1 = "+ str(yelpValidVote))
    print("TestBayes - Frequency F1 = " + str(yelpTestVote) + "\n")

    print("Yelp Bayes - Binary")
    yelpBayes = trainBayes(binaryTable(yelpTrain), yTrain)
    yelpValidBayes = f1_score(yValid, predictBayes(yelpBayes, binaryTable(yelpValid)),average = 'micro')
    yelpTestBayes = f1_score(yTest,predictBayes(yelpBayes, binaryTable(yelpTest)),average = 'micro')
    print("ValidBayes - Binary  F1 = "+ str(yelpValidBayes))
    print("TestBayes - Binary F1 = " + str(yelpTestBayes) + "\n")

    print("Yelp Bayes - Frequency")
    yelpBayes = trainGBayes(frequencyTable(yelpTrain), yTrain)
    yelpValidBayes = f1_score(yValid, predictGBayes(yelpBayes, frequencyTable(yelpValid)),average = 'micro')
    yelpTestBayes = f1_score(yTest,predictGBayes(yelpBayes, frequencyTable(yelpTest)),average = 'micro')
    print("ValidGBayes - Frequency  F1 = "+ str(yelpValidBayes))
    print("TestGBayes - Frequency F1 = " + str(yelpTestBayes) + "\n")

    print("Yelp Tree - Binary")
    yelpTree = trainTree(binaryTable(yelpTrain), yTrain)
    yelpValidTree = f1_score(yValid, predictTree(yelpTree, binaryTable(yelpValid)),average = 'micro')
    yelpTestTree = f1_score(yTest,predictTree(yelpTree, binaryTable(yelpTest)),average = 'micro')
    print("ValidTree - Binary  F1 = "+ str(yelpValidTree))
    print("TestTree - Binary F1 = " + str(yelpTestTree) + "\n")

    print("Yelp Tree - Frequency")
    yelpTree = trainTree(frequencyTable(yelpTrain), yTrain)
    yelpValidTree = f1_score(yValid, predictTree(yelpTree, frequencyTable(yelpValid)),average = 'micro')
    yelpTestTree = f1_score(yTest,predictTree(yelpTree, frequencyTable(yelpTest)),average = 'micro')
    print("ValidTree - Frequency  F1 = "+ str(yelpValidTree))
    print("TestTree - Frequency F1 = " + str(yelpTestTree) + "\n")

    print("Yelp SVM - Binary")
    yelpSVM = trainSVM(binaryTable(yelpTrain), yTrain)
    yelpValidSVM = f1_score(yValid, predictSVM(yelpSVM, binaryTable(yelpValid)),average = 'micro')
    yelpTestSVM = f1_score(yTest,predictSVM(yelpSVM, binaryTable(yelpTest)),average = 'micro')
    print("ValidSVM - Binary  F1 = "+ str(yelpValidSVM))
    print("TestSVM - Binary F1 = " + str(yelpTestSVM) + "\n")

    print("Yelp SVM - Frequency")
    yelpSVM = trainSVM(frequencyTable(yelpTrain), yTrain)
    yelpValidSVM = f1_score(yValid, predictSVM(yelpSVM, frequencyTable(yelpValid)),average = 'micro')
    yelpTestSVM = f1_score(yTest,predictSVM(yelpSVM, frequencyTable(yelpTest)),average = 'micro')
    print("ValidSVM - Frequency  F1 = "+ str(yelpValidSVM))
    print("TestSVM - Frequency F1 = " + str(yelpTestSVM) + "\n")
 
def IMDB():
    print("Prep IMDB")
    IMDBTrain, yTrain = getData("IMDB-train.txt")
    IMDBValid, yValid = getData("IMDB-valid.txt")
    IMDBTest, yTest = getData("IMDB-test.txt")
    IMDBTrain = splitData(IMDBTrain)
    IMDBValid = splitData(IMDBValid)
    IMDBTest = splitData(IMDBTest)
    topList = topWords(IMDBTrain, "IMDB-vocab.txt")
    print("IMDB Train")
    IMDBTrain = convertData(IMDBTrain, yTrain, topList, 'IMDB-train.txt')
    print("IMDB Valid")
    IMDBValid = convertData(IMDBValid, yValid, topList, 'IMDB-valid.txt')
    print("IMDB Test")
    IMDBTest = convertData(IMDBTest, yTest, topList, 'IMDB-test.txt')
    
    print("IMDB Bayes - Binary")
    IMDBBayes = trainBayes(binaryTable(IMDBTrain), yTrain)
    IMDBTestBayes = f1_score(yTest, predictBayes(IMDBBayes, binaryTable(IMDBTest)),average = 'micro')
    IMDBValidBayes = f1_score(yValid,predictBayes(IMDBBayes, binaryTable(IMDBValid)),average = 'micro')
    print("TestBayes - Binary  F1 = "+ str(IMDBTestBayes))
    print("ValidBayes - Binary F1 = " + str(IMDBValidBayes))

    print("IMDB Bayes - Frequency")
    IMDBBayes = trainGBayes(frequencyTable(IMDBTrain), yTrain)
    IMDBTestBayes = f1_score(yTest, predictGBayes(IMDBBayes, frequencyTable(IMDBTest)),average = 'micro')
    IMDBValidBayes = f1_score(yValid,predictGBayes(IMDBBayes, frequencyTable(IMDBValid)),average = 'micro')
    print("TestGBayes - Frequency  F1 = "+ str(IMDBTestBayes))
    print("ValidGBayes - Frequencey F1 = " + str(IMDBValidBayes))

    print("IMDB Tree - Binary")
    IMDBTree = trainTree(binaryTable(IMDBTrain), yTrain)
    IMDBTestTree = f1_score(yTest, predictTree(IMDBTree, binaryTable(IMDBTest)),average = 'micro')
    IMDBValidTree = f1_score(yValid,predictTree(IMDBTree, binaryTable(IMDBValid)),average = 'micro')
    print("TestTree - Binary  F1 = "+ str(IMDBTestTree))
    print("ValidTree - Binary F1 = " + str(IMDBValidTree))

    print("IMDB Tree - Frequency")
    IMDBTree = trainTree(frequencyTable(IMDBTrain), yTrain)
    IMDBTestTree = f1_score(yTest, predictTree(IMDBTree, frequencyTable(IMDBTest)),average = 'micro')
    IMDBValidTree = f1_score(yValid,predictTree(IMDBTree, frequencyTable(IMDBValid)),average = 'micro')
    print("TestTree - Frequency  F1 = "+ str(IMDBTestTree))
    print("ValidTree - Frequency F1 = " + str(IMDBValidTree))


    print("IMDB SVM - Binary")
    IMDBSVM = trainSVM(binaryTable(IMDBTrain), yTrain)
    IMDBTestSVM = f1_score(yTest, predictSVM(IMDBSVM, binaryTable(IMDBTest)),average = 'micro')
    IMDBValidSVM = f1_score(yValid,predictSVM(IMDBSVM, binaryTable(IMDBValid)),average = 'micro')
    print("TestSVM - Binary  F1 = "+ str(IMDBTestSVM))
    print("ValidSVM - Binary F1 = " + str(IMDBValidSVM))

    print("IMDB SVM - Frequency")
    IMDBTestSVM = f1_score(yTest, predictSVM(IMDBSVM, frequencyTable(IMDBTest)),average = 'micro')
    IMDBValidSVM = f1_score(yValid,predictSVM(IMDBSVM, frequencyTable(IMDBValid)),average = 'micro')
    print("TestSVM - Frequency  F1 = "+ str(IMDBTestSVM))
    print("ValidSVM - Frequency F1 = " + str(IMDBValidSVM))

def getData(file):
    data = pd.read_csv(file, sep = "\t", header = None)
    exclude = set(string.punctuation)
    for line in range(data[0].count()):
        temp = data[0][line]
        temp = ''.join(ch for ch in temp if ch not in exclude)
        temp = temp.lower()
        data[0][line] = temp
    return data.values, data.values[:,1].astype(int)

def splitData(data):
    output = []
    for i in data:
        temp = "".join(str(i[j]) for j in range(len(i) - 1))
        temp = temp.split()
        output.append(temp)
    return output

def topWords(data, file):
    dictionary = {}
    for i in tqdm(data):
        for j in i:
            if j in dictionary:
                dictionary[j] += 1
            else:
                dictionary[j] = 1
    data = sorted(dictionary.items(), key = operator.itemgetter(1),reverse = True)[:10000]
    output = {}
    with open("./dataset/" + file, 'w') as f:
        for i in tqdm(range(len(data))):
            write = ''. join(str(data[i][0]) + "\t" + str(i + 1) + "\t" + str(data[i][1]))
            if(i != len(data)):
                write += "\n"
            f.write(write)
            output[data[i][0]] = i+1
    return output

def convertData(data, y, topWords, file):
    output = []
    with open("./dataset/" + file, 'w') as f:
        for i in tqdm(range(len(data))):
            temp = []
            for j in data[i]:
                if j in topWords.keys():
                    temp.append(topWords[j])
            write = ''.join(str(h) + ' ' for h in temp)
            write += '\t' + str(y[i]) + '\n'
            f.write(write)
            output.append(temp)
    return output

def binaryTable(inputs):
    table = np.zeros((len(inputs), 10000))
    # Turns into a set then list to get only one occurance of each object
    for i in tqdm(range(len(inputs))):
        for j in inputs[i]:
            table[i][j - 1] = 1
    return table

def frequencyTable(inputs):
    table = np.zeros((len(inputs), 10000))
    # Turns into a set then list to get only one occurance of each object
    for i in tqdm(range(len(inputs))):
        count = len(inputs[i])
        for j in inputs[i]:
            table[i][j - 1] = float(inputs[i].count(j))/float(count)
    return table

def trainDummy(x,y):
    dummy = DummyClassifier(strategy = 'uniform')
    dummy.fit(x,y)
    return dummy

def predictDummy(dummy, predict):
    return dummy.predict(predict)

def trainVote(x,y):
    dummy = DummyClassifier(strategy = 'most_frequent')
    dummy.fit(x,y)
    return dummy

def predictVote(vote, predict):
    return vote.predict(predict)

def trainBayes(x,y):
    bayes =  BernoulliNB()
    bayes.fit(x, y)
    return bayes

def predictBayes(bayes, predict):
    return bayes.predict(predict)

def trainTree(x,y):
    dTree = tree.DecisionTreeClassifier()
    dTree = dTree.fit(x,y)
    return dTree

def predictTree(tree, predict):
    return tree.predict(predict)

def trainSVM(x,y):
    SVM = LinearSVC()
    SVM.fit(x, y)
    return SVM  

def predictSVM(SVM, predict):
    return SVM.predict(predict)

def trainGBayes(x,y):
    GBayes = GaussianNB()
    GBayes.fit(x, y)
    return GBayes

def predictGBayes(GBayes, predict):
    return GBayes.predict(predict)

main()