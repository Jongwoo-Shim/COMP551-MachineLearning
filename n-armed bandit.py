import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt


SLOTS = 10
TRIALS = 500
arms = np.random.rand(SLOTS)

av = np.ones(SLOTS)
counts = np.zeros(SLOTS)

#Create a matrix for softmax stored probability ranks
#Initializes them all to have the same probability
av_softmax = np.zeros(SLOTS)
av_softmax[:] = 1/SLOTS
tau = 1.12

def reward(prob):
    reward = 0
    for i in range(SLOTS):
        if random.random() < prob:
            reward += 1
    return reward

def softmax(av):
    probs = np.zeros(SLOTS)
    for i in range(SLOTS):
        softm = (np.exp(av[i]/tau)/np.sum(np.exp(av[:]/tau)))
        probs[i] = softm
    return probs

def main():
    plt.xlabel("Plays")
    plt.ylabel("Average Reward")
    for i in range(TRIALS):
        # Greedy, picking the arm that has had the best result
        choice = np.where(arms ==np.random.choice(arms, p = av_softmax))[0][0]
        counts[choice] += 1
        k = counts[choice]
        rwd = reward(arms[choice])
        old_avg = av[choice]
        new_avg = old_avg + (1/k)*(rwd-old_avg)
        av[choice] = new_avg
        av_softmax = softmax(av)
        runningMean = np.average(av,weights = np.array(counts[j]/np.sum(counts) for j in range(len(counts))))
    plt.scatter(i,runningMean)
main()