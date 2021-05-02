import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math

from measures import *
from model import *
from fitnessMaps import *
from animate import *

RADIUS = -0.03 #radius around goal point for assessing solutions
N_BINS = 10

#function to add damage to a joint of the robot
def addDamage(damageKind, angles, policy):
    # damageKind=0: joint is stuck at an a certain angle
    # damageKind=1: joint has an offset of angle
    for i in range(7):
        if damageKind[i] == 0:
            policy[i] = angles[i]
        elif damageKind[i] == 1:
            policy[i] += angles[i]
        # endif
    # endloop
    return policy

def splitEntry(line):
    key = tuple(map(str, line.split(' - ')))
    policy = tuple(map(float, key[0][1:len(key[0])-1].split(', ')))
    fitness = tuple(map(float, key[1][1:len(key[1])-1].split(', ')))
    dimensions = tuple(map(float, key[2][1:len(key[2])-2].split(', ')))

    return policy, fitness[0], dimensions

#function to read final behaviour map from file
def readMap():
    archive = {}
    file = open("FINALMAPS2/finalMapSeed8.txt", "r")
    lines = file.readlines()
    for x in lines:
        index = x[1:14]
        key = tuple(map(int, index.split(', ')))
        policy, fitness, dimensions = splitEntry(x[16:])
        policy = np.array([[policy[0]], [policy[1]], [policy[2]], [policy[3]], [policy[4]], [policy[5]], [policy[6]], [policy[7]], [policy[8]], [policy[9]], [policy[10]]])
        dimensions = np.array([dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4]])
        archive[key] = [[policy], [fitness], [dimensions]]
    #endloop
    file.close()
    return archive

#find the distance between two points
def findDistance(pointA, pointB, dimensions):
    if dimensions == 3:
        x = pow(pointB[0][0]-pointA[0][0], 2)
        y = pow(pointB[1][0]-pointA[1][0], 2)
        z = pow(pointB[2][0]-pointA[2][0], 2)
        distance = math.sqrt(x+y+z)
        return distance
    elif dimensions == 2:
        x = pow(pointB[0][0]-pointA[0][0], 2)
        y = pow(pointB[1][0]-pointA[1][0], 2)
        distance = math.sqrt(x+y)
        return distance
    #endif

#function to calculate the fitness
def calculateFitness(ind, goal):
    dGoalE = findDistance(ind, goal, 3)
    fitness = -1*dGoalE
    return round(fitness, 6)

#function to plot a histogram of the fitnesses over the map
def createHistogram(archive, index, title):
    fitnessList = []
    #create a list containing all the fitnesses from archive
    pIdentifier = list(archive.keys())
    if index == 2:
        for i in range(len(pIdentifier)):
            fitnessList.append(archive[pIdentifier[i]][index])
        #endloop
    else:
        for i in range(len(pIdentifier)):
            fitnessList.append(archive[pIdentifier[i]][index][0])
        #endloop
    #endif

    #plot histogram
    fig, ax = plt.subplots()
    ax.hist(fitnessList, bins=N_BINS, color='teal')
    ax.set_title(title)
    ax.set_xlabel("Fitness")
    ax.set_ylabel("Frequency")
    plt.show()

#function to add damage to a final map and check the suitability of solutions
def assess():
    goal = np.array([[0.8], [0.7], [1]]) #target point
    damageAngle = [1, 1, 1, 1, 1, 1, 1]
    angles = [0, 0, 0, -0.785, 0, 0, 0]

    archive = readMap()
    createHistogram(archive, 1, "Original Map")

    suitable = 0
    pIdentifier = list(archive.keys())
    for i in range(len(pIdentifier)):
        if archive[pIdentifier[i]][1][0] > RADIUS:
            suitable += 1
        #endif
    #endloop
    print("Suitable solutions: ", suitable)

    #for each individual in map add damamge
    pIdentifier = list(archive.keys())
    #for each individual in the final map add damage and recalculate fitness
    for i in range(len(pIdentifier)):
        archive[pIdentifier[i]][0][0] = addDamage(damageAngle, angles, archive[pIdentifier[i]][0][0])
        valid, endPoint = testIndividual(archive[pIdentifier[i]][0][0], False, goal, False, 0)
        #if a valid policy determine fitness. If not set fitness to NaN
        if valid:
            fitness = calculateFitness(endPoint, goal)
        else:
            fitness = math.nan
        #endif
        archive[pIdentifier[i]] = [archive[pIdentifier[i]][0][0], archive[pIdentifier[i]][1][0], fitness, archive[pIdentifier[i]][2][0]]
    #endloop

    for i in range(0, len(pIdentifier), 500):
        print("i: ", i)
        print("fitness: ", archive[pIdentifier[i]])
        animate(archive[pIdentifier[i]][0])
        print(archive[pIdentifier[i]][0])
    #endloop

    #create updated histogram
    createHistogram(archive, 2, "Amended Map")
    #plot fitness map
    generateMap(archive)
    #print map measurements
    print("Suitable solutions: ", assessMap(archive, RADIUS))

#run assessMap
assess()
