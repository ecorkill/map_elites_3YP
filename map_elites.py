import numpy as np
import math
import random

from model import *
from fitnessMaps import *

MAX_EVAL = 3750000 #max number of evaluations
NUM_BINS = 3125 #number of bins
START_POP = 750 #starting population
BATCH_SZ = 75 #size of each batch
ALPHA = 0.2 #constant used in crossover to generate offspring
#length of arm is 8.125 fully stretched
SAMPLE = 100 #the sample period
SEED = 8 #np.ramdom seed
RADIUS = -0.01 #radius around goal point for assessing solutions

#function to generate the population
def generate():
    goal = np.array([[0.7], [0.7], [1]]) #target point
    startBase = np.array([[0.6], [0.3], [0]]) #base start position
    startEnd = np.array([[1.0264], [0.4369], [0.3177]]) #end effector start position

    #archive to store fitest individuals
    archive = {}
    #number of evaluations
    random.seed(SEED)
    seeds = [random.randint(0, 200000) for iter in range(40000)]
    numEval = 0
    c = 0
    seedIndex = 0
    generations = 0
    avgFitList = []
    bestFitList = []
    coverList = []
    while numEval < MAX_EVAL:
        #list of individuals to evaluate fitness
        to_eval = []
        #check that the inital population has been generated
        if len(archive) < START_POP:
            #generate the initial population
            i = 0
            c = 0
            #create list of seeds to iterate through to generate the random initial population
            random.seed(seeds[seedIndex])
            stepSeeds = [random.randint(0, 200000) for iter in range(60000)]
            while i < BATCH_SZ:
                #generate a matrix of random numbers for the joint angles
                z = 0
                ind = np.empty((11,1))
                while z < 7:
                    np.random.seed(stepSeeds[c])
                    randNum = (np.random.uniform(-5, 5, size=1)).round(8)
                    validJoint = testIndividual(randNum, False, goal, True, z)
                    if validJoint:
                        ind[z] = randNum
                        z += 1
                    #endif
                    c += 1
                #endloop

                #generate a matrix of random numbers for the base position, base orientation and torso height
                np.random.seed(stepSeeds[c])
                ind[7] = (np.random.uniform(0, 2, size=1)).round(8)
                c += 1
                np.random.seed(stepSeeds[c])
                ind[8] = (np.random.uniform(0, 2, size=1)).round(8)
                c += 1
                np.random.seed(stepSeeds[c])
                ind[9] = (np.random.uniform(0, 6.283, size=1)).round(8)
                c += 1
                np.random.seed(stepSeeds[c])
                ind[10] = (np.random.uniform(0, 0.35, size=1)).round(8)
                c += 1

                #check that the generated individual is valid
                valid, endPosition = testIndividual(ind, False, goal, False, 0)
                if valid:
                    print("Valid start individual: ")
                    print("archive length: ", len(archive))
                    i += 1
                    to_eval.append((ind, endPosition))
                #endif
            #endloop
            seedIndex += 1

        else:
            #parent selection and variation
            pIdentifier = list(archive.keys())
            #generate two arrays of random numbers that decide sets of parents
            randList1 = np.random.randint(0, len(pIdentifier), size=BATCH_SZ*3)
            randList2 = np.random.randint(0, len(pIdentifier), size=BATCH_SZ*3)
            i = 0
            #variable added so child is only mutated twice to try and achieve a valid solution
            mutated = 0
            while i < BATCH_SZ:
                parent1 = archive[pIdentifier[randList1[i]]][0]
                parent2 = archive[pIdentifier[randList2[i]]][0]
                child = generateChild(parent1, parent2)
                mutated += 1

                #check that the generated individual is valid
                valid, endPosition = testIndividual(child, False, goal, False, 0)
                if valid:
                    mutated = 0
                    i += 1
                    #send individual to physics model
                    to_eval.append((child, endPosition))
                elif mutated > 2:
                    mutated = 0
                    i += 1
                #endif
            #endloop
            generations += 1
            print("Generation: ", generations, ". Number of Evaluations: ", numEval)
        #endif

        fitnessList = []
        #calculate the fitness of all the individuals in to_eval
        for i in range(len(to_eval)):
            fitness = calculateFitness(to_eval[i], goal, startBase, startEnd)
            fitnessList.append((to_eval[i][0], to_eval[i][1], fitness))
        #endloop

        #add the contents of to_eval to archive, if the fitness of an individual is greater than that currently in the bin add to the archive
        for i in range(len(fitnessList)):
            bin, dimensions = determineBin(fitnessList[i][0], fitnessList[i][1])
            extend = list(fitnessList[i])
            extend.append(dimensions)
            fitnessList[i] = tuple(extend)
            addToArchive(fitnessList[i], bin, archive)
        #endloop
        if generations != 0:
            numEval += len(to_eval)
        #endif

        #sample and save the current population after every SAMPLE generations
        if (generations%SAMPLE == 0) and (generations != 0):
            avgFitList, bestFitList, coverList = sample(archive, avgFitList, bestFitList, coverList, SEED)
        #endif
    #endloop

    #sample at the end
    avgFitList, bestFitList, coverList = sample(archive, avgFitList, bestFitList, coverList, SEED)
    #send final set of policies to model
    pIdentifier = list(archive.keys())
    print("number of filled bins: ", len(pIdentifier))
    for i in range(0, len(pIdentifier), 200):
        print("i: ", i)
        print("fitness: ", archive[pIdentifier[i]][2])
        testIndividual(archive[pIdentifier[i]][0], True, goal, False, 0)
        print(archive[pIdentifier[i]][0])
    #endloop

    #plot fitness map
    generateMap(archive)
    #plot the avg fitness, best fitness and coverage over generateCuboidCoordinates
    plotMeasures(avgFitList, bestFitList, coverList, SAMPLE)
    print("Suitable solutions: ", assessMap(archive, RADIUS))
    writeMapFile(archive, "FINALMAPS2/finalMapSeed"+str(SEED)+".txt")

#function to generate the offspring from two parents
def generateChild(parent1, parent2):
    #use "whole" arithmetic crossover to generate offspring
    child = np.empty((11,1))
    i = 0
    while i < 11:
        p1 = parent1[i][0]
        p2 = parent2[i][0]
        res = (ALPHA*p1 + (1-ALPHA)*p2).round(4)
        child[i] = [res]
        i += 1
    #endloop
    #mutate a genome
    mutate = np.random.randint(1, 40, size=1)
    if mutate[0] < 29:
        element = np.random.randint(0, 10, size=1)
        if 0 <= element[0] < 7:
            row = random.randint(0, 6)
            newValue = np.random.randint(-5, 5, size=1)
            child[row] = newValue[0]
        elif element[0] == 7 or element[0] == 8:
            newPos = random.randint(0, 1)
            newValue = np.random.randint(0, 2, size=1)
            child[7+newPos] = newValue
        elif element[0] == 9:
            newValue = np.random.uniform(0, 6.283, size=1)
            child[9] = newValue.round(8)
        elif element[0] == 10:
            newValue = np.random.uniform(0, 0.35, size=1)
            child[10] = newValue.round(8)
    #endif

    return child

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

#function to calculate the fitness of an individual
def calculateFitness(ind, goal, startB, startE):
    dGoalE = findDistance(ind[1], goal, 3)
    fitness = -1*dGoalE
    return round(fitness, 6)

#function to normalise input value
def normalise(value, min, max):
    normaliseVal = (value-min)/(max-min)
    return round(normaliseVal, 8)

def splitOrientation(angle):
    return math.sin(angle), math.cos(angle)

#function to calculate the variance between policy angles
def calculateVariance(angles):
    mean = (angles[0][0]+angles[1][0]+angles[2][0]+angles[3][0]+angles[4][0]+angles[5][0]+angles[6][0])/7
    total = 0
    for i in range(len(angles)):
        total += pow((angles[i][0]-mean), 2)
    #endloop
    variance = total/(len(angles)-1)
    return variance

#function to determine what bin an individual belongs to
def determineBin(ind, endPoint):
    #dim2=arm extension, dim4=variance of joint angles, dim1=shoulder joint angle
    #dim3=elbow joint angle, dim5=wrist joint angle
    torso = np.array([[ind[7][0]], [ind[8][0]], [0]])
    distanceTA = findDistance(torso, endPoint, 2)
    distanceTA -= 0.13693
    dim2 = normalise(abs(distanceTA), 0.2, 0.78)
    var = calculateVariance(ind)
    dim4 = normalise(var, 1, 8.5)
    dim1 = normalise(ind[1][0], -3.142, -0.480)
    dim3 = normalise(ind[3][0], -1.964, 0.785)
    dim5 = normalise(ind[5][0], -1.414, 1.414)
    dimensions = (dim1, dim2, dim3, dim4, dim5)

    bin = []
    for i in dimensions:
        if 0<=i<0.2:
            bin.append(0)
        elif 0.2<=i<0.4:
            bin.append(1)
        elif 0.4<=i<0.6:
            bin.append(2)
        elif 0.6<=i<0.8:
            bin.append(3)
        else:
            bin.append(4)
        #endif
    #endloop

    return (bin[0], bin[1], bin[2], bin[3], bin[4]), np.array([[dim1], [dim2], [dim3], [dim4], [dim5]])
    #return (bin[0], bin[1], bin[2], bin[3], bin[4], bin[5])

#function to add an individual to the archive if its fitness is greater than the individual currently in the bin
def addToArchive(ind, bin, archive):
    #ind contains the policy, endPosition and fitness
    if bin in archive:
        if ind[2] > archive[bin][2]:
            archive[bin] = ind
        #endif
    else:
        archive[bin] = ind
    #endif

#main
generate()
