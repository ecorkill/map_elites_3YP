import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math

NUM_BINS = 3125 #number of bins

#function to read from the stored fitnesses
def readFile(filename):
    bestBins = {}
    file = open(filename, "r")
    lines = file.readlines()
    for x in lines:
        index = x[1:14]
        key = tuple(map(int, index.split(', ')))
        fitness = float(x[16:])
        bestBins[key] = fitness
    #endloop
    file.close()
    return bestBins

#function to update the stored fitnesses
def updateBestBins(archive, bestBins):
    pIdentifier = list(archive.keys())
    for i in range(len(pIdentifier)):
        if pIdentifier[i] in bestBins:
            if archive[pIdentifier[i]][2] > bestBins[pIdentifier[i]]:
                print("updated bestBins")
                bestBins[pIdentifier[i]] = archive[pIdentifier[i]][2]
            #endif
        else:
            bestBins[pIdentifier[i]] = archive[pIdentifier[i]][2]
        #endif
    #endloop
    return bestBins

#function to build the entry to the file
def constructFileEntry(bin, fitness):
    strBin = "("+str(bin[0])+", "+str(bin[1])+", "+str(bin[2])+", "+str(bin[3])+", "+str(bin[4])+")"
    entry = strBin + " " + str(fitness) + "\n"
    return entry

#function to write back the update fitnesses to the file
def writeFile(bestBins, filename):
    file = open(filename, "w")
    pIdentifier = list(bestBins.keys())
    for i in range(len(pIdentifier)):
        entry = constructFileEntry(pIdentifier[i], bestBins[pIdentifier[i]])
        file.write(entry)
    #endloop

#function to calculate the global performance of the fitness map
def calPerformance(archive, bestBins):
    bestKey = max(bestBins, key=bestBins.get)
    archiveVal = -2.5

    pIdentifier = list(archive.keys())
    for i in range(len(pIdentifier)):
        if archive[pIdentifier[i]][2] > archiveVal:
            archiveVal = archive[pIdentifier[i]][2]
        #endif
    #endloop

    globalP = round(bestBins[bestKey]/archiveVal, 6)
    return globalP

#function to calculate the global reliability of the fitness map
def calReliability(archive, bestBins):
    length = len(bestBins)
    rel = 0
    bIden = list(bestBins.keys())
    for i in range(len(bIden)):
        if (bIden[i] in archive) and (not math.isnan(archive[bIden[i]][2])):
            rel += (bestBins[bIden[i]]/archive[bIden[i]][2])
        #endif
    #endloop
    rel = round(rel/length, 6)
    return rel

#function to calculate the precision of the fitness map
def calPrecision(archive, bestBins):
    length = len(archive)
    prec = 0
    bIden = list(bestBins.keys())
    for i in range(len(bIden)):
        if (bIden[i] in archive) and (not math.isnan(archive[bIden[i]][2])):
            prec += (bestBins[bIden[i]]/archive[bIden[i]][2])
        #endif
    #endloop
    prec = round(prec/length, 6)
    return prec

#function to calculate the coverage of the fitness map
def coverage(archive):
    filledBins = 0
    bIden = list(archive.keys())
    for i in range(len(bIden)):
        if not (math.isnan(archive[bIden[i]][2])):
            filledBins += 1
        #endif
    #endloop
    cover = filledBins/NUM_BINS
    return cover

#function to calculate the average fitness in a population
def averageFitness(archive):
    length = len(archive)
    total = 0
    pIdentifier = list(archive.keys())
    for i in range(len(pIdentifier)):
        total += archive[pIdentifier[i]][2]
    #endloop
    avg = round(total/length, 6)
    return avg

#function to find the best fitness in a population
def topFitness(archive):
    topVal = -2.5
    pIdentifier = list(archive.keys())
    for i in range(len(pIdentifier)):
        if archive[pIdentifier[i]][2] > topVal:
            topVal = archive[pIdentifier[i]][2]
        #endif
    #endloop
    return round(topVal, 6)

def assessMap(archive, radius):
    suitable = 0
    pIdentifier = list(archive.keys())
    for i in range(len(pIdentifier)):
        if archive[pIdentifier[i]][2] > radius:
            suitable += 1
        #endif
    #endloop
    return suitable

#function to right the final map to a file
def writeMapFile(archive, filename):
    file = open(filename, "w")
    pIdentifier = list(archive.keys())
    for i in range(len(pIdentifier)):
        policy = archive[pIdentifier[i]][0]
        policy = tuple((policy[0][0], policy[1][0], policy[2][0], policy[3][0], policy[4][0], policy[5][0], policy[6][0], policy[7][0], policy[8][0], policy[9][0], policy[10][0]))
        fitness = archive[pIdentifier[i]][2]
        dimensions = archive[pIdentifier[i]][3]
        dimensions = tuple((dimensions[0][0], dimensions[1][0], dimensions[2][0], dimensions[3][0], dimensions[4][0]))
        entry = str(policy)+ " - ("+str(fitness)+") - "+str(dimensions)
        entry = constructFileEntry(pIdentifier[i], entry)
        file.write(entry)
    #endloop
