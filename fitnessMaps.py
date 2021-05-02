import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm

from measures import *

#function to generate a fitness map
def generateMap(archive):
    #update file containing all the best fitnesses for each bin in the map
    bestBins = readFile("fitnessRecords2.txt")
    bestBins = updateBestBins(archive, bestBins)
    writeFile(bestBins, "fitnessRecords2.txt")

    #calculate fitness map measurements
    print("-------------------------------------")
    performance = calPerformance(archive, bestBins)
    reliability = calReliability(archive, bestBins)
    precision = calPrecision(archive, bestBins)
    cover = coverage(archive)
    print(performance)
    print(reliability)
    print(precision)
    print(cover)
    print("-------------------------------------")

    #generate the x,y,z coordinates
    keys = list(archive.keys())
    xCoord, yCoord = np.mgrid[slice(0, 1 + 0.008, 0.008),
                    slice(0, 1 + 0.04, 0.04)]

    zCoord = np.nan * np.empty((125, 25))
    for i in range(len(keys)):
        xIndex = keys[i][0]*25 + keys[i][2]*5 + keys[i][4]
        yIndex = keys[i][1]*5 + keys[i][3]
        zCoord[xIndex][yIndex] = archive[keys[i]][2]
    #endloop
    #plot the fitness map
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xCoord, yCoord, zCoord, cmap='magma', vmin=-2.5, vmax=0)
    ax.set_title('Fitness Map')
    ax.axis([0, 1, 0, 1])
    fig.colorbar(c, ax=ax)
    plt.show()

#function to plot the average fitness, best fitness and coverage over generations
def plotMeasures(avgList, bestList, cover, period):
    #create x coordinates
    xCoord = []
    for i in range(len(avgList)):
        xCoord.append(i*period)
    #endloop

    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.tight_layout()
    ax = plt.subplot(311)
    ax.plot(xCoord, avgList, color='teal')
    ax.set_title("Average Fitness over Generations")
    ax = plt.subplot(312)
    ax.plot(xCoord, bestList, color='teal')
    ax.set_title("Best Fitness over Generations")
    ax = plt.subplot(313)
    ax.plot(xCoord, cover, color='teal')
    ax.set_title("Coverage over Generations")
    plt.show()

#function that takes a sample of the current population
def sample(archive, avgFitList, bestFitList, coverList, seed):
    bestBins = readFile("fitnessRecords2.txt")
    bestBins = updateBestBins(archive, bestBins)

    #calculate map related measurements
    avgFitness = averageFitness(archive)
    bestFitness = topFitness(archive)
    cover = coverage(archive)
    performance = calPerformance(archive, bestBins)
    reliability = calReliability(archive, bestBins)
    precision = calPrecision(archive, bestBins)

    avgFitList.append(avgFitness)
    bestFitList.append(bestFitness)
    coverList.append(cover)

    #filename that the sample data is saved to
    filename = "FINAL2/Seed"+str(seed)+"DimSet2FitFunc1.txt"

    file = open(filename, "a+")
    file.write(str(avgFitness))
    file.write("\n")
    file.write(str(bestFitness))
    file.write("\n")
    file.write("%s %s %s %s" % (performance, reliability, precision, cover))
    file.write("\n")
    file.write("Population: \n")

    #write the contents of archive to the filename
    keys = list(archive.keys())
    for i in range(len(keys)):
        for c in range(len(archive[keys[i]])):
            if isinstance(archive[keys[i]][c], float):
                file.write("%s " % str(archive[keys[i]][c]))
            else:
                for element in range(len(archive[keys[i]][c])):
                    file.write("%s " % str(archive[keys[i]][c][element]))
                #endloop
            #endif
        #endloop
        file.write("\n")
        file.write("\n")
    #endloop
    file.close()

    return avgFitList, bestFitList, coverList
