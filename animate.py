from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pt
import matplotlib as mpl
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import math

from robotClasses import *

FRAMES = 40

# function plots the robot on a 3D axis
def drawRobot(torsoHeight, orientation, torso, x, y, z, base, goal):
    fig = pt.figure()
    axis = Axes3D(fig)
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.set_zlim(0, 2)

    # the matrix used to adjust the joint positions the the correct place in relation to the base and the torso
    adjustPoints = np.array([[0], [0.13693], [0.89+torsoHeight-0.2317]])
    # rotation matrix used for rotating the arm and torso with the base
    rotMatrix = np.array([[math.cos(orientation), -math.sin(orientation), 0],
                      [math.sin(orientation), math.cos(orientation), 0],
                      [0, 0, 1]])
    adjustPoints = (rotMatrix.dot(adjustPoints)).round(4)

    # create 3 arrays containing the x, y, z coordinates
    xCoord = [torso.centre[0][0] + adjustPoints[0][0]]
    yCoord = [torso.centre[1][0] + adjustPoints[1][0]]
    zCoord = [adjustPoints[2][0]]

    for i in range(len(x)):
        xCoord.append(x[i])
        yCoord.append(y[i])
        zCoord.append(z[i])
    #endfor

    # matrix that specifies the corners that make each cuboid surface
    corners1 = [[0,1,2,3], [3,6,4,0], [2,3,6,7], [1,2,7,5], [0,1,5,4], [4,5,7,6]]
    corners2 = [[0,1,2,3], [3,7,4,0], [2,3,7,6], [1,2,6,5], [0,1,5,4], [4,5,6,7]]

    surfaceArray1 = generateCuboidCoordinates(torso.corners1, corners1)
    surfaceArray2 = generateCuboidCoordinates(torso.corners2, corners2)
    surfaceArray3 = generateCuboidCoordinates(base.cuboidCorners, corners1)

    # add the robot arm links and joints to the figure
    axis.plot(xCoord, yCoord, zCoord, color='black')
    axis.scatter(xCoord, yCoord, zCoord, color='darkslategray')
    axis.scatter(goal[0], goal[1], goal[2], color='red')
    # add the cuboids to the figure
    axis.add_collection3d(Poly3DCollection(surfaceArray1, facecolors='bisque', edgecolors='black', alpha=0.7))
    axis.add_collection3d(Poly3DCollection(surfaceArray2, facecolors='bisque', edgecolors='black', alpha=0.7))
    # add base to the figure
    axis.plot_surface(base.cylinderX, base.cylinderY, base.cylinderZ, color='bisque', edgecolors='black', alpha=0.7, shade=False)
    p = Circle((base.baseCentre[0][0], base.baseCentre[1][0]), 0.27, facecolor='bisque', edgecolor='black', alpha=0.8)
    q = Circle((base.baseCentre[0][0], base.baseCentre[1][0]), 0.27, facecolor='bisque', edgecolor='black', alpha=0.8)
    axis.add_patch(p)
    axis.add_patch(q)
    art3d.pathpatch_2d_to_3d(p, z=0.3, zdir="z")
    art3d.pathpatch_2d_to_3d(q, z=0, zdir="z")
    # add the base cuboid to the figure
    axis.add_collection3d(Poly3DCollection(surfaceArray3, facecolors='bisque', edgecolors='black', alpha=0.7))

    axis.set_xlabel("x axis")
    axis.set_ylabel("y axis")
    axis.set_zlabel("z axis")
    pt.draw()
    pt.pause(0.00005)
    pt.close('all')

# function that generates the set of coordinates for all cuboid faces from the corner coordinates
def generateCuboidCoordinates(corners, cornersOrder):
    xCoord = []
    yCoord = []
    zCoord = []

    for i in corners:
        xCoord.append(i[0][0].round(4))
        yCoord.append(i[1][0].round(4))
        zCoord.append(i[2][0].round(4))
    # endloop

    # array containing all the coordinates of the cuboid corners
    coordinates = list(zip(xCoord, yCoord, zCoord))
    # creates a matrix of corner coordinates. Each row is one surface of the cuboid
    rec = [[coordinates[cornersOrder[x][y]] for y in range(len(cornersOrder[0]))] for x in range(len(cornersOrder))]
    return rec

# function that adds adapt the polict structure
def addConstants(policy):
    constantDH = np.array([[0, 0.1, 1.5708],
                          [0, 0, -1.5708],
                          [0.3127, 0, 1.5708],
                          [0, 0, -1.5708],
                          [0.3318, 0, 1.5708],
                          [0, 0, 1.5708],
                          [0.068, 0, 0]])

    jointA = np.array([policy[0], policy[1], policy[2], policy[3], policy[4], policy[5], policy[6]])
    base = np.array([policy[7], policy[8], policy[9], policy[10]])
    policy = np.hstack((jointA, constantDH))
    policy = np.vstack([policy, base.transpose()])

    return policy

# function to adjust the position of the arm to match the height of the robot torso and base
def adjustArm(torso, arm):
    robotTop = torso.top + torso.extension
    # the matrix used to adjust the joint positions the the correct place in relation to the base and the torso
    adjustPoints = np.array([[0], [0.13693], [robotTop-0.2317]])
    # rotation matrix used for rotating the arm and torso with the base
    rotMatrix = np.array([[math.cos(torso.rotation), -math.sin(torso.rotation), 0],
                      [math.sin(torso.rotation), math.cos(torso.rotation), 0],
                      [0, 0, 1]])
    adjustPoints = rotMatrix.dot(adjustPoints)

    for i in arm.positions:
        i[0][0] = (i[0][0]+torso.centre[0][0]+adjustPoints[0][0]).round(4)
        i[1][0] = (i[1][0]+torso.centre[1][0]+adjustPoints[1][0]).round(4)
        i[2][0] = (i[2][0]+adjustPoints[2][0]).round(4)
    # endloop

#function to calculate the distance between the start and end point for each joint
def difference(startx, starty, startz, arm):
    distances = []
    for i in range(len(arm.positions)):
        startP = np.array([[startx[i]], [starty[i]], [startz[i]]])
        goalP = arm.positions[i]
        distance = startP - goalP
        distances.append(distance)
    #endfor
    return distances

#function that plots the successive movement of robot
def plot(distances, height, startx, starty, startz, startPos, ori, extension, distanceBase, torsoJ, policy, goal):
    for i in range(FRAMES+1):
        div = i/FRAMES
        xCoord = []
        yCoord = []
        zCoord = []
        for c in range(len(distances)):
            difference = distances[c]
            xCoord.append((startx[c] - div*difference[0][0]).round(4))
            yCoord.append((starty[c] - div*difference[1][0]).round(4))
            zCoord.append((startz[c] - div*difference[2][0]).round(4))
        #endloop
        policy[7][3] = div*extension
        policy[7][0] = startPos[0] - div*distanceBase[0]
        policy[7][1] = startPos[1] - div*distanceBase[1]
        policy[7][2] = ori - div*distanceBase[2]
        torso = RobotTorso(0.89, torsoJ, policy)
        base = RobotBase(torso.centre, torso.rotation)
        drawRobot(policy[7][3], policy[7][2], torso, xCoord, yCoord, zCoord, base, goal)
    #endloop

def animate(policy):
    j1 = Segment(0, 0, 2.749, 0)
    j2 = Segment(0, 0, -0.480, -3.142)
    j3 = Segment(0, 0, 4.713, -0.392)
    j4 = Segment(0, 0, 0.785, -1.964)
    j5 = Segment(0, 0, -1.048, -5.236)
    j6 = Segment(0, 0, 1.414, -1.414)
    j7 = Segment(0, 0, 2.094, -2.094)
    torsoJ = Segment(1, 0, 0.35, 0)
    joints = [j1, j2, j3, j4, j5, j6, j7]
    links = []

    #start positions of points
    startx = [0.6, 0.7, 0.7, 1.0126, 1.0264, 1.0264, 0.9595]
    starty = [0.4369, 0.4369, 0.4369, 0.4368, 0.4369, 0.4369, 0.4369]
    startz = [0.6583, 0.6583, 0.6583, 0.6492, 0.6492, 0.3177, 0.3177]

    goal = np.array([[0.7], [0.7], [1]]) #target point
    startB = np.array([[0.6], [0.3], [0]]) #base start position
    startE = np.array([[1.0264], [0.4369], [0.3177]]) #end effector start position

    height = 0.89

    policy = addConstants(policy)
    arm = RobotArm(links, joints, policy)
    torso = RobotTorso(0.89, torsoJ, policy)
    base = RobotBase(torso.centre, torso.rotation)
    adjustArm(torso, arm)

    #find the distance between the start and end points of each joint
    distances = difference(startx, starty, startz, arm)
    distanceBase = []
    distanceBase.append((startB[0][0]-policy[7][0]).round(4))
    distanceBase.append((startB[1][0]-policy[7][1]).round(4))
    distanceBase.append((0-policy[7][2]).round(4))
    plot(distances, height, startx, starty, startz, startB, 0, policy[7][3], distanceBase, torsoJ, policy, goal)
