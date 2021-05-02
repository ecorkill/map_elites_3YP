
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

# function plots the robot on a 3D axis
def drawRobot(torso, arm, base):
    fig = pt.figure()
    axis = Axes3D(fig)
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.set_zlim(0, 2)

    robotTop = torso.top + torso.extension
    # the matrix used to adjust the joint positions the the correct place in relation to the base and the torso
    adjustPoints = np.array([[0], [0.13693], [robotTop-0.2317]])
    # rotation matrix used for rotating the arm and torso with the base
    rotMatrix = np.array([[math.cos(torso.rotation), -math.sin(torso.rotation), 0],
                      [math.sin(torso.rotation), math.cos(torso.rotation), 0],
                      [0, 0, 1]])
    adjustPoints = (rotMatrix.dot(adjustPoints)).round(4)

    print(arm.positions)
    # create 3 arrays containing the x, y, z coordinates
    xCoord = [torso.centre[0][0] + adjustPoints[0][0]]
    yCoord = [torso.centre[1][0] + adjustPoints[1][0]]
    zCoord = [adjustPoints[2][0]]
    for i in arm.positions:
        xCoord.append(i[0][0])
        yCoord.append(i[1][0])
        zCoord.append(i[2][0])
    # endloop

    # matrix that specifies the corners that make each cuboid surface
    corners1 = [[0,1,2,3], [3,6,4,0], [2,3,6,7], [1,2,7,5], [0,1,5,4], [4,5,7,6]]
    corners2 = [[0,1,2,3], [3,7,4,0], [2,3,7,6], [1,2,6,5], [0,1,5,4], [4,5,6,7]]

    surfaceArray1 = generateCuboidCoordinates(torso.corners1, corners1)
    surfaceArray2 = generateCuboidCoordinates(torso.corners2, corners2)
    surfaceArray3 = generateCuboidCoordinates(base.cuboidCorners, corners1)

    # add the robot arm links and joints to the figure
    axis.plot(xCoord, yCoord, zCoord, color='black')
    axis.scatter(xCoord, yCoord, zCoord, color='darkslategray')
    axis.scatter(0.7, 0.7, 1, color='red')
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
    pt.show()

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

# function to verify that all policy angles fall within joint angle contraints
def verifyAngles(joints, policy ):
    valid = False
    if len(policy)-1 == len(joints):
        c = 0
        for i in range(len(policy)-1):
            joint = policy[i]
            if joints[c].getLowerLimit() <= joint[0] <= joints[c].getUpperLimit():
                valid = True
            else:
                #print("invalid joint: ", i+1)
                valid = False
                break
            # endif
            c += 1
        # endloop
    # endif
    return valid

# function to verify that the torso joint extension is within the acceptable range
def checkExtension(joint, policy):
    extension = policy[len(policy)-1]
    if joint.getLowerLimit() <= extension[3] <= joint.getUpperLimit():
        return True
    else:
        return False
    # endif

# function to find the point of intersection of the two lines
def findIntersection(lineA, lineB):
    # solve simultaneous equations to find value of variables for when lines intersect
    m1 = np.array([[lineA[2][0][0], -lineB[2][0][0]], [lineA[2][1][0], -lineB[2][1][0]]])
    m2 = np.array([[lineB[0][0][0]-lineA[0][0][0]], [lineB[0][1][0]-lineA[0][1][0]]])
    # if the determinant is 0 then the lines overlap (not parallel as already determined there's a point of intersection)
    if np.linalg.det(m1) == 0:
        return False
    # endif
    ans = np.linalg.solve(m1, m2)

    pointA = lineA[0][2]+ans[0]*lineA[2][2]
    pointB = lineB[0][2]+ans[1]*lineB[2][2]
    # check variables calculated correctly in the final equation
    if pointA[0].round(4) != pointB[0].round(4):
        #print("Error: simultaneous equations not solved correctly")
        return True
    else:
        # find point of intersection
        x = lineA[0][0][0]+lineA[2][0][0]*ans[0]
        y = lineA[0][1][0]+lineA[2][1][0]*ans[0]
        z = lineA[0][2][0]+lineA[2][2][0]*ans[0]
        intPoint = np.array([[x], [y], [z]]).round(4)
        # if the intersection is between the start and end points of lineA then there is a collision
        if (lineA[0][0] < intPoint[0] < lineA[1][0]) and (lineA[0][1] < intPoint[1] < lineA[1][1]) and (lineA[0][2] < intPoint[2] < lineA[1][2]):
            collision = True
        else:
            collision = False
        # endif
        return collision
    # endif

# function to check that there are no collisions between any links of arm
def collisionCheckArm(arm, lines):
    positions = arm.getPositions()
    i = 0
    # create an array containg vector line equation variables
    for i in range(len(positions)):
        if i == 0:
            base = np.array([[0], [0], [0]])
            direction = positions[i] - base
            lines.append(np.array([base, positions[i], direction]))
        else:
            direction = positions[i] - positions[i-1]
            lines.append(np.array([positions[i-1], positions[i], direction]))
        # endif
    # endloop

    # check for intersection between lines. intersect is 0 if the lines intersect
    for i in range(0, 6):
        for c in range(i+1, 7):
            crossP = np.cross(lines[i][2], lines[c][2], axis=0)
            sub = lines[i][0]-lines[c][0]
            intersect = crossP[0]*sub[0] + crossP[1]*sub[1] + crossP[2]*sub[2]
            if intersect == 0:
                collision = findIntersection(lines[i], lines[c])
                if collision:
                    print("Collision between links")
                    break
                # endif
            # endif
        # endloop
    # endloop
    return collision

# function to check that there are no collisions between the robot arm and torso and/or the robot arm and base
def collisionCheckTorsoBase(arm, torso, base):
    # index 3 contains upper bounds of axis ranges
    # index 5 contains the lower bounds of axis ranges
    upper = [torso.corners1[3], torso.corners2[3], base.cuboidCorners[3]]
    lower = [torso.corners1[5], torso.corners2[5], base.cuboidCorners[5]]
    joints = []
    linkPoints = []
    steps = np.linspace(0, 1, 11)
    valid = True

    for i in range(0, 7, 2):
        joints.append(arm.positions[i])
    # endloop
    # check for collisions between the arm and the 3 body cuboids
    for cuboid in range(len(upper)):
        for link in range(len(joints)-1):
            for c in range(len(steps)):
                x = (joints[link][0][0]+(joints[link+1][0][0]-joints[link][0][0])*steps[c]).round(4)
                y = (joints[link][1][0]+(joints[link+1][1][0]-joints[link][1][0])*steps[c]).round(4)
                z = (joints[link][2][0]+(joints[link+1][2][0]-joints[link][2][0])*steps[c]).round(4)
                if cuboid == 0:
                    linkPoints.append(np.array([[x], [y], [z]]))
                # endif
                # check if the intermediate point is in the cuboid
                if upper[cuboid][0]>=x>=lower[cuboid][0] and upper[cuboid][1]>=y>=lower[cuboid][1] and upper[cuboid][2]>=z>=lower[cuboid][2]:
                    valid = False
                    break
                # endif
            # endloop
        # endloop
    # endloop

    # check for a collision between the arm and the base cylinder if there is no collision between arm and body cuboids
    if valid:
        for points in linkPoints:
            if (pow(points[0][0]-base.baseCentre[0][0], 2)+pow(points[1][0]-base.baseCentre[1][0], 2))<=0.729 and 0.3>=points[2][0]>=0:
                valid = False
                break
            # endif
        # endloop
    # endif
    return valid

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

#add damage to a joint of the robot to check that the motion is correct
def addDamage(joints, damageKind, angles, policy):
    #damageKind=0: joint is stuck at an a certain angle
    #damageKind=1: joint has an offset of angle
    if len(joints) == len(angles):
        for i in range(0, len(joints)):
            if damageKind[i] == 0:
                policy[joints[i]-1][0] = angles[i]
            elif damageKind[i] == 1:
                policy[joints[i]-1][0] += angles[i]
            #endif
        #endloop
        return policy
    else:
        print("Joints and angles don't match")
    #endif

def main():
    #create robot links and joints
    #form segment(jointType, length, upperLimit, lowerLimit)
    #jointType 0 is revolute, 3 is segment/link
    l1 = Segment(3, 0.1, 0, 0)
    l2 = Segment(3, 0.082, 0, 0)
    l3 = Segment(3, 0.23, 0, 0)
    l4 = Segment(3, 0.192, 0, 0)
    l5 = Segment(3, 0.14, 0, 0)
    l6 = Segment(3, 0.068, 0, 0)
    l7 = Segment(3, 0, 0, 0)

    j1 = Segment(0, 0, 2.749, 0)
    j2 = Segment(0, 0, -0.480, -3.142)
    j3 = Segment(0, 0, 4.713, -0.392)
    j4 = Segment(0, 0, 0.785, -1.964)
    j5 = Segment(0, 0, -1.048, -5.236)
    j6 = Segment(0, 0, 1.414, -1.414)
    j7 = Segment(0, 0, 2.094, -2.094)
    torsoJ = Segment(1, 0, 0.35, 0)

    #matrix: [[theta1, d1, a1, alpha1], ..., [theta7, d7, a7, alpha7]]
    #bottom row of matrix: [xpos, ypos, theta, extension]
    policy = np.array([[1.1449, 0, 0.1, 1.5708],
                      [-1.6118, 0, 0, -1.5708],
                      [1, 0.3127, 0, 1.5708],
                      [0.2844, 0, 0, -1.5708],
                      [-1.9505, 0.3318, 0, 1.5708],
                      [-0.3862, 0, 0, 1.5708],
                      [0.2036, 0.068, 0, 0],
                      [0.9842, 1.4658, 3.1112, 0.2641]])

    links = [l1, l2, l3, l4, l5, l6, l7]
    joints = [j1, j2, j3, j4, j5, j6, j7]
    endPoint = np.array([[], [], []])
    lines =[]
    np.set_printoptions(threshold=np.inf)

    #damageJoints = [1, 4]
    #damageKind = [0, 1]
    #angles = [0, -1.6]
    #policy = addDamage(damageJoints, damageKind, angles, policy)

    valid = verifyAngles(joints, policy)
    if valid:
        arm = RobotArm(links, joints, policy)
        print(arm.positions)
        print("calculated end: ", arm.getPosEnd())
        collision = collisionCheckArm(arm, lines)
        if collision:
            print("Policy invalid - collision occured between joints")
        else:
            validTorso = checkExtension(torsoJ, policy)
            if validTorso:
                torso = RobotTorso(0.89, torsoJ, policy)
                print("torso corners: ", torso.corners1)
                base = RobotBase(torso.centre, torso.rotation)
                adjustArm(torso, arm)
                noCollision = collisionCheckTorsoBase(arm, torso, base)
                if noCollision:
                    drawRobot(torso, arm, base)
                else:
                    print("Collision between arm and torso/base")
                #endif
            else:
                print("Policy invalid - incorrect torso extension")
            #endif
        #endif
        #else:
            #print("Policy invalid - incorrect end position reached")
        #endif
    else:
        print("Policy invalid - joint angles not within contraints")
    #endif

#call main function
main()
