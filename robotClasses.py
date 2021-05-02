import numpy as np
import math

#class for defining links and joints of the robot arm
class Segment:
    def __init__(self, jointType, length, upperLimit, lowerLimit):
        #define all class variables
        self.jointType = jointType
        self.upperLimit = upperLimit
        self.lowerLimit = lowerLimit
        if jointType != 3 and length > 0:
            print("Error! Joint should have no length")
        else:
            self.length = length
        #endif

    def getUpperLimit(self):
        return self.upperLimit

    def getLowerLimit(self):
        return self.lowerLimit

    def getJointType(self):
        return self.jointType

#class for creating the arm from the DH method variables given in the policy
#class also calculates the joint positions and orientations and checks the final end effector position
class RobotArm:
    def __init__(self, links, joints, policy):
        #define all class variables
        self.numSegments = len(joints)
        self.posend = np.array([[0], [0], [0]])
        self.rotMatrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.trfMatrix = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        self.angles = []
        self.positions = []
        self.rotation = policy[7][2]

        c = 0
        for i in range(len(policy)-1):
            joint = policy[i]
            #transformation matrix from DH method
            trfMatrix = np.array([[math.cos(joint[0]), -math.sin(joint[0])*math.cos(joint[3]), math.sin(joint[0])*math.sin(joint[3]), joint[2]*math.cos(joint[0])],
                              [math.sin(joint[0]), math.cos(joint[0])*math.cos(joint[3]), -math.cos(joint[0])*math.sin(joint[3]), joint[2]*math.sin(joint[0])],
                              [0, math.sin(joint[3]), math.cos(joint[3]), joint[1]],
                              [0, 0, 0, 1]])

            rotMatrix = np.array([[math.cos(self.rotation), -math.sin(self.rotation), 0],
                              [math.sin(self.rotation), math.cos(self.rotation), 0],
                              [0, 0, 1]])

            #set the rotation matrix, new end efector position and joint posistion
            if joints[c].getJointType() == 0:
                if c == 0:
                    self.rotMatrix = trfMatrix[0:4, 0:4]
                    self.trfMatrix = trfMatrix.round(4)
                    self.posend = (rotMatrix.dot(self.trfMatrix[0:3, 3:])).round(4)
                    self.positions.append(self.posend)
                else:
                    self.trfMatrix = np.dot(self.trfMatrix, trfMatrix).round(4)
                    self.rotMatrix = self.trfMatrix[0:4, 0:4]
                    self.posend = (rotMatrix.dot(self.trfMatrix[0:3, 3:])).round(4)
                    self.positions.append(self.posend)
                #endif
            #endif
            c += 1
        #endloop

    def getPosEnd(self):
        return self.posend

    def getPositions(self):
        return self.positions

#class for creating the robot torso
#will draw the torso as simple cuboids and determine the starting position for the robot arm
class RobotTorso:
    def __init__(self, heightTop, joint, policy):
        self.centre = np.array([[policy[7][0]], [policy[7][1]], [0]])
        self.rotation = policy[7][2]
        self.extension = policy[7][3]
        self.top = heightTop
        self.corners1 = []
        self.corners2 = []
        self.calculateCorners()

    #determine all the corner positions from the centre point and orientation
    def calculateCorners(self):
        cuboidCorners1 = []
        cuboidCorners2 = []
        #centre point for the top rectangle of the torse
        newCentre1 = np.array([[self.centre[0][0]],
                              [self.centre[1][0]],
                              [self.top - 0.067 + (self.extension)]]) #0.067 is half the height of the top rectangle
        #centre point for the bottom rectangle of the torso
        newCentre2 = np.array([[self.centre[0][0]],
                              [self.centre[1][0]],
                              [self.top - 0.308 + (self.extension/2)]]) #0.308 is the height of the top rectangle + half the height of the second rectangle

        #calculate the corners of the top cuboid
        #length=3.2942 (x), width=2.7386 (y), height=1.3374 (z)
        cuboidCorners1.append(np.array([[-0.16471], [0.13693], [0.06687]])) #topleft
        cuboidCorners1.append(np.array([[-0.16471], [-0.13693], [0.06687]])) #bottomleft
        cuboidCorners1.append(np.array([[0.16471], [-0.13693], [0.06687]])) #topright
        cuboidCorners1.append(np.array([[0.16471], [0.13693], [0.06687]])) #bottomright

        cuboidCorners1.append(np.array([[-0.16471], [0.13693], [-0.06687]]))
        cuboidCorners1.append(np.array([[-0.16471], [-0.13693], [-0.06687]]))
        cuboidCorners1.append(np.array([[0.16471], [0.13693], [-0.06687]]))
        cuboidCorners1.append(np.array([[0.16471], [-0.13693], [-0.06687]]))

        #calculate the corners of the bottom cuboid
        #length=2.483 (x), width=2.246 (y), height=3.4934 (z)
        cuboidCorners2.append(np.array([[-0.12415], [0.1123], [0.17467 + (self.extension/2)]]))
        cuboidCorners2.append(np.array([[-0.12415], [-0.1123], [0.17467 + (self.extension/2)]]))
        cuboidCorners2.append(np.array([[0.12415], [-0.1123], [0.17467 + (self.extension/2)]]))
        cuboidCorners2.append(np.array([[0.12415], [0.1123], [0.17467 + (self.extension/2)]]))

        cuboidCorners2.append(np.array([[-0.12415], [0.1123], [-0.17467 - (self.extension/2)]]))
        cuboidCorners2.append(np.array([[-0.12415], [-0.1123], [-0.17467 - (self.extension/2)]]))
        cuboidCorners2.append(np.array([[0.12415], [-0.1123], [-0.17467 - (self.extension/2)]]))
        cuboidCorners2.append(np.array([[0.12415], [0.1123], [-0.17467 - (self.extension/2)]]))

        #rotation matrix used for rotating the arm and torso with the base
        rotMatrix = np.array([[math.cos(self.rotation), -math.sin(self.rotation), 0],
                          [math.sin(self.rotation), math.cos(self.rotation), 0],
                          [0, 0, 1]])

        #rotate the corners in relation to the rotation of the base
        for i in range(8):
            cuboidCorners1[i] = rotMatrix.dot(cuboidCorners1[i]).round(4)
            self.corners1.append(newCentre1 + cuboidCorners1[i])
            cuboidCorners2[i] = rotMatrix.dot(cuboidCorners2[i]).round(4)
            self.corners2.append(newCentre2 + cuboidCorners2[i])
        #endloop

class RobotBase:
    def __init__(self, torsoCentre, torsoRotation):
        self.baseCentre = torsoCentre
        self.rotation = torsoRotation
        self.cuboidCorners = self.createBaseCuboid()
        self.cylinderX, self.cylinderY, self.cylinderZ = self.createBaseCoordinates()

    #function to generate the coordinates for the cylinder base
    def createBaseCoordinates(self):
        var = np.linspace(0, 5.4*math.pi, 50)
        z = np.linspace(0, 0.3, 2)
        var, z = np.meshgrid(var, z)

        x = 0.27*np.cos(var) + self.baseCentre[0][0]
        y = 0.27*np.sin(var) + self.baseCentre[1][0]
        return x, y, z

    #function to create the corner coordinates for the cuboid that sits on top of the base
    def createBaseCuboid(self):
        cuboidCorners = []
        finalCorners = []
        #centre point for the top rectangle of the torse
        newCentre = np.array([[self.baseCentre[0][0]], [self.baseCentre[1][0]], [0.3535]])

        #calculate the corners of the top cuboid
        #length=2.4774 (x), width=4.0294 (y), height=1.07 (z)
        cuboidCorners.append(np.array([[0.12387], [-0.20147], [0.0535]])) #topleft
        cuboidCorners.append(np.array([[-0.12387], [-0.20147], [0.0535]])) #bottomleft
        cuboidCorners.append(np.array([[-0.12387], [0.20147], [0.0535]])) #topright
        cuboidCorners.append(np.array([[0.12387], [0.20147], [0.0535]])) #bottomright

        cuboidCorners.append(np.array([[0.12387], [-0.20147], [-0.0535]]))
        cuboidCorners.append(np.array([[-0.12387], [-0.20147], [-0.0535]]))
        cuboidCorners.append(np.array([[0.12387], [0.20147], [-0.0535]]))
        cuboidCorners.append(np.array([[-0.12387], [0.20147], [-0.0535]]))

        #rotation matrix used for rotating the arm and torso with the base
        rotMatrix = np.array([[math.cos(self.rotation), -math.sin(self.rotation), 0],
                          [math.sin(self.rotation), math.cos(self.rotation), 0],
                          [0, 0, 1]])

        #rotate the corners in relation to the rotation of the base
        for i in range(8):
            cuboidCorners[i] = rotMatrix.dot(cuboidCorners[i]).round(4)
            finalCorners.append(newCentre + cuboidCorners[i])
        #endloop

        return finalCorners
