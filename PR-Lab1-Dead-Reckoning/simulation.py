from DifferentialDriveSimulatedRobot import *
import numpy as np

# feature map. Position of 2 point features in the world frame.
M2D = [np.array([[-40, 5]]).T,
        np.array([[-5, 40]]).T,
        np.array([[-5, 25]]).T,
        np.array([[-3, 50]]).T,
        np.array([[-20, 3]]).T,
        np.array([[40,-40]]).T]
xs0 = np.zeros((6,1))   # initial simulated robot pose

robot = DifferentialDriveSimulatedRobot(xs0, M2D) # instantiate the simulated robot object

# circular path
usk = np.array([0.5,0.025]).transpose()
xsk_1 = robot.xsk_1

for i in range(3*2500):
    xsk = robot.fs(xsk_1,usk)

    xsk_1 = xsk

# 8-shape path
#xsk_1 = robot.xsk_1
usk1 = np.array([0.5,0.025]).transpose()
usk2 = np.array([0.5,-0.025]).transpose()

for t in range(3):
    for i in range(2500):
        xsk = robot.fs(xsk_1,usk1)
        xsk_1 = xsk

    for i in range(2500):
        xsk = robot.fs(xsk_1,usk2)
        xsk_1 = xsk
