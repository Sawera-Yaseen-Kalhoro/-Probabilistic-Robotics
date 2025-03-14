from Localization import *
import numpy as np
from Pose3D import *

class DR_3DOFDifferentialDrive(Localization):
    """
    Dead Reckoning Localization for a Differential Drive Mobile Robot.
    """
    def __init__(self, index, kSteps, robot, x0, *args):
        """
        Constructor of the :class:`prlab.DR_3DOFDifferentialDrive` class.

        :param args: Rest of arguments to be passed to the parent constructor
        """

        super().__init__(index, kSteps, robot, x0, *args)  # call parent constructor

        self.dt = 0.1  # dt is the sampling time at which we iterate the DR
        self.t_1 = 0.0  # t_1 is the previous time at which we iterated the DR
        self.wheelRadius = 0.1  # wheel radius
        self.wheelBase = 0.5  # wheel base
        self.robot.pulse_x_wheelTurns = 4096  # number of pulses per wheel turn

    def Localize(self, xk_1, uk):  # motion model
        """
        Motion model for the 3DOF (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`) Differential Drive Mobile robot using as input the readings of the wheel encoders (:math:`u_k=[n_L~n_R]^T`).

        :parameter xk_1: previous robot pose estimate (:math:`x_{k-1}=[x_{k-1}~y_{k-1}~\psi_{k-1}]^T`)
        :parameter uk: input vector (:math:`u_k=[u_{k}~v_{k}~w_{k}~r_{k}]^T`)
        :return xk: current robot pose estimate (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`)
        """

        # Store previous state and input for Logging purposes
        self.etak_1 = xk_1  # store previous state
        self.uk = uk  # store input
        ###### uk = pulse reading from the encoders [ticks]

        # TODO: to be completed by the student
        etak_1 = Pose3D(self.etak_1)

        ## convert ticks into robot velocity
        d = uk/self.robot.pulse_x_wheelTurns * 2*np.pi*self.wheelRadius
        v = d/self.dt

        v_L = v[0,0]
        v_R = v[1,0]
        
        vr = (v_L+v_R)/2
        wr = (v_R-v_L)/self.wheelBase

        nu_d = np.array([[vr],[0],[wr]])

        self.xk = etak_1.oplus(nu_d*self.dt)

        return self.xk

    def GetInput(self):
        """
        Get the input for the motion model. In this case, the input is the readings from both wheel encoders.

        :return: uk:  input vector (:math:`u_k=[n_L~n_R]^T`)
        """

        # TODO: to be completed by the student
        (uk,Qk) = self.robot.ReadEncoders()

        # # convert to displacement
        # d = uk/self.robot.pulse_x_wheelTurns * 2*np.pi*self.wheelRadius

        # dr = (d[0]+d[1])/2
        # dthetar = (d[1]-d[0])/self.wheelBase


        return (uk,Qk)

