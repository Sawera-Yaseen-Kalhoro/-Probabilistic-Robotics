import numpy as np
from math import atan2, cos, sin

class Pose3D(np.ndarray):
    """
    Definition of a robot pose in 3 DOF (x, y, yaw). The class inherits from a ndarray.
    This class extends the ndarray with the $oplus$ and $ominus$ operators and the corresponding Jacobians.
    """
    def __new__(cls, input_array):
        """
        Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarry numpy class.

        :param input_array: array used to initialize the class
        :returns: the instance of a Pose3D class object
        """
        assert input_array.shape == (3, 1), "mean must be a 3x1 vector"

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def oplus(AxB, BxC):
        """
        Given a Pose3D object *AxB* (the self object) and a Pose3D object *BxC*, it returns the Pose3D object *AxC*.

        .. math::
            \\mathbf{{^A}x_B} &= \\begin{bmatrix} ^Ax_B & ^Ay_B & ^A\\psi_B \\end{bmatrix}^T \\\\
            \\mathbf{{^B}x_C} &= \\begin{bmatrix} ^Bx_C & ^By_C & & ^B\\psi_C \\end{bmatrix}^T \\\\

        The operation is defined as:

        .. math::
            \\mathbf{{^A}x_C} &= \\mathbf{{^A}x_B} \\oplus \\mathbf{{^B}x_C} =
            \\begin{bmatrix}
                ^Ax_B + ^Bx_C  \\cos(^A\\psi_B) - ^By_C  \\sin(^A\\psi_B) \\\\
                ^Ay_B + ^Bx_C  \\sin(^A\\psi_B) + ^By_C  \\cos(^A\\psi_B) \\\\
                ^A\\psi_B + ^B\\psi_C
            \\end{bmatrix}
            :label: eq-oplus3dof

        :param BxC: C-Frame pose expressed in B-Frame coordinates
        :returns: C-Frame pose expressed in A-Frame coordinates
        """

        # TODO: to be completed by the student

        # define AxC
        AxC = np.empty((3,1))
        
        # compute the direct compounding
        AxC[0] = AxB[0] + BxC[0]*cos(AxB[2]) - BxC[1]*sin(AxB[2])
        AxC[1] = AxB[1] + BxC[0]*sin(AxB[2]) + BxC[1]*cos(AxB[2])
        AxC[2] = AxB[2] + BxC[2]

        return AxC

    def ominus(AxB):
        """
        Inverse pose compounding of the *AxB* pose (the self objetc):

        .. math::
            ^Bx_A = \\ominus ^Ax_B =
            \\begin{bmatrix}
                -^Ax_B \\cos(^A\\psi_B) - ^Ay_B \\sin(^A\\psi_B) \\\\
                ^Ax_B \\sin(^A\\psi_B) - ^Ay_B \\cos(^A\\psi_B) \\\\
                -^A\\psi_B
            \\end{bmatrix}
            :label: eq-ominus3dof

        :returns: A-Frame pose expressed in B-Frame coordinates (eq. :eq:`eq-ominus3dof`)
        """

        # TODO: to be completed by the student

        # define BxA
        BxA = np.empty((3,1))

        # compute inverse compounding
        BxA[0] = -AxB[0]*cos(AxB[2]) - AxB[1]*sin(AxB[2])
        BxA[1] = AxB[0]*sin(AxB[2]) - AxB[1]*cos(AxB[2])
        BxA[2] = -AxB[2]

        return BxA

