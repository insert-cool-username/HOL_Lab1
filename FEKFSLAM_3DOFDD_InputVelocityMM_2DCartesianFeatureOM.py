from FEKFSLAM import *
from FEKFMBL import *
from EKF_3DOFDifferentialDriveInputDisplacement import *
from Pose import *
from blockarray import *
from MapFeature import *
import numpy as np
from FEKFSLAMFeature import *

class FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM(FEKFSLAM2DCartesianFeature, FEKFSLAM, EKF_3DOFDifferentialDriveInputDisplacement):
    def __init__(self, *args):

        self.Feature = globals()["CartesianFeature"]
        self.Pose = globals()["Pose3D"]
       
        super().__init__(*args)

    def GetMeasurements(self):
        zk, Rk, Hk, Vk = super().GetMeasurements()
        nf = int((len(self.xk_1) - self.xB_dim) / self.xF_dim)
        Hk = np.hstack((Hk, np.zeros((1,nf * self.xF_dim))))
<<<<<<< HEAD

=======
>>>>>>> david
        return zk, Rk, Hk, Vk

    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student
        h = np.array([[0, 0, 1]])
        nf = int((len(self.xk_1) - self.xB_dim) / self.xF_dim)
        h = np.hstack((h, np.zeros((1,nf * self.xF_dim))))
        h = h @ xk
<<<<<<< HEAD
=======
        # Falta definir aqui hf
>>>>>>> david
        return h  # return the expected observations

    # def GetFeatures(self):
    # Get features is inherited from EKF_3DOFDifferentialDriveInputDisplacement


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-6, 40]]).T),
           CartesianFeature(np.array([[-6, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6, 1))
    kSteps = 5000
    alpha = 0.99

    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object

    x0 = Pose3D(np.zeros((3, 1)))
    dr_robot = DR_3DOFDifferentialDrive(index, kSteps, robot, x0)
    robot.SetMap(M)

    auv = FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM([], alpha, kSteps, robot)

    P0 = np.eye((3)) * 0.1
    usk=np.array([[0.5, 0.03]]).T
    auv.LocalizationLoop(x0, P0, usk)

    exit(0)
