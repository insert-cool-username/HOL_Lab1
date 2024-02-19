from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *
from Feature import CartesianFeature

class EKF_3DOFDifferentialDriveInputDisplacement(GFLocalization, DR_3DOFDifferentialDrive, EKF):
    """
    This class implements an EKF localization filter for a 3 DOF Diffenteial Drive using an input displacement motion model incorporating
    yaw measurements from the compass sensor.
    It inherits from :class:`GFLocalization.GFLocalization` to implement a localization filter, from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class and, finally, it inherits from
    :class:`EKF.EKF` to use the EKF Gaussian filter implementation for the localization.
    """
    def __init__(self, kSteps, robot, *args):
        """
        Constructor. Creates the list of  :class:`IndexStruct.IndexStruct` instances which is required for the automated plotting of the results.
        Then it defines the inital stawe vecto mean and covariance matrix and initializes the ancestor classes.

        :param kSteps: number of iterations of the localization loop
        :param robot: simulated robot object
        :param args: arguments to be passed to the base class constructor
        """

        self.dt = 0.1  # dt is the sampling time at which we iterate the KF
        x0 = np.zeros((3, 1))  # initial state x0=[x y z psi u v w r]^T
        P0 = np.zeros((3, 3))  # initial covariance

        # this is required for plotting
        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1)]

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(index, kSteps, robot, x0, P0, *args)

    def f(self, xk_1, uk):
        # TODO: To be completed by the student
        # xk_1 = Pose3D(xk_1)
        xk_bar = xk_1.oplus(uk)
        return xk_bar

    def Jfx(self, xk_1, uk): # Added uk
        # TODO: To be completed by the student
        # xk_1 = Pose3D(xk_1)
        J = xk_1.J_1oplus(uk)
        return J

    def Jfw(self, xk_1):
        # TODO: To be completed by the student
        J = xk_1.J_2oplus()
        return J

    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student
        h = np.array([[0, 0, 1]])
        h = h @ xk
        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk
        """
        # TODO: To be completed by the student
        
        # We call the method GetInput() from the DR class to get the robot encoder readings
        n, rsk = DR_3DOFDifferentialDrive.GetInput(self)

        # Compute the displacement on each wheel
        d_l = n[0]*(1/self.robot.pulse_x_wheelTurns)*(2*np.pi)*(self.wheelRadius)
        d_r = n[1]*(1/self.robot.pulse_x_wheelTurns)*(2*np.pi)*(self.wheelRadius)
        
        # Compute robot displacement
        d = (d_r+d_l)/2
        
        # #Angular Displacement
        delta_theta = (d_r - d_l)/self.wheelBase
        
        # We return uk 
        temp = np.vstack([d, 0, delta_theta]).reshape(3, 1)

        uk = Pose3D(temp)
        
        # # Split the displacement into its x, y components
        # uk_x = (d*np.cos(delta_theta))
        # uk_y = (d*np.sin(delta_theta))    
        
        # We compute our Qk (covariance of the input)
        # For displacement
        variance_nl = rsk[0,0]  # Left Encoder 
        variance_nr = rsk[1,1]  # Right Encoder
        cov_d = (np.pi*self.wheelRadius*(1/self.robot.pulse_x_wheelTurns))**2 * (variance_nl + variance_nr)

        # For delta_theta
        cov_delta_theta = (np.pi**2 * self.wheelRadius)*(1/(self.robot.pulse_x_wheelTurns**2 * self.wheelBase**2)) * (variance_nl + variance_nr)

        Qk = np.diag(np.array([cov_d, 0, cov_delta_theta]))

        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        # TODO: To be completed by the student
        zk, Rk = self.robot.ReadCompass()
        Hk = np.array([[0, 0, 1]])
        Vk = np.array([[1]]) # What size should Vk has to be?
        return zk, Rk, Hk, Vk


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 5000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    x0 = Pose3D(np.zeros((3, 1)))
    P0 = np.zeros((3, 3))

    dd_robot = EKF_3DOFDifferentialDriveInputDisplacement(kSteps,robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)

    exit(0)