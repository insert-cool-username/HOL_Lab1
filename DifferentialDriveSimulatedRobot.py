from SimulatedRobot import *
from IndexStruct import *
from Pose import *
import scipy
from roboticstoolbox.mobile.Animations import *
import numpy as np
import conversions
from Feature import PolarFeature, CartesianFeature


class DifferentialDriveSimulatedRobot(SimulatedRobot):
    """
    This class implements a simulated differential drive robot. It inherits from the :class:`SimulatedRobot` class and
    overrides some of its methods to define the differential drive robot motion model.
    """
    def __init__(self, xs0, map=[],*args):
        """
        :param xs0: initial simulated robot state :math:`\\mathbf{x_{s_0}}=[^Nx{_{s_0}}~^Ny{_{s_0}}~^N\psi{_{s_0}}~]^T` used to initialize the  motion model
        :param map: feature map of the environment :math:`M=[^Nx_{F_1},...,^Nx_{F_{nf}}]`

        Initializes the simulated differential drive robot. Overrides some of the object attributes of the parent class :class:`SimulatedRobot` to define the differential drive robot motion model:

        * **Qsk** : Object attribute containing Covariance of the simulation motion model noise.

        .. math::
            Q_k=\\begin{bmatrix}\\sigma_{\\dot u}^2 & 0 & 0\\\\
            0 & \\sigma_{\\dot v}^2 & 0 \\\\
            0 & 0 & \\sigma_{\\dot r}^2 \\\\
            \\end{bmatrix}
            :label: eq:Qsk

        * **usk** : Object attribute containing the simulated input to the motion model containing the forward velocity :math:`u_k` and the angular velocity :math:`r_k`

        .. math::
            \\bf{u_k}=\\begin{bmatrix}u_k & r_k\\end{bmatrix}^T
            :label: eq:usk

        * **xsk** : Object attribute containing the current simulated robot state

        .. math::
            x_k=\\begin{bmatrix}{^N}x_k & {^N}y_k & {^N}\\theta_k & {^B}u_k & {^B}v_k & {^B}r_k\\end{bmatrix}^T
            :label: eq:xsk

        where :math:`{^N}x_k`, :math:`{^N}y_k` and :math:`{^N}\\theta_k` are the robot position and orientation in the world N-Frame, and :math:`{^B}u_k`, :math:`{^B}v_k` and :math:`{^B}r_k` are the robot linear and angular velocities in the robot B-Frame.

        * **zsk** : Object attribute containing :math:`z_{s_k}=[n_L~n_R]^T` observation vector containing number of pulses read from the left and right wheel encoders.
        * **Rsk** : Object attribute containing :math:`R_{s_k}=diag(\\sigma_L^2,\\sigma_R^2)` covariance matrix of the noise of the read pulses`.
        * **wheelBase** : Object attribute containing the distance between the wheels of the robot (:math:`w=0.5` m)
        * **wheelRadius** : Object attribute containing the radius of the wheels of the robot (:math:`R=0.1` m)
        * **pulses_x_wheelTurn** : Object attribute containing the number of pulses per wheel turn (:math:`pulseXwheelTurn=1024` pulses)
        * **Polar2D_max_range** : Object attribute containing the maximum Polar2D range (:math:`Polar2D_max_range=50` m) at which the robot can detect features.
        * **Polar2D\_feature\_reading\_frequency** : Object attribute containing the frequency of Polar2D feature readings (50 tics -sample times-)
        * **Rfp** : Object attribute containing the covariance of the simulated Polar2D feature noise (:math:`R_{fp}=diag(\\sigma_{\\rho}^2,\\sigma_{\\phi}^2)`)

        Check the parent class :class:`prpy.SimulatedRobot` to know the rest of the object attributes.
        """
        super().__init__(xs0, map,*args) # call the parent class constructor

        # Initialize the motion model noise
        # self.Qsk = np.diag(np.array([0.1 ** 2, 0.01 ** 2, np.deg2rad(1) ** 2]))  # simulated acceleration noise (Original Line)
        self.Qsk = np.array([0.1 ** 2, 0.01 ** 2, np.deg2rad(1) ** 2]).reshape(3,1)  # I changed the line above to the following (David's Modification) because in fs, I'm operating on vectors, not matrices.
        self.usk = np.zeros((3, 1))  # simulated input to the motion model

        # Inititalize the robot parameters
        self.wheelBase = 0.5  # distance between the wheels
        self.wheelRadius = 0.1  # radius of the wheels
        self.pulse_x_wheelTurns = 1024  # number of pulses per wheel turn

        # Initialize the sensor simulation
        self.encoder_reading_frequency = 1  # frequency of encoder readings
        self.Re= np.diag(np.array([10 ** 2, 10 ** 2]))  # covariance of simulated wheel encoder noise

        self.Polar2D_feature_reading_frequency = 500  # frequency of Polar2D feature readings
        self.Polar2D_max_range = 50  # maximum Polar2D range, used to simulate the field of view
        self.Rfp = np.diag(np.array([1 ** 2, np.deg2rad(5) ** 2]))  # covariance of simulated Polar2D feature noise

        self.xy_feature_reading_frequency = 1  # frequency of XY feature readings
        self.xy_max_range = 35#25  # maximum XY range, used to simulate the field of view
        self.Feature2D_covariance = np.diag(np.array([0.25**2, 0.25**2])) # Covariance of 2D features

        self.yaw_reading_frequency = 1  # frequency of Yasw readings
        self.v_yaw_std = np.deg2rad(5)  # std deviation of simulated heading noise

    def fs(self, xsk_1, usk):  # input velocity motion model with velocity noise
        """ Motion model used to simulate the robot motion. Computes the current robot state :math:`x_k` given the previous robot state :math:`x_{k-1}` and the input :math:`u_k`:

        .. math::
            \\eta_{s_{k-1}}&=\\begin{bmatrix}x_{s_{k-1}} & y_{s_{k-1}} & \\theta_{s_{k-1}}\\end{bmatrix}^T\\\\
            \\nu_{s_{k-1}}&=\\begin{bmatrix} u_{s_{k-1}} &  v_{s_{k-1}} & r_{s_{k-1}}\\end{bmatrix}^T\\\\
            x_{s_{k-1}}&=\\begin{bmatrix}\\eta_{s_{k-1}}^T & \\nu_{s_{k-1}}^T\\end{bmatrix}^T\\\\
            u_{s_k}&=\\nu_{d}=\\begin{bmatrix} u_d& r_d\\end{bmatrix}^T\\\\
            w_{s_k}&=\\dot \\nu_{s_k}\\\\
            x_{s_k}&=f_s(x_{s_{k-1}},u_{s_k},w_{s_k}) \\\\
            &=\\begin{bmatrix}
            \\eta_{s_{k-1}} \\oplus (\\nu_{s_{k-1}}\\Delta t + \\frac{1}{2} w_{s_k}) \\\\
            \\nu_{s_{k-1}}+K(\\nu_{d}-\\nu_{s_{k-1}}) + w_{s_k} \\Delta t
            \\end{bmatrix} \\quad;\\quad K=diag(k_1,k_2,k_3) \\quad k_i>0\\\\
            :label: eq:fs

        Where :math:`\\eta_{s_{k-1}}` is the previous 3 DOF robot pose (x,y,yaw) and :math:`\\nu_{s_{k-1}}` is the previous robot velocity (velocity in the direction of x and y B-Frame axis of the robot and the angular velocity).
        :math:`u_{s_k}` is the input to the motion model contaning the desired robot velocity in the x direction (:math:`u_d`) and the desired angular velocity around the z axis (:math:`r_d`).
        :math:`w_{s_k}` is the motion model noise representing an acceleration perturbation in the robot axis. The :math:`w_{s_k}` acceleration is the responsible for the slight velocity variation in the simulated robot motion.
        :math:`K` is a diagonal matrix containing the gains used to drive the simulated velocity towards the desired input velocity.

        Finally, the class updates the object attributes :math:`xsk`, :math:`xsk\_1` and  :math:`usk` to made them available for plotting purposes.

        **To be completed by the student**.

        :parameter xsk_1: previous robot state :math:`x_{s_{k-1}}=\\begin{bmatrix}\\eta_{s_{k-1}}^T & \\nu_{s_{k-1}}^T\\end{bmatrix}^T`
        :parameter usk: model input :math:`u_{s_k}=\\nu_{d}=\\begin{bmatrix} u_d& r_d\\end{bmatrix}^T`
        :return: current robot state :math:`x_{s_k}`
        """

        # TODO: to be completed by the student
        # 0.1**2, 0.01**2, np.deg2rad(1)**2  # Covariance Values to test 
        # np.randomnormal use it idk why haha
        K = np.eye(3) 
        '''
        David's Notes
        At this point I received usk (velocity vector) like this usk = [u r]
        Where: u is velocity in x
               r is angular velocity in z
        As we're talking about a turtleBot, I don't have velocity in y (v) 
        due to the config. of the wheels. Leaving it as it is has been causing 
        me issues with the vector's sum. Therefore, I'm going to add a '0' for 
        the y velocity (v), resulting in a 3×1 column vector. That's the reason
        of the line below
        '''
        usk = np.insert(usk, 1, 0).reshape(-1, 1)
    
        eta_s_k_1 = Pose3D(xsk_1[0:3,0].reshape(3,1))
        nu_s_k_1 = Pose3D(xsk_1[3:6,0].reshape(3,1))
        temp0 = nu_s_k_1*self.dt + (1/2)*self.Qsk*(self.dt**2) 
        eta_s_k = eta_s_k_1.oplus(temp0)
        
        # print("eta_s_k:\n",eta_s_k)
        
        '''
        David's Notes
        Remember that
        usk      is my Desired Velocity
        nu_s_k_1 is my Previous Simulated Velocity (k-1)
        
        Therefore the difference between them is my ERROR
        '''
        nu_s_k = nu_s_k_1 + K@(usk - nu_s_k_1) + self.Qsk*self.dt # @ stands for matrix multiplication
        
        # print("nu_s_k:\n",nu_s_k)
        
        self.xsk = np.block([[eta_s_k], [nu_s_k]])
        
        # print("xsk:\n",self.xsk)


        #

        if self.k % self.visualizationInterval == 0:
                self.PlotRobot()
                self.xTraj.append(self.xsk[0, 0])
                self.yTraj.append(self.xsk[1, 0])
                self.trajectory.pop(0).remove()
                self.trajectory = plt.plot(self.xTraj, self.yTraj, marker='.', color='orange', markersize=1)

        self.k += 1
        return self.xsk


    def ReadEncoders(self):
        """ Simulates the robot measurements of the left and right wheel encoders.

        **To be completed by the student**.

        :return zsk,Rsk: :math:`zk=[n_L~n_R]^T` observation vector containing number of pulses read from the left and right wheel encoders. :math:`R_{s_k}=diag(\\sigma_L^2,\\sigma_R^2)` covariance matrix of the read pulses.
        """

        # TODO: to be completed by the student
        lin_vel = np.sqrt(self.xsk[3]**2 + self.xsk[4]**2)
        leftWheel_vel  = lin_vel - (self.xsk[5] * self.wheelBase/2)
        rightWheel_vel = lin_vel + (self.xsk[5] * self.wheelBase/2)
        
        leftWheel_displacement = leftWheel_vel * self.dt
        rightWheel_displacement = rightWheel_vel * self.dt
        
        n_L = leftWheel_displacement * (self.pulse_x_wheelTurns/1) * (1/(2*np.pi)) * ((2*np.pi)/(2*np.pi*self.wheelRadius))
        n_R = rightWheel_displacement * (self.pulse_x_wheelTurns/1) * (1/(2*np.pi)) * ((2*np.pi)/(2*np.pi*self.wheelRadius))
        
        random_noise = np.random.multivariate_normal([0,0], self.Re).reshape(2,1)
        # random_noise = np.array([[0],[0]]) # Zero Noise
        
        Rsk = self.Re
        zsk = np.array([[n_L.item()],[n_R.item()]])
        return zsk + random_noise, Rsk
    
    def ReadCompass(self):
        """ Simulates the compass reading of the robot.

        :return: yaw and the covariance of its noise *R_yaw*
        """

        # TODO: to be completed by the student
        
        # Where:
        #   self.xsk[2]: angular velocity in z (yaw)
        #   v_yaw_std: std deviation of simulated heading noise
        if self.k % self.yaw_reading_frequency == 0:
            random_noise = np.random.normal(0,self.v_yaw_std)
            std=np.array([[self.v_yaw_std]])
            yaw=np.array([[self.xsk[2,0] + random_noise]])
            return yaw, std
        else:
             x=None
             return x, self.v_yaw_std

    def ReadCartesian2DFeature(self):
        #  Bvk is noise
        #  for all_features we compute:
        #      BxF = (-)hsk [+] M[i] + Bvk

        # xy_feature_reading_frequency: frequency of XY feature readings
        # xy_max_range: maximum XY range, used to simulate the field of view
        
        # F_noise_std = 0.1   # Feature Noise sigma
        valid_F = []
        valid_C = []
        if self.k % self.xy_feature_reading_frequency == 0:            
            for NxFj in self.M:
                d = np.sqrt((self.xsk[0] - NxFj[0])**2 + (self.xsk[1] - NxFj[1])**2)  # Euclidean distance between feature and robot
                if d <= self.xy_max_range:
                    noise = np.random.normal(0, self.Feature2D_covariance).diagonal().reshape((2, 1))
                    BxN = Pose3D(self.xsk[:3]).ominus()
                    BxF = CartesianFeature(BxN.boxplus(NxFj))
                    valid_F.append(BxF + noise)
                    valid_C.append(self.Feature2D_covariance)
        # print("F:{}".format(valid_F))

        return valid_F, valid_C
    
        # if self.k % self.xy_feature_reading_frequency != 0:
        #     # Provide readings at the predefined frequency
        #     print("NO FEATURES")
        #     return [], []
        
        # print("Reading features Cartesian 2D")

        # readings = []
        # for f in self.M:
        #     distance = np.linalg.norm(self.xsk[:2] - f)
        #     if distance < self.xy_max_range:
        #         noise = np.random.normal(0, self.Feature2D_covariance).diagonal().reshape((2, 1))
        #         readings.append(Pose3D(self.xsk[:3]).ominus().boxplus(f) + noise)

        # return readings, scipy.linalg.block_diag(*[self.Feature2D_covariance]*len(readings))
        #return readings, [self.Feature2D_covariance]*len(readings)

    def ReadPolarFeature(self):     # Polar Observed
        valid_F = []
        valid_C = []
        if self.k % self.Polar2D_feature_reading_frequency == 0:
            for NxFj in self.M:
                # Identify if feature is polar or not
                if isinstance(NxFj, PolarFeature):
                    # print("Feature is polar:", NxFj)
                    NxFj_c = conversions.p2c(NxFj)
                    NxFj_p = NxFj
                else:
                    # print("Feature is cartesian:", NxFj)
                    NxFj_c = NxFj
                    NxFj_p = conversions.c2p(NxFj)

                d = np.sqrt((self.xsk[0] - NxFj_c[0])**2 + (self.xsk[1] - NxFj_c[1])**2)  # Euclidean distance between feature and robot
                if d <= self.Polar2D_max_range:
                    noise = np.random.normal(0, self.Rfp).diagonal().reshape((2, 1))
                    BxN = Pose3D(self.xsk[:3]).ominus()
                    BxF = BxN.boxplus(NxFj_p)                    
                    valid_F.append(BxF + noise)
                    valid_C.append(self.Rfp)
        # print("F:{}".format(valid_F))

        return valid_F, valid_C

    def PlotRobot(self):
        """ Updates the plot of the robot at the current pose """

        self.vehicleIcon.update([self.xsk[0], self.xsk[1], self.xsk[2]])
        plt.pause(0.0000001)
        return

