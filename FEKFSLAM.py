from MapFeature import *
from FEKFMBL import *

class FEKFSLAM(FEKFMBL):
    """
    Class implementing the Feature-based Extended Kalman Filter for Simultaneous Localization and Mapping (FEKFSLAM).
    It inherits from the FEKFMBL class, which implements the Feature Map based Localization (MBL) algorithm using an EKF.
    :class:`FEKFSLAM` extends :class:`FEKFMBL` by adding the capability to map features previously unknown to the robot.
    """
 
    def __init__(self,  *args):

        super().__init__(*args)

        # self.xk_1 # state vector mean at time step k-1 inherited from FEKFMBL

        self.nzm = 0  # number of measurements observed
        self.nzf = 0  # number of features observed

        self.H = None  # Data Association Hypothesis
        self.nf = 0  # number of features in the state vector

        self.plt_MappedFeaturesEllipses = []

        return
    
    def hm(self,xk):
        """
        Measurement observation model. This method computes the expected measurements :math:`h_m(x_k,v_m)` given the
        mean state vector :math:`x_k` and the measurement noise :math:`v_m`. It is implemented by calling to the ancestor
        class :meth:`EKF.EKF.h` method.

        :param xk: mean state vector.
        :return: expected measruments.
        """

        # TODO: To be completed by the student
        if self.zm is not None:

            hm = EKF_3DOFDifferentialDriveInputDisplacement.h(self, xk)
            print("2")
            
        else:
            hm = None
        return hm

 
    def AddNewFeatures(self, xk, Pk, znp, Rnp):
        """
        This method adds new features to the map. Given:

        * The SLAM state vector mean and covariance:

        .. math::
            {{}^Nx_{k}} & \\approx {\\mathcal{N}({}^N\\hat x_{k},{}^NP_{k})}\\\\
            {{}^N\\hat x_{k}} &=   \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k}}&=
            \\begin{bmatrix}
            {{}^NP_{B}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance

        * And the vector of non-paired feature observations (feature which have not been associated with any feature in the map), and their covariance matrix:

            .. math::
                {z_{np}} &=   \\left[ {}^Bz_{F_1} ~  \\cdots ~ {}^Bz_{F_{n_{zf}}}  \\right]^T \\\\
                {R_{np}}&= \\begin{bmatrix}
                {}^BR_{F_1} &  \\cdots & 0  \\\\
                \\vdots &  \\ddots & \\vdots \\\\
                0 & \\cdots & {}^BR_{F_{n_{zf}}}
                \\end{bmatrix}
                :label: FEKFSLAM-non-paire-feature-observations

        this method creates a grown state vector ([xk_plus, Pk_plus]) by adding the new features to the state vector.
        Therefore, for each new feature :math:`{}^Bz_{F_i}`, included in the vector :math:`z_{np}`, and its corresponding feature observation noise :math:`{}^B R_{F_i}`, the state vector mean and covariance are updated as follows:

            .. math::
                {{}^Nx_{k}^+} & \\approx {\\mathcal{N}({}^N\\hat x_{k}^+,{}^NP_{k}^+)}\\\\
                {{}^N x_{k}^+} &=
                \\left[ {{}^N x_{B_k}^T} ~ {{}^N x_{F_1}^T} ~ \\cdots ~{{}^N x_{F_n}^T}~ |~\\left({{}^N x_{B_k} \\boxplus ({}^Bz_{F_i} }+v_k)\\right)^T \\right]^T \\\\
                {{}^N\\hat x_{k}^+} &=
                \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~ \\cdots ~{{}^N\\hat x_{F_n}^T}~ |~{{}^N\\hat x_{B_k} \\boxplus {}^Bz_{F_i}^T } \\right]^T \\\\
                {P_{k}^+}&= \\begin{bmatrix}
                {{}^NP_{B_k}}  &  {{}^NP_{B_kF_1}}   &  \\cdots   &  {{}^NP_{B_kF_n}} & | & {{}^NP_{B_k} J_{1 \\boxplus}^T}\\\\
                {{}^NP_{F_1B_k}}  &  {{}^NP_{F_1}}   &  \\cdots   &  {{}^NP_{F_1F_n}} & | & {{}^NP_{F_1B_k} J_{1 \\boxplus}^T}\\\\
                \\vdots  & \\vdots & \\ddots  & \\vdots & | &  \\vdots \\\\
                {{}^NP_{F_nB_k}}  &  {{}^NP_{F_nF_1}}   &  \\cdots   &  {{}^NP_{F_n}}  & | & {{}^NP_{F_nB_k} J_{1 \\boxplus}^T}\\\\
                \\hline
                {J_{1 \\boxplus} {}^NP_{B_k}}  &  {J_{1 \\boxplus} {}^NP_{B_kF_1}}   &  \\cdots   &  {J_{1 \\boxplus} {}^NP_{B_kF_n}}  & | &  {J_{1 \\boxplus} {}^NP_R J_{1 \\boxplus} ^T} + {J_{2\\boxplus}} {{}^BR_{F_i}} {J_{2\\boxplus}^T}\\\\
                \\end{bmatrix}
                :label: FEKFSLAM-add-a-new-feature

        :param xk: state vector mean
        :param Pk: state vector covariance
        :param znp: vector of non-paired feature observations (they have not been associated with any feature in the map)
        :param Rnp: Matrix of non-paired feature observation covariances
        :return: [xk_plus, Pk_plus] state vector mean and covariance after adding the new features
        """

        # assert znp.size > 0, "AddNewFeatures: znp is empty" # I commented it because in my code znp is a list

        ## To be completed by the student
        # znp: List of features

        f2add = len(znp)    # Number of features to add
        if f2add == 0:
            # No features to add
            return xk, Pk

        xk_1_R = Pose3D(xk[0:self.xB_dim, 0:self.xB_dim])   # Extract robot's pose
        Pk_1_robot = Pk[0:self.xB_dim, 0:self.xB_dim]       # Extract covariance of the robot

        xk_plus = xk     # Initialization of state vector
        
        
        for i in range(f2add):
            nf = int((Pk.shape[0] - self.xB_dim) / self.xF_dim)   # Number of features included in the current covariance matrix (local variable)

            # State Vector
            NxFi = self.g(xk_1_R, znp[i])     # Inverse sensor model, compute feature pose wrt N-frame (NxB [+] BxF)
            xk_plus = np.vstack((xk_plus, NxFi))

            # Covariance of feature i (red)
            Jgxi = self.Jgx(xk_1_R, znp[i])
            Jgvi = self.Jgv(xk_1_R, znp[i])
            NpFi = Jgxi @ Pk_1_robot @ Jgxi.T + Jgvi @ Rnp[i] @ Jgvi.T

            # Correlated Covariance btw robot pose and features
            row1_yellowMAT = Pk[0:self.xB_dim, :]       # Covariances with correlation btw robot pose and feature
            if Pk.shape[0] == self.xB_dim:  # I don't have features in the state vector yet, therefore Pk only has the covariance of the robot
                green_row = Jgxi @ Pk
            else:
                temp0 = np.split(row1_yellowMAT, [self.xB_dim], axis=1)
                temp1 = np.hsplit(temp0[1], nf)
                green_row = Jgxi @ temp0[0]     # 1st Element of green_row
                for each_NpBkFi in temp1:       # For all other elements
                    column = Jgxi @ each_NpBkFi
                    green_row = np.hstack((green_row, column))    # Stack the arrays horizontally
            
            # Full Pk_plus
            Pk_plus_top    = np.hstack((Pk,green_row.T))
            Pk_plus_bottom = np.hstack((green_row,NpFi))
            Pk_plus = np.vstack((Pk_plus_top,Pk_plus_bottom))
            
            Pk = Pk_plus
        
        self.nf = int((xk_plus.shape[0] - self.xB_dim) / self.xF_dim)  # We update the number of features in the state vector (attribute of object self)

        return xk_plus, Pk_plus

    def Prediction(self, uk, Qk, xk_1, Pk_1):
        """
        This method implements the prediction step of the FEKFSLAM algorithm. It predicts the state vector mean and
        covariance at the next time step. Given state vector mean and covariance at time step k-1:

        .. math::
            {}^Nx_{k-1} & \\approx {\\mathcal{N}({}^N\\hat x_{k-1},{}^NP_{k-1})}\\\\
            {{}^N\\hat x_{k-1}} &=   \\left[ {{}^N\\hat x_{B_{k-1}}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k-1}}&=
            \\begin{bmatrix}
            {{}^NP_{B_{k-1}}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance-k-1

        the control input and its covariance :math:`u_k` and :math:`Q_k`, the method computes the state vector mean and covariance at time step k:

        .. math::
            {{}^N\\hat{\\bar x}_{k}} &=   \\left[ {f} \\left( {{}^N\\hat{x}_{B_{k-1}}}, {u_{k}}  \\right)  ~  { {}^N\\hat x_{F_1}^T} \\cdots { {}^N\\hat x_{F_n}^T}\\right]^T\\\\
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}
            :label: FEKFSLAM-prediction-step

        where

        .. math::
            {F_{1_k}} &= \\left.\\frac{\\partial {f_S({}^Nx_{k-1},u_k,w_k)}}{\\partial {{}^Nx_{k-1}}}\\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}} \\\\
             &=
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{B_{k-1}}}} &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{Fn}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}} \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_x}} & {0} & \\cdots & {0} \\\\
            {0}   & {I} & \\cdots & {0} \\\\
            \\vdots& \\vdots  & \\ddots & \\vdots  \\\\
            {0}   & {0} & \\cdots & {I} \\\\
            \\end{bmatrix}
            \\\\{F_{2_k}} &= \\left. \\frac{\\partial {f({}^Nx_{k-1},u_k,w_k)}}{\\partial {w_{k}}} \\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}}
            =
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {w_{k}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {w_{k}}}\\\\
            \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {w_{k}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_w}}\\\\
            {0}\\\\
            \\vdots\\\\
            {0}\\\\
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-Jacobian

        obtaining the following covariance matrix:
        
        .. math::
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}{{}^N\\bar P_{k}}
             &=
            \\begin{bmatrix}
            {J_{f_x}P_{B_{k-1}} J_{f_x}^T} + {J_{f_w}Q J_{f_w}^T}  & |  &  {J_{f_x}P_{B_kF_1}} & \\cdots & {J_{f_x}P_{B_kF_n}}\\\\
            \\hline
            {{}^NP_{F_1B_k} J_{f_x}^T} & |  &  {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_n}}\\\\
            \\vdots & | & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_nB_k} J_{f_x}^T} & | &  {{}^NP_{F_nF_1}} & \\cdots & {{}^NP_{F_n}}
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-covariance

        The method returns the predicted state vector mean (:math:`{}^N\\hat{\\bar x}_k`) and covariance (:math:`{{}^N\\bar P_{k}}`).

        :param uk: Control input
        :param Qk: Covariance of the Motion Model noise
        :param xk_1: State vector mean at time step k-1
        :param Pk_1: Covariance of the state vector at time step k-1
        :return: [xk_bar, Pk_bar] predicted state vector mean and covariance at time step k
        """
        ## To be completed by the student
        self.xk_1 = xk_1 if xk_1 is not None else self.xk_1
        self.Pk_1 = Pk_1 if Pk_1 is not None else self.Pk_1

        self.uk = uk;
        self.Qk = Qk  # store the input and noise covariance for logging

        # Robot Pose
        
        # xk_1_R: Previous Robot Pose
        xk_1_R = Pose3D(xk_1[0:self.xB_dim, 0:self.xB_dim]) # Extract the part corresponding to the robot's previous pose
        xk_1[0:self.xB_dim, 0:self.xB_dim] = self.f(xk_1_R, uk) # Compute new pose

        # Covariance of robot pose
        Ak = self.Jfx(xk_1_R, uk)
        Wk = self.Jfw(xk_1_R)
        
        Pk_1_robot = Pk_1[0:self.xB_dim, 0:self.xB_dim]
        Pk_robot = Ak @ Pk_1_robot @ Ak.T + Wk @ Qk @ Wk.T
        
        # Handling case where we don't have feature in the vector state yet
        nf = int((Pk_1.shape[0] - self.xB_dim) / self.xF_dim)   # Number of features included in the current covariance matrix
        if nf == 0:     # No features in the vector state
            self.xk_bar = xk_1
            self.Pk_bar = Pk_robot
            return self.xk_bar, self.Pk_bar

        # Correlated covariances between robot pose & features
        columns_to_pick = nf * self.xF_dim  # Computes how many columns we need to pick from Pk_1
        NpBF1_Fn = Pk_1[0:self.xB_dim, -columns_to_pick:]       # Covariances with correlation btw robot pose and feature
        green_row = Ak @ np.hsplit(NpBF1_Fn, nf)
        green_row = np.hstack(green_row)    # Stack the arrays horizontally

        # Correlated covariances between features
        yellow_mat = Pk_1[self.xB_dim:, -columns_to_pick:]

        # Full Pk_bar
        Pk_bar_top    = np.hstack((Pk_robot,green_row))
        Pk_bar_bottom = np.hstack((green_row.T,yellow_mat))
        Pk_bar = np.vstack((Pk_bar_top,Pk_bar_bottom))

        self.xk_bar = xk_1
        self.Pk_bar = Pk_bar
        return self.xk_bar, self.Pk_bar
        # return xk_bar, Pk_bar

    def Localize(self, xk_1, Pk_1):
        """
        This method implements the FEKFSLAM algorithm. It localizes the robot and maps the features in the environment.
        It implements a single interation of the SLAM algorithm, given the current state vector mean and covariance.
        The unique difference it has with respect to its ancestor :meth:`FEKFMBL.Localize` is that it calls the method
        :meth:`AddNewFeatures` to add new non-paired features to the map.

        :param xk_1: state vector mean at time step k-1
        :param Pk_1: covariance of the state vector at time step k-1
        :return: [xk, Pk] state vector mean and covariance at time step k
        """

        ## To be completed by the student
        uk, Qk = self.GetInput()
        xk_bar, Pk_bar = self.Prediction(uk, Qk, xk_1, Pk_1)
        zm, Rm, Hm, Vm = self.GetMeasurements()
        self.zm = zm
        zf, Rf = self.GetFeatures()
        self.nz = len(zf) # if zf is not None else 0 # I don't need it anymore
        self.Hp = self.DataAssociation(xk_bar, Pk_bar, zf, Rf)
        zk, Rk, Hk, Vk, znp, Rnp = self.StackMeasurementsAndFeatures(zm, Rm, Hm, Vm, zf, Rf, self.Hp)

        if zk is not None:
            xk, Pk = self.Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk)
        else:
            xk, Pk = xk_bar, Pk_bar

        # xk,Pk = self.AddNewFeatures(xk,Pk,znp,Rnp)

        self.xk = xk
        self.Pk = Pk
        self.Log(self.robot.xsk, xk, Pk, xk_bar, zk)

        # Use the variable names zm, zf, Rf, znp, Rnp so that the plotting functions work
        # self.Log(self.robot.xsk, self._GetRobotPose(self.xk), self._GetRobotPoseCovariance(self.Pk),
        #          self._GetRobotPose(self.xk_bar), zm)  # log the results for plotting

        zf_new_format = None
        Rf_new_format = None
        if zf:
            for zfi, Rfi in zip(zf, Rf):
                zf_new_format = np.concatenate([zf_new_format, zfi]) if zf_new_format is not None else zfi
                Rf_new_format = scipy.linalg.block_diag(Rf_new_format, Rfi) if Rf_new_format is not None else Rfi
            print("zf new shape", zf_new_format.shape, zf_new_format)

        self.PlotUncertainty(zf_new_format, Rf_new_format, znp, Rnp)
        return self.xk, self.Pk
    
    def LocalizationLoop(self, x0, P0, usk):
        xk_1 = x0
        Pk_1 = P0
        xsk_1 = self.robot.xsk_1

        zf,Rf = self.GetFeatures()
        # zf = [CartesianFeature(np.array([[4, 4]]).T),
        #       CartesianFeature(np.array([[8, 8]]).T)]
        # Rf = [np.eye(2),np.eye(2)]

        xk_1, Pk_1 = self.AddNewFeatures(xk_1,Pk_1,zf,Rf)

        for self.k in range(self.kSteps):
            xsk = self.robot.fs(xsk_1, usk)  # Simulate the robot motion
            xk, Pk = self.Localize(xk_1, Pk_1)  # Localize the robot
            xsk_1 = xsk  # current state becomes previous state for next iteration
            xk_1 = xk
            Pk_1 = Pk
            # self.PlotUncertainty(xk,Pk) # Tanakrit commented to make the MapBased Localization work
            #, but when I try the EKF input velocity, it didn't plot things

        self.PlotState()  # plot the state estimation results
        plt.show()

    def PlotMappedFeaturesUncertainty(self):
        """
        This method plots the uncertainty of the mapped features. It plots the uncertainty ellipses of the mapped
        features in the environment. It is called at each Localization iteration.
        """
        # remove previous ellipses
        for i in range(len(self.plt_MappedFeaturesEllipses)):
            self.plt_MappedFeaturesEllipses[i].remove()
        self.plt_MappedFeaturesEllipses = []

        self.xk=BlockArray(self.xk,self.xF_dim, self.xB_dim)
        self.Pk=BlockArray(self.Pk,self.xF_dim, self.xB_dim)

        # draw new ellipses
        for Fj in range(self.nf):
            feature_ellipse = GetEllipse(self.xk[[Fj]],
                                         self.Pk[[Fj,Fj]])  # get the ellipse of the feature (x_Fj,P_Fj)
            plt_ellipse, = plt.plot(feature_ellipse[0], feature_ellipse[1], 'r')  # plot it
            self.plt_MappedFeaturesEllipses.append(plt_ellipse)  # and add it to the list

    def PlotUncertainty(self, zf, Rf, znp, Rnp):
        """
        This method plots the uncertainty of the robot (blue), the mapped features (red), the expected feature observations (black) and the feature observations.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of feature observations
        :param znp: vector of non-paired feature observations
        :param Rnp: covariance matrix of non-paired feature observations
        :return:
        """
        if self.k % self.robot.visualizationInterval == 0:
            self.PlotRobotUncertainty()
            self.PlotFeatureObservationUncertainty(znp, Rnp,'b')
            self.PlotFeatureObservationUncertainty(zf, Rf,'g')
            self.PlotExpectedFeaturesObservationsUncertainty()
            self.PlotMappedFeaturesUncertainty()

    def DataAssociation(self, xk, Pk, zf, Rf):
        """
        Data association algorithm. Given state vector (:math:`x_k` and :math:`P_k`) including the robot pose and a set of feature observations
        :math:`z_f` and its covariance matrices :math:`R_f`,  the algorithm  computes the expected feature
        observations :math:`h_f` and its covariance matrices :math:`P_f`. Then it calls an association algorithms like
        :meth:`ICNN` (JCBB, etc.) to build a pairing hypothesis associating the observed features :math:`z_f`
        with the expected features observations :math:`h_f`.

        The vector of association hypothesis :math:`H` is stored in the :attr:`H` attribute and its dimension is the
        number of observed features within :math:`z_f`. Given the :math:`j^{th}` feature observation :math:`z_{f_j}`, *self.H[j]=i*
        means that :math:`z_{f_j}` has been associated with the :math:`i^{th}` feature . If *self.H[j]=None* means that :math:`z_{f_j}`
        has not been associated either because it is a new observed feature or because it is an outlier.

        :param xk: mean state vector including the robot pose
        :param Pk: covariance matrix of the state vector
        :param zf: vector of feature observations
        :param Rf: Covariance matrix of the feature observations
        :return: The vector of asociation hypothesis H
        """

        # TODO: To be completed by the student
        hF_list = []
        PF_list = []
        for i in range(self.nf):
            hF_list.append(self.hfj(xk,i)) # Sensor Model, it tells me the pose of feature i wrt robot frame
            j = self.Jhfjx(xk,i) # Jacobian of feature i, used to compute covariance of feature i
            
            PFi =  self.GetRobotPoseCovariance(j) @ self.GetRobotPoseCovariance(Pk) @ self.GetRobotPoseCovariance(j).T # + Jhfv(xk) @ Rf @ Jhfv(xk).T # Covariance of feature i

            PF_list.append(PFi) 

        Hp = self.ICNN(hF_list,PF_list,zf,Rf,self.zfi_dim)
        return Hp