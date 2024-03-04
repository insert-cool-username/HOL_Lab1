from MapFeature import *
from blockarray import *
class FEKFSLAMFeature(MapFeature):
    """
    This class extends the :class:`MapFeature` class to implement the Feature EKF SLAM algorithm.
    The  :class:``MapFeature`` class is a base class providing support to localize the robot using a map of point features.
    The main difference between FEKMBL and FEAKFSLAM is that the former uses the robot pose as a state variable,
    while the latter uses the robot pose and the feature map as state variables. This means that few methods provided by
    class need to be overridden to gather the information from state vector instead that from the deterministic map.
    """
    def hf(self, xk):  # Observation function for al zf observations
        """
        This is the direct observation model, implementing the feature observation equation for the data
        association hypothesis :math:`\mathcal{H}`, the features observation vector :math:`z_f, the state vector :math:`x_k`,
        and the observation noise :math:`v_k`:

        .. math::
            \mathcal{H}&=[F_a~\\cdots~F_b~\\cdots~F_c]\\\\
            z_f&=[z_{f_1}^T~\\cdots~z_{f_i}^T~\\cdots~z_{f_{n_{zf}}}^T]^T\\\\
            x_k&=[^Nx_B^T~x_{rest}^T]^T\\\\
            v_k&=[v_{f1_k}^T~\\cdots~v_{fi_k}^T~\\cdots~v_{fn_{zf_k}}^T]^T\\\\\
            z_f&=h_f(x_k,v_k) \\\\
            :label: eq-hf

        which may be expanded as follows:

        .. math::
            \\begin{bmatrix} z_{f_1} \\\\ \\vdots  \\\\ z_{f_i} \\\\ \\vdots \\\\ z_{n_{zf}} \\end{bmatrix} = \\begin{bmatrix} h_{F_a}(x_k,v_k) \\\\ \\vdots \\\\ h_{F_b}(x_k,v_k) \\\\ \\vdots \\\\ h_{Fc}(x_k,v_k) \\end{bmatrix}
            = \\begin{bmatrix} s2o(\\ominus ^Nx_B \\boxplus^Nx_{F_{a}})+ v_{f1_k}\\\\ \\vdots \\\\  s2o(\\ominus ^Nx_B \\boxplus^Nx_{F_{b}})+ v_{fi_k}\\\\ \\vdots \\\\ s2o(\\ominus ^Nx_B \\boxplus ^Nx_{F_{c}}) + v_{fn_{zf}}\\end{bmatrix}
            :label: eq-hf-element-wise

        being :math:`h_{F_j}(\cdot)` (:meth:`hfj`) the observation function (eq. :eq:`eq-hfj`) for the data association hypothesis :math:`\\mathcal{H}` and  :meth:`s2o` the conversion
        function from the storage representation to the observation one.

        The method computes the expected observations :math:`h_{f}` for the observed features contained within the :math:`z_{f}` features observation vector.
        To do it, it iterates over each feature observation :math:`z_{f_i}` calling the method :meth:`hfj` for its corresponding associated feature :math:`\mathcal{H}_i=F_j`
        to compute the expected observation :math:`h_{F_j}`, collecting all them in the returned vector.

        :param xk: state vector mean :math:`\\hat x_k`.
        :return: vector of expected features observations corresponding to the vector of observed features :math:`z_f`.
        """
        
        zf = None
        for i in range(self.nz): # nz: Number of observations
            Fj = self.Hp[i]      # What is H? -> Data Association Matrix
            if Fj is not None:
                zfj = self.hfj(xk, Fj)
                zf = np.concatenate((zf, zfj)) if zf is not None else zfj
        return zf

    def hfj(self, xk_bar, Fj):  # Observation function for zf_i and x_Fj
        """
        This method implements the direct observation model for a single feature observation  :math:`z_{f_i}` , so it implements its related
        observation function (see eq. :eq:`eq-FEKFSLAM-hfj`). For a single feature observation :math:`z_{f_i}` of the feature :math:`^Nx_{F_j}` the method computes its
        expected observation from the current robot pose :math:`^Nx_B`.
        This function uses a generic implementation through the following equation:

        .. math::
            z_{f_i}=h_{Fj}(x_k,v_k)=s2o(\\ominus ^Nx_B \\boxplus ^Nx_{F_j}) + v_{fi_k}
            :label: eq-FEKFSLAM-hfj

        Where :math:`^Nx_B` is the robot pose and :math:`^Nx_{F_j}` are both included within the state vector:

        .. math::
            x_k=[^Nx_B^T~\cdots~^Nx_{F_j}~\cdots~^Nx_{F_{nf}}]^T
            :label: eq-FEKFSLAM-xk

        and :meth:`s2o` is a conversion function from the store representation to the observation representation.

        The method is called by :meth:`FEKFSLAM.hf` to compute the expected observation for each feature
        observation contained in the observation vector :math:`z_f=[z_{f_1}^T~\\cdots~z_{f_i}^T~\\cdots~z_{f_{n_zf}}^T]^T`.

        :param xk_bar: mean of the predicted state vector
        :param Fj: map index of the observed feature.
        :return: expected observation of the feature :math:`^Nx_{F_j}`
        """
<<<<<<< HEAD
        
        NxB = self.GetRobotPose(xk_bar)
        Nx_Fj = np.setdiff1d(xk_bar, NxB)

        #print(type(NxB), type(Nx_Fj))
=======

        # This method is called from DataAssociation and Fj = i (index of for loop) if there is any features in the state vector
        NxB = self.GetRobotPose(xk_bar) # Get Robot Pose
        xBpose_dim = NxB.shape[0]   # Saves dimension of the robot pose, the proper way to do it might be instead of using 'xBpose_dim'
                                    # use 'xB_dim' as in 'FEKFMBL'. If we include velocities in our state vector, I should start taking
                                    # features from the row 7 instead of row 4 (for example). In this case it will start taking features
                                    # from row 4 even if later we expand our state vector to include velocities
        xF_dim = self.Feature.feature.shape[0]  # Feature dimensionality, as it is implemented in 'FEKFMBL'

        Fj_start = (Fj * xF_dim) + xBpose_dim
        Fj_end = Fj_start + xF_dim
        Nx_Fj = CartesianFeature(xk_bar[Fj_start:Fj_end,0].reshape(2,1))

>>>>>>> david
        return self.s2o((NxB.ominus()).boxplus(Nx_Fj))

    def Jhfjx(self, xk, Fj):  # Observation function for zf_i and x_Fj
        """
        Jacobian of the single feature direct observation model :meth:`hfj` (eq. :eq:`eq-FEKFSLAM-hfj`)  with respect to the state vector :math:`\\bar{x}_k`:

        .. math::
            x_k&=[^Nx_B^T~\cdots~^Nx_{F_j}~\cdots~^Nx_{F_{nf}}]^T\\\\
            J_{hfjx}&=\\frac{\\partial h_{f_{zfi}}({x}_k, v_k)}{\\partial {x}_k}=
            \\frac{\\partial s2o(\\ominus ^Nx_B \\boxplus ^Nx_{F_j})+v_{fi_k}}{\\partial {x}_k}\\\\
            &=
            \\begin{bmatrix}
            \\frac{\\partial{h_{F_j}(x_k,v_k)}}{ \\partial {{}^Nx_{B_k}}} & \\frac{\\partial{h_{F_j}(x_k,v_k)}}{ \\partial {{}^Nx_{F_1}}} & \\cdots &\\frac{\\partial{h_{F_j}(x_k,v_k)}}{ \\partial {{}^Nx_{F_j}}} & \\cdots & \\frac{\\partial{h_{F_j}(x_k,v_k)}}{ \\partial {{}^Nx_{F_n}} } \\\\
            \\end{bmatrix} \\\\
            &=
            \\begin{bmatrix}
            J_{s2o}{J_{1\\boxplus} J_\\ominus} & {0} & \\cdots & J_{s2o}{J_{2\\boxplus}} & \\cdots &{0}\\\\
            \\end{bmatrix}\\\\
            :label: eq-FEKFSLAM-Jhfjx

        where we have used the abreviature:

        .. math::
            J_{s2o} &\equiv J_{s2o}(\\ominus ^Nx_B \\boxplus^Nx_{F_j})\\\\
            J_{1\\boxplus} &\equiv J_{1\\boxplus}(\\ominus ^Nx_B,^Nx_{F_j} )\\\\
            J_{2\\boxplus} &\equiv J_{2\\boxplus}(\\ominus ^Nx_B,^Nx_{F_j} )\\\\

        :param xk: state vector mean
        :param Fj: map index of the observed feature
        :return: Jacobian matrix defined in eq. :eq:`eq-Jhfjx`        """

<<<<<<< HEAD
        ## To be completed by the student
        NxB = self.GetRobotPose(xk)
        Nx_F = np.setdiff1d(xk, NxB)
        Nx_Fj = Nx_F[int(Fj)]
=======
        # This method is called from DataAssociation and Fj = i (index of for loop) if there is any features in the state vector
        NxB = self.GetRobotPose(xk) # Get Robot Pose
        xBpose_dim = NxB.shape[0]   # Saves dimension of the robot pose, the proper way to do it might be instead of using 'xBpose_dim'
                                    # use 'xB_dim' as in 'FEKFMBL'. If we include velocities in our state vector, I should start taking
                                    # features from the row 7 instead of row 4 (for example). In this case it will start taking features
                                    # from row 4 even if later we expand our state vector to include velocities
        xF_dim = self.Feature.feature.shape[0]  # Feature dimensionality, as it is implemented in 'FEKFMBL'

        Fj_start = (Fj * xF_dim) + xBpose_dim
        Fj_end = Fj_start + xF_dim
        Nx_Fj = CartesianFeature(xk[Fj_start:Fj_end,0].reshape(2,1))
>>>>>>> david
        Jp = (self.J_s2o(NxB.ominus().boxplus(Nx_Fj))) @ (NxB.ominus().J_1boxplus(Nx_Fj)) @ NxB.J_ominus()
        return Jp

class FEKFSLAM2DCartesianFeature(FEKFSLAMFeature, Cartesian2DMapFeature):
    """
    Class to inherit from both :class:`FEKFSLAMFeature` and :class:`Cartesian2DMapFeature` classes.
    Nothing else to do here (if using s2o & o2s), only needs to be defined.
    """
    pass


