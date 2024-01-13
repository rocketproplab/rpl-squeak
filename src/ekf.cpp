#include "ekf.hpp"

#include <iostream>

using Quat = Eigen::Quaterniond;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Vector<double, 6>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

void EKF::init(Mat3 const &Rw_in, Mat3 const &Rv_in)
{
    this->q = Quat(1, 0, 0, 0);
    this->w.setZero();
    this->P.setZero();
    const double P_diag = 1.0e2;
    this->P.diagonal().setConstant(P_diag);

    this->Qw.setZero();
    this->Qv.setZero();
    const double Qw_diag = 1.0, Qv_diag = 1.0e-2;
    this->Qw.diagonal().setConstant(Qw_diag);
    this->Qv.diagonal().setConstant(Qv_diag);

    this->Rw = Rw_in;
    this->Rv = Rv_in;
}

void EKF::update(Vec3 const &vExt, Vec3 const &v_int, Vec3 const &w_measured,
                 double dt)
{
    const double w_norm = this->w.norm();

    Quat delta_w;
    if (w_norm > 0.0) {
        const double tmp = 0.5 * w_norm * dt;
        delta_w.w() = cos(tmp);
        delta_w.vec() = (w_measured / w_norm) * sin(tmp);
    } else {
        delta_w = Quat(1, 0, 0, 0);
    }

    // TODO: we can remove a bunch of temp variables here for possible
    // performance gains

    // Updated orientation quaternion
    const Quat q_pred = this->q * delta_w;
    // Eq (33)
    Mat6 Qn;
    const double dt2 = dt * dt / 2.0, dt3 = dt * dt * dt / 3.0;
    Qn.block<3, 3>(0, 0) = this->Qw * dt3;
    Qn.block<3, 3>(0, 3) = -this->Qw * dt2;
    Qn.block<3, 3>(3, 0) = -this->Qw * dt2;
    Qn.block<3, 3>(3, 3) = this->Qw * dt;

    // Eq (31)
    Mat3 R_delta = delta_w.toRotationMatrix().transpose();
    Mat6 Fn;
    Fn.block<3, 3>(0, 0) = R_delta;
    auto upper_right = Fn.block<3, 3>(0, 3);
    upper_right.setIdentity();
    upper_right *= dt;
    Fn.block<3, 3>(3, 0).setZero();
    Fn.block<3, 3>(3, 3).setIdentity();

    // Eq (32)
    Mat6 P_n = Fn * (this->P + Qn) * Fn.transpose();

    Mat3 R_q = q_pred.toRotationMatrix().transpose();
    Vec3 v_n = R_q * vExt;

    Mat6 Hn;
    Hn.block<3, 3>(0, 0) = cross_matrix(v_n);
    Hn.block<3, 3>(3, 0).setZero();
    Hn.block<3, 3>(0, 3).setZero();
    Hn.block<3, 3>(3, 3).setIdentity();

    const Mat6 M = P_n * Hn.transpose();
    Mat6 S = Hn * M;
    S.block<3, 3>(0, 0) += this->Qv + this->Rv;
    S.block<3, 3>(3, 3) += this->Rw;

    // Kalman gain
    const Mat6 K = M * S.inverse();
    // Warning: In the sample code, M (K here) after MEKF::solve will be
    // transposed

    Vec6 dy;
    dy.segment<3>(0) = v_int - v_n;
    dy.segment<3>(3) = w_measured - this->w;
    const Vec6 dx = K * dy;

    const Quat delta = from_chart_point(dx.segment<3>(0));

    // Compute q_n, normalize (slightly hacky)
    Quat q_n = q_pred * delta;
    const double q_n_norm = q_n.norm();
    q_n.coeffs() /= q_n_norm;

    Mat6 P_nn = (Mat6::Identity() - K * Hn) * P_n;

#if defined CHART_UPDATE
    // UNTESTED, but little benefit shown in paper

    // If this code path is taken, we update P_nn to reflect
    // the new quaternion in the chart
    // The paper showed little benefit in doing this
    Mat6 A;  // Big matrix in eq 47b
    A.block<3, 3>(0, 0) = chart_update_matrix(q_n);
    A.block<3, 3>(3, 0).setZero();
    A.block<3, 3>(0, 3).setZero();
    A.block<3, 3>(3, 3).setIdentity();
    P_nn = A * P_nn * A.transpose();
#endif

    // Update quaternion state
    this->q = q_n;
    // Add to current angular velocity
    this->w += dx.segment<3>(3);
    this->P = P_nn;
}
