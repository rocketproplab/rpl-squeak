#include "ekf.hpp"

#include <iostream>
using namespace std;

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
    const double Pdiag = 1.0e2;
    this->P.diagonal().setConstant(Pdiag);

    this->Qw.setZero();
    this->Qv.setZero();
    const double Qwdiag = 1.0, Qvdiag = 1.0e-2;
    this->Qw.diagonal().setConstant(Qwdiag);
    this->Qv.diagonal().setConstant(Qvdiag);

    this->Rw = Rw_in;
    this->Rv = Rv_in;
}

void EKF::update(Vec3 const &vExt, Vec3 const &vInt, Vec3 const &wMeasured,
                 double dt)
{
    const double wNorm = this->w.norm();

    Quat delta_w;
    if (wNorm > 0.0) {
        const double tmp = 0.5 * wNorm * dt;
        delta_w.w() = cos(tmp);
        delta_w.vec() = (wMeasured / wNorm) * sin(tmp);
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
    auto upperRight = Fn.block<3, 3>(0, 3);
    upperRight.setIdentity();
    upperRight *= dt;
    Fn.block<3, 3>(3, 0).setZero();
    Fn.block<3, 3>(3, 3).setIdentity();

    // Eq (32)
    Mat6 Pn = Fn * (this->P + Qn) * Fn.transpose();

    Mat3 R_q = q_pred.toRotationMatrix().transpose();
    Vec3 vn = R_q * vExt;

    Mat6 Hn;
    Hn.block<3, 3>(0, 0) = crossMatrix(vn);
    Hn.block<3, 3>(3, 0).setZero();
    Hn.block<3, 3>(0, 3).setZero();
    Hn.block<3, 3>(3, 3).setIdentity();

    const Mat6 M = Pn * Hn.transpose();
    Mat6 S = Hn * M;
    S.block<3, 3>(0, 0) += this->Qv + this->Rv;
    S.block<3, 3>(3, 3) += this->Rw;

    // Kalman gain
    const Mat6 K = M * S.inverse();
    // Warning: In the sample code, M (K here) after MEKF::solve will be
    // transposed

    Vec6 dy;
    dy.segment<3>(0) = vInt - vn;
    dy.segment<3>(3) = wMeasured - this->w;
    const Vec6 dx = K * dy;

    const Quat delta = fromChartPoint(dx.segment<3>(0));

    // Compute q_n, normalize (slightly hacky)
    Quat q_n = q_pred * delta;
    const double q_n_norm = q_n.norm();
    q_n.coeffs() /= q_n_norm;
    // --- VALIDATED UP TO HERE ------

    Mat6 P_nn = (Mat6::Identity() - K * Hn) * Pn;

#if defined CHART_UPDATE
    // If this code path is taken, we update P_nn to reflect
    // the new quaternion in the chart
    // The paper showed little benefit in doing this
    Mat6 A;  // Big matrix in eq 47b
    A.block<3, 3>(0, 0) = chartUpdateMatrix(q_n);
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
