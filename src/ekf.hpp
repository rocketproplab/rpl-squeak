#include <Eigen/Dense>
using Quat = Eigen::Quaterniond;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Vector<double, 6>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

class EKF {
   public:
    void init(Mat3 const &Rw_in, Mat3 const &Rv_in);
    void update(Vec3 const &vExt, Vec3 const &vInt, Vec3 const &wMeasured,
                double dt);

    inline Quat get_quat() { return Quat(this->q); }
    inline void get_q(double *qout)
    {
        auto c = this->q.coeffs();
        qout[0] = c(0);
        qout[1] = c(1);
        qout[2] = c(2);
        qout[3] = c(3);
    }

   private:
    Quat q;
    Vec3 w;
    Mat6 P;
    Mat3 Qw;
    Mat3 Qv;
    Mat3 Rw;
    Mat3 Rv;

    inline Quat fromChartPoint(Vec3 const &e)
    {
        const double aux = 1.0 / sqrt(4.0 + e.squaredNorm());
        return Quat(2.0 * aux, e.x() * aux, e.y() * aux, e.z() * aux);
    }

    // TODO: fix for quat component ordering
    // output:
    //     T_out: transformation matrix to update cov matrix for chart centered
    //     at quat p
    inline Mat3 chartUpdateMatrix(Quat delta)
    {
        Mat3 T;
        auto const d = delta.w() * delta.coeffs();  // or delta.z() ??
        double const d0 = d(0), d1 = d(1), d2 = d(2), d3 = d(3);
        // clang-format off
        T << d0,  d3, -d2,
            -d3,  d0,  d1,
             d2, -d1,  d0;
        // clang-format on
        return T;
    }

    inline Mat3 crossMatrix(Vec3 v)
    {
        Mat3 out;
        double const v1 = v(0), v2 = v(1), v3 = v(2);
        // clang-format off
        out <<  0, -v3,  v2,
               v3,   0, -v1,
              -v2,  v1,   0;
        // clang-format on
        return out;
    }
};
