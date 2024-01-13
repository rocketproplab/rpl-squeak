#include "ekf.hpp"

using namespace Eigen;
using Quat = Eigen::Quaterniond;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Vector<double, 6>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

/*
 * Copyright (C) 2017 P.Bernal-Polo
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * File:   OE_Tester.cpp
 * Author: P.Bernal-Polo
 *
 * Created on February 19, 2017, 2:05 PM
 */

#include <ctime>

using namespace std;

// pi definition
#define PI 3.14159265358979323846

#include <math.h>    // round(), sqrt()
#include <stdlib.h>  // rand_r()

#include <fstream>   // input, output to files
#include <iomanip>   // io manipulation (set width, digits, ...)
#include <iostream>  // input, output
#include <sstream>   // std::ostringstream
#include <thread>    // to multithreading

// OE_Tester: class used for testing the performance of an EKF algorithm
//
//   This class implements some methods with the objective of testing, in
//   an easy way, the performance of an algorithm of the
//   EKF class
class Simulator {
   public:
    // PUBLIC METHODS
    Simulator();
    Simulator(const Simulator &orig);
    virtual ~Simulator();

    // sets the title for files to save results
    void set_title(std::string titleIn);

    // sets the frequencies for which the performance of the filter will be
    // evaluated. the frequency vector introduced here will be modified choosing
    // the nearest frequencies that produce an update at Tsim
    void set_f(double *fIn, int NfrequenciesIn);

    // sets the convergence threshold for the convergence step of the simulation
    void set_convergenceThreshold(double convergenceThresholdIn);

    // sets the maximum number of updates for the convergence step of the
    // simulation
    void set_maxConvergenceUpdates(int maxConvergenceUpdatesIn);

    // sets the time for which the simulation is run
    void set_Tsim(double TsimIn);

    // sets the simulation steps carried before each EKF update
    void set_simStepsPerUpdate(int dtdtsiminIn);

    // sets the number of trajectories generated for each frequency in order of
    // testing the EKF
    void set_Ntimes(int NtimesIn);

    // sets the maximum variance for the process noise in the angular velocity
    void set_QwMax(double QwMaxIn);

    // sets the maximum variance for the process noise in the vector
    void set_QvMax(double QvMaxIn);

    // sets the variance for the angular velocity measurement
    void set_Rw(double RwIn);

    // sets the variance for the vector measurement
    void set_Rv(double RvIn);

    // uses the OE_Tester parameters to test an EKF, and saves the results in a
    // file
    void test(EKF *myEstimator, double infoUpdateTime);

    // it perform the test() method in multiple threads (to improve the testing
    // speed) to compile this function we need to use the C++11 compiler:
    // properties->C++ Compiler->C++ Standard->C++11
    // and link a library:
    // properties->Linker->Libraries->Add Standard Library->Posix Threads
    void multithread_test(EKF *myEstimator, double infoUpdateTime,
                          int Nthreads);

    // performs an evaluation of the computational cost of an estimator
    void test_computationalCost(long Ntimes, EKF *theEstimator);

   private:
    // PRIVATE VARIABLES
    // title used to save results
    std::string title;
    // number of frequency samples
    int Nfrequencies;
    // vector of frequency samples
    double f[50];
    // threshold for which convergence is considered achieved (it is stored in
    // radians, but introduced in degrees)
    double convergenceThreshold;
    // maximum number of updates when trying to achieve convergence
    int maxConvergenceUpdates;
    // simulation time (s)
    double Tsim;
    // (dt/dtsim) measure of how big is the update time step (dt) vs the
    // simulation time step (dtsim)
    int dtdtsim;
    // times to repeat the simulation for each frequency
    int Ntimes;
    // maximum angular velocity variance per unit of time (rad^2/s^3)
    double QwMax;
    // maximum variance of the measured vector
    double QvMax;
    // variance in measured angular velocity (rad^2/s^2)
    double Rw;
    // variance in the vector measurement
    double Rv;

    // PRIVATE METHODS

    // used to save results of the test method
    void save_results(int *Ncrashes, int *NnoConv, double *eT, double *eT2,
                      double *eQ, double *eQ2);

    // PRIVATE STATIC METHODS

    // generates a trajectory step from a previous known state
    void trajectoryStep(Quat &q, Vec3 &w, double dtsim, double nw,
                        unsigned int *seed);

    // generates a vector measurement from a known state
    void get_measurement(Quat const &q, Vec3 const &w, Vec3 const &v, double nv,
                         double rv, double rw, unsigned int *seed, Vec3 &vm,
                         Vec3 &wm);

    // computes the angle between real and estimated orientations
    double thetaQ(Quat const &qr, Quat const &qe);

    // generates a sample of a normal distribution
    static double myNormalRandom(unsigned int *seed, double sigma);

    // checks if there has been a crash in the algorithm
    bool hasCrashed(Quat const &q);

    // a wrapper to use the test() method with multithreading
    static void testWrapper(Simulator *myOE_Tester, EKF *myEstimator,
                            double infoUpdateTime);

    static void printMatrix(double *M, int n, int m);
};

Simulator::Simulator()
{
    title = "simulationResult";
    // number of frequency samples
    Nfrequencies = 4;
    // vector of frequency samples
    f[0] = 1.0;
    f[1] = 5.0;
    f[2] = 10.0;
    f[3] = 100.0;
    // threshold for which convergence is considered achieved (it is stored in
    // radians, but introduced in degrees)
    convergenceThreshold = 3.0 * PI / 180.0;
    // maximum number of updates when trying to achieve convergence
    maxConvergenceUpdates = 1000;
    // simulation time (s)
    Tsim = 1.0;
    // (dt/dtsim) measure of how big is the update time step (dt) vs the
    // simulation time step (dtsim)
    dtdtsim = 10;
    // times to repeat the simulation for each frequency
    Ntimes = 100;
    // maximum angular velocity variance per unit of time (rad^2/s^3)
    QwMax = 1.0e0;
    // maximum vector variance
    QvMax = 1.0e-2;
    // variance in measured angular velocity (rad^2/s^2)
    Rw = 1.0e-4;
    // variance in the vector measurement
    Rv = 1.0e-4;
}

Simulator::Simulator(const Simulator &orig) { (void)orig; }

Simulator::~Simulator() {}

// Method: set_title
// sets the title for files to save results
// inputs:
//  titleIn: title of the file to save results
// outputs:
void Simulator::set_title(std::string titleIn)
{
    title = titleIn;

    return;
}

// Method: set_f
// sets the frequencies for which the performance of the filter will be
// evaluated. the frequency vector introduced here will be modified choosing the
// nearest frequencies that produce an update at obj
// inputs:
//  fIn: vector of frequencies (Hz)
//  NfrequenciesIn: number of frequencies in fIn
void Simulator::set_f(double *fIn, int NfrequenciesIn)
{
    // first the Nfrequencies
    if (NfrequenciesIn < 0 || 50 < NfrequenciesIn) {
        std::cout
            << "Error in \"set_f()\": Nfrequencies must be between 1 and 50\n";
        throw 1;
    }
    Nfrequencies = NfrequenciesIn;
    // then the frequencies
    for (int k = 0; k < Nfrequencies; k++) {
        if (fIn[k] < 1.0 / Tsim) {
            cout << " " << k << " " << fIn[k] << " " << Tsim << "\n";
            std::cout << "Error in \"set_f()\": frequencies must be greater "
                         "than 1.0/Tsim\n";
            std::cout << "Is it possible that the variable Nfrequencies is not "
                         "in agreement with "
                         "the definition of f[]?";
            throw 1;
        }
    }
    for (int k = 0; k < Nfrequencies; k++) f[k] = fIn[k];

    return;
}

// Method: set_convergenceThreshold
// sets the convergence threshold for the convergence step of the simulation
// inputs:
//  convergenceThresholdIn: degrees difference for which the convergence is
//  considered achieved (degrees)
// outputs:
void Simulator::set_convergenceThreshold(double convergenceThresholdIn)
{
    if (convergenceThresholdIn <= 0.0) {
        std::cout << "Error in \"set_convergenceThreshold()\": "
                     "convergenceThreshold must be positive\n";
        throw 1;
    }
    convergenceThreshold = convergenceThresholdIn * PI / 180.0;

    return;
}

// Method: set_maxConvergenceUpdates
// sets the maximum number of updates for the convergence step of the simulation
// inputs:
//  maxConvergenceUpdatesIn: maximum number of updates for the convergence step
// outputs:
void Simulator::set_maxConvergenceUpdates(int maxConvergenceUpdatesIn)
{
    if (maxConvergenceUpdatesIn < 1) {
        std::cout << "Error in \"set_maxConvergenceUpdates()\": "
                     "maxConvergenceUpdates can not be "
                     "negative\n";
        throw 1;
    }
    maxConvergenceUpdates = maxConvergenceUpdatesIn;

    return;
}

// Method: set_Tsim
// sets the time for which the simulation is run
// inputs:
//  TsimIn: period of time for which the simulation is run (s)
// outputs:
void Simulator::set_Tsim(double TsimIn)
{
    double minf = 1.0e9;
    for (int k = 0; k < Nfrequencies; k++) {
        if (f[k] < minf) minf = f[k];
    }
    if (TsimIn < 1.0 / minf) {
        std::cout
            << "Error in \"set_Tsim()\": Tsim must be greater than 1.0/fmin\n";
        throw 1;
    }
    Tsim = TsimIn;

    return;
}

// Method: set_simStepsPerUpdate
// sets the simulation steps carried before each EKF update
// inputs:
//  dtdtsiminIn: number of simulation steps carried before each update (that
//  turns out to be equal to dt/dtsim)
// outputs:
void Simulator::set_simStepsPerUpdate(int dtdtsimIn)
{
    if (dtdtsimIn < 1) {
        std::cout << "Error in \"set_simStepsPerUpdate()\": dtdtsim must be "
                     "positive\n";
        throw 1;
    }
    dtdtsim = dtdtsimIn;

    return;
}

// Method: set_Ntimes
// sets the number of trajectories generated for each frequency in order of
// testing the EKF inputs:
//  NtimesIn: number of trajectories generated for each frequency
// outputs:
void Simulator::set_Ntimes(int NtimesIn)
{
    if (NtimesIn < 1) {
        std::cout << "Error in \"set_Ntimes()\": Ntimes must be positive\n";
        throw 1;
    }
    Ntimes = NtimesIn;

    return;
}

// Method: set_QwMax
// sets the maximum variance for the process noise in the angular velocity
// inputs:
//  QwMaxIn: maximum variance for the process noise in the angular velocity per
//  unit of time (rad^2/s^3)
// outputs:
void Simulator::set_QwMax(double QwMaxIn)
{
    if (QwMaxIn <= 0.0) {
        std::cout << "Error in \"set_QwMax()\": QwMax must be positive\n";
        throw 1;
    }
    QwMax = QwMaxIn;

    return;
}

// Method: set_QvMax
// sets the maximum variance for the process noise in the measured vector
// inputs:
//  QvMaxIn: maximum variance for the process noise in the measured vector
// outputs:
void Simulator::set_QvMax(double QvMaxIn)
{
    if (QvMaxIn <= 0.0) {
        std::cout << "Error in \"set_QvMax()\": QvMax must be positive\n";
        throw 1;
    }
    QvMax = QvMaxIn;

    return;
}

// Method: set_Rw
// sets the variance for the angular velocity measurement
// inputs:
//  RwIn: variance of the angular velocity measurement (rad^2/s^2)
// outputs:
void Simulator::set_Rw(double RwIn)
{
    if (RwIn <= 0.0) {
        std::cout << "Error in \"set_Rw()\": Rw must be positive\n";
        throw 1;
    }
    Rw = RwIn;

    return;
}

// Method: set_Rv
// sets the variance for the vector measurement
// inputs:
//  RvIn: variance for the vector measurement
// outputs:
void Simulator::set_Rv(double RvIn)
{
    if (RvIn <= 0.0) {
        std::cout << "Error in \"set_Rv()\": Rv must be positive\n";
        throw 1;
    }
    Rv = RvIn;

    return;
}

// Method: test
// uses the OE_Tester parameters to test an EKF, and
// saves the results in a file
// inputs:
//  myEstimator: the EKF object for which we want to find its performance
//  infoUpdateTime: time step to show information about the simulation progress.
//  If it is negative, no information will be shown
// outputs:
void Simulator::test(EKF *myEstimator, double infoUpdateTime)
{
    // first of all, let me write about the decisions taken for this simulation.
    // The first thought was to generate a sequence of states that generate a
    // smooth and continuous trajectory for the orientations. Then we would take
    // simulated measurements, and used these sequences for updating our
    // estimator. There are two major problems with this approach:
    // - we can not assert the coincidence of the final step of the simulation
    // with the final update
    // - we can not assert the coincidence of all the estimation updates for all
    // the frequencies, with a given time step of the sequence of simulated
    // states A second approach to overcome these problems is to generate a
    // different trajectory for each frequency. It is done defining an entire
    // quantity dtdtsim, that determines the simulation steps performed before
    // each estimator update (asserting the equality "n*dtsim = dt", with n an
    // integer). Then, the frequencies are transformed to satisfy the equality
    // "m*dt = Tsim", with m an integer.
    // After explaining this, we can start with the code

    // we require the condition that the final update happens at the end of the
    // simulation:
    //    n*dt = Tsim  =>  f = 1.0/dt = n/Tsim = round(Tsim/dt)/Tsim =
    //    round(Tsim*f)/Tsim
    for (int nf = 0; nf < Nfrequencies; nf++) {
        f[nf] = round(Tsim * f[nf]) / Tsim;
    }
    // we do not want repeated frequencies
    for (int nf1 = 0; nf1 < Nfrequencies; nf1++) {
        for (int nf2 = nf1 + 1; nf2 < Nfrequencies; nf2++) {
            if (round(Tsim * f[nf1]) == round(Tsim * f[nf2])) {
                Nfrequencies--;
                for (int nf = nf2; nf < Nfrequencies; nf++) f[nf] = f[nf + 1];
                nf2--;
            }
        }
    }

    // we compute the sigmas of the measurement covariances
    double rw = sqrt(Rw);
    double rv = sqrt(Rv);

    // we will save our error definitions in these vectors (general definitions:
    // outside threads)
    int NnoConvergence[Nfrequencies];
    int Ncrashes[Nfrequencies];
    double TConv[Nfrequencies];
    double TConv2[Nfrequencies];
    double errorQ[Nfrequencies];
    double errorQ2[Nfrequencies];
    for (int nf = 0; nf < Nfrequencies; nf++) {
        NnoConvergence[nf] = 0;
        Ncrashes[nf] = 0;
        TConv[nf] = 0.0;
        TConv2[nf] = 0.0;
        errorQ[nf] = 0.0;
        errorQ2[nf] = 0.0;
    }

    // we will use this variable to display some information
    std::string msg = "";

    // we initialize the seed
    unsigned int seed = 42;
    // unsigned int seed = std::hash<std::thread::id>()(
    //     std::this_thread::get_id());  // hasher( this_thread::get_id() );
    // //time(NULL);

    // we take the initial time to give time estimations
    int beginingTime = time(NULL);
    int lastTime = beginingTime;

    // we do it several times to get a good performance estimation
    for (int n = 0; n < Ntimes; n++) {
        // we test for each frequency
        for (int nf = 0; nf < Nfrequencies; nf++) {
            // we compute the update time step
            double dt = 1.0 / f[nf];

            // we generate random sigmas for the process uniformly using the
            // maximum variances
            double nw =
                ((rand_r(&seed) % 1000000 + 1) / 1000000.0) * sqrt(QwMax);
            double nv =
                ((rand_r(&seed) % 1000000 + 1) / 1000000.0) * sqrt(QvMax);

            // we take a random initial orientation, and zero angular velocity
            Quat q0(myNormalRandom(&seed, 1.0), myNormalRandom(&seed, 1.0),
                    myNormalRandom(&seed, 1.0), myNormalRandom(&seed, 1.0));

            double q0norm = q0.norm();  // or sqrt(q0.norm) ?
            q0.coeffs() /= q0norm;
            Vec3 w0 = Vec3::Zero();

            // and we initialize the estimator
            Mat3 RwIn = Mat3::Zero();
            Mat3 RvIn = Mat3::Zero();
            RwIn.diagonal().setConstant(Rw);
            RvIn.diagonal().setConstant(Rv);
            myEstimator->init(RwIn, RvIn);

            // the first simulation part is to maintain the sensor static until
            // it reaches convergence, too many updates, or a crash
            double theta = 1.0e2;
            int Nupdates = 0;
            bool crashed = false;
            while (theta > this->convergenceThreshold &&
                   Nupdates < this->maxConvergenceUpdates) {
                // first we generate a random unit vector measured in the
                // external reference frame (it could also have a norm different
                // from one, but we do so for simplicity)
                Vec3 v(myNormalRandom(&seed, 1), myNormalRandom(&seed, 1),
                       myNormalRandom(&seed, 1));
                v /= v.norm();
                // we compute the simulated measurement
                Vec3 vm, wm;
                this->get_measurement(q0, w0, v, nv, rv, rw, &seed, vm, wm);
                // and we update the filter with the measurements
                myEstimator->update(v, vm, wm, dt);
                Nupdates += 1;
                // we take the estimated orientation, and we check if the
                // algorithm has crashed
                Quat qEst = myEstimator->get_quat();
                if (hasCrashed(qEst)) {
                    cout << "crashed at 552" << endl;
                    crashed = true;
                    break;
                }
                // we compute the error in the gravity estimation
                theta = thetaQ(q0, qEst);
            }
            // we check if there has been convergence
            if (crashed || Nupdates >= this->maxConvergenceUpdates) {
                // if not, we add one to the no-convergences, and we go with the
                // next frequency
                NnoConvergence[nf] += 1;
                continue;
            }

            // we go ahead with the metrics computations, and with the
            // simulation
            TConv[nf] += Nupdates;
            TConv2[nf] += Nupdates * Nupdates;

            // we take the initial state
            Quat q(q0);
            Vec3 w(w0);

            // and the second step of the simulation is to generate a random
            // orientation trajectory, and try to estimate it
            //   we need the simulation time step
            double dtsim = dt / this->dtdtsim;
            //   and the number of updates
            Nupdates = round(this->Tsim * this->f[nf]);
            //   we also need the integral of thetaQ, and thetaQ^2
            double thQ = 0.0, thQ2 = 0.0;
            //   now we can start iterating
            for (int nup = 0; nup < Nupdates; nup++) {
                // first we iterate in the simulation (trajectory generation)
                for (int nt = 0; nt < this->dtdtsim; nt++) {
                    trajectoryStep(q, w, dtsim, nw, &seed);
                }
                // we generate a random unit vector measured in the external
                // reference frame
                Vec3 v(myNormalRandom(&seed, 1), myNormalRandom(&seed, 1),
                       myNormalRandom(&seed, 1));
                v /= v.norm();
                // then we simulate the measurement
                Vec3 vm, wm;
                this->get_measurement(q, w, v, nv, rv, rw, &seed, vm, wm);
                // finally we update the estimator with the measurement
                myEstimator->update(v, vm, wm, dt);
                // and we add the errors
                //   we get the quaternion
                Quat qEst = myEstimator->get_quat();
                if (hasCrashed(qEst)) {
                    cout << "Crashed at 611" << endl;
                    crashed = true;
                    break;
                }
                //   angle between orientations (defined as the angle we have to
                //   rotate)
                theta = thetaQ(q, qEst);
                thQ += theta;
                thQ2 += theta * theta;
                // we repeat this for Nupdates
            }
            // we check if there has been a crash
            if (crashed) {
                Ncrashes[nf] += 1;
                continue;
            }

            // now we can compute the error definitions
            double dtTsim = dt / Tsim;
            //   error in orientation
            errorQ[nf] += thQ * dtTsim;
            errorQ2[nf] += thQ2 * dtTsim;
        }

        // if enough time has passed
        if (infoUpdateTime > 0.0) {
            int currentTime = time(NULL);
            if (currentTime - lastTime > infoUpdateTime) {
                // we display some information about resting time, and about the
                // simulation progress
                double eTime2end =
                    (currentTime - beginingTime) * ((double)(Ntimes - n)) / n;
                int eTime2endD = floor(eTime2end / 86400.0);
                int eTime2endH =
                    floor((eTime2end - eTime2endD * 86400.0) / 3600.0);
                int eTime2endM = floor(
                    (eTime2end - eTime2endD * 86400.0 - eTime2endH * 3600.0) /
                    60.0);
                int eTime2endS = ceil(eTime2end - eTime2endD * 86400.0 -
                                      eTime2endH * 3600.0 - eTime2endM * 60.0);
                double proportion = ((double)n) / Ntimes;

                std::ostringstream message;
                message << "[";
                for (int i = 0; i < round(proportion * 100.0); i++)
                    message << "-";
                for (int i = 0; i < 100 - round(proportion * 100.0); i++)
                    message << " ";
                message << "] (" << proportion
                        << ") Estimated remaining time: ";
                if (eTime2endD > 0) message << eTime2endD << " d ";
                if (eTime2endH > 0) message << eTime2endH << " h ";
                if (eTime2endM > 0) message << eTime2endM << " m ";
                message << eTime2endS << " s                      ";

                int msgLength = msg.length();
                msg = message.str();
                for (int i = 0; i < msgLength; i++) std::cout << "\b";
                std::cout << msg << std::flush;

                // we update lastTime for the next infoUpdate
                lastTime = currentTime;
            }
        }
    }

    // we save the results, but first we correct the measurements
    for (int nf = 0; nf < Nfrequencies; nf++) {
        TConv[nf] /= f[nf];
        TConv2[nf] /= f[nf] * f[nf];
        errorQ[nf] *= 180.0 / PI;
        errorQ2[nf] *= 180.0 / PI * 180.0 / PI;
    }
    save_results(NnoConvergence, Ncrashes, TConv, TConv2, errorQ, errorQ2);

    // if the user want (and to not printing on all threads)
    if (infoUpdateTime > 0.0) {
        // we print the time taken
        int currentTime = time(NULL);
        double eTime2end = currentTime - beginingTime;
        int eTime2endD = floor(eTime2end / 86400.0);
        int eTime2endH = floor((eTime2end - eTime2endD * 86400.0) / 3600.0);
        int eTime2endM = floor(
            (eTime2end - eTime2endD * 86400.0 - eTime2endH * 3600.0) / 60.0);
        int eTime2endS = ceil(eTime2end - eTime2endD * 86400.0 -
                              eTime2endH * 3600.0 - eTime2endM * 60.0);

        std::ostringstream message;
        int msgLength = msg.length();
        for (int i = 0; i < msgLength; i++) message << "\b";
        message << "Completed. Time taken: ";
        if (eTime2endD > 0) message << eTime2endD << " d ";
        if (eTime2endH > 0) message << eTime2endH << " h ";
        if (eTime2endM > 0) message << eTime2endM << " m ";
        message << eTime2endS << " s                      ";
        for (int i = 0; i < msgLength; i++) message << " ";
        message << "\n";

        std::cout << message.str() << std::flush;
    }
}

// Method: saveResults
// used to save results of the test method
// inputs:
//  Ncrashes: vector (for each frequency) with number of algorithm crashes
//  NnoConv: vector (for each frequency) with number of non-achieved
//  convergences at the second step of the simulation eTs: vector (for each
//  frequency) with the sum of convergence times eT2s: vector (for each
//  frequency) with the sum of squared convergence times eQs: vector (for each
//  frequency) with the sum of errors in orientation estimations eQ2s: vector
//  (for each frequency) with the sum of squared errors in orientation
//  estimations
// outputs:
void Simulator::save_results(int *NnoConvs, int *Ncrashes, double *eTs,
                             double *eT2s, double *eQs, double *eQ2s)
{
    // we open the file
    std::ofstream file;
    std::string myString = this->title + ".dat";
    cout << "writing to " << myString << endl;
    file.open(myString.c_str());

    // we set up the format
    file.precision(10);

    // we save some information about the tester
    file << "# title = " << this->title << "\n";
    file << "# convergenceThreshold = "
         << this->convergenceThreshold * 180.0 / PI << "\n";
    file << "# maxConvergenceUpdates = " << this->maxConvergenceUpdates << "\n";
    file << "# Tsim = " << this->Tsim << "\n";
    file << "# dtdtsim = " << this->dtdtsim << "\n";
    file << "# Ntimes = " << this->Ntimes << "\n";
    file << "# QwMax = " << this->QwMax << "\n";
    file << "# QvMax = " << this->QvMax << "\n";
    file << "# Rw = " << this->Rw << "\n";
    file << "# Rv = " << this->Rv << "\n";

    // now, for each frequency we save:
    file << "# frequency (Hz) |";
    file << " n no convergences | n crashes |";
    file << " mean convergence time (s) | sigma convergence time (s) | sigma "
            "mean convergence time "
            "(s) |";
    file << " mean error in orientation (degrees) | sigma of error in "
            "orientation (degrees) | "
            "sigma mean error in orientation (degrees) \n";
    for (int nf = 0; nf < this->Nfrequencies; nf++) {
        // first we need the number of samples
        int NsamplesT = this->Ntimes - NnoConvs[nf];
        if (NsamplesT == 0) NsamplesT = 1;
        int NsamplesSim = this->Ntimes - Ncrashes[nf];
        if (NsamplesSim == 0) NsamplesSim = 1;
        // - frequency (Hz)
        file << f[nf];
        // - number of crashes
        file << " " << Ncrashes[nf];
        // - number of no convergences
        file << " " << NnoConvs[nf];
        // - mean of the convergence time (s)
        double meT = eTs[nf] / NsamplesT;
        file << " " << meT;
        // - sigma of the convergence time (s)
        double seT = sqrt(max(eT2s[nf] / NsamplesT - meT * meT, 0.0));
        file << " " << seT;
        // - sigma of the computation of the mean convergence time (s)
        double smeT = seT / sqrt(NsamplesT);
        file << " " << smeT;
        // - mean of the error in orientation estimation (degrees)
        double meQ = eQs[nf] / NsamplesSim;
        file << " " << meQ;
        // - sigma of the computation of the mean error in orientation
        // estimation (degrees)
        double seQ = sqrt(max(eQ2s[nf] / NsamplesSim - meQ * meQ, 0.0));
        file << " " << seQ;
        // - sigma of the computation of the mean error in orientation
        // estimation (degrees)
        double smeQ = seQ / sqrt(NsamplesSim);
        file << " " << smeQ;
        file << "\n";
    }
    file << endl;

    file.close();
    cout << "Closed file" << endl;

    return;
}

// Method: testWrapper
// a wrapper to use the test() method with multithreading
// inputs:
//  myOE_Tester: pointer to the OE_Tester object with the testing parameters
//  myEstimator: pointer to the EKF object that we want to test
//  infoUpdateTime: the infoUpdateTime variable to use in the test() method
// outputs:
void Simulator::testWrapper(Simulator *myOE_Tester, EKF *myEstimator,
                            double infoUpdateTime)
{
    myOE_Tester->test(myEstimator, infoUpdateTime);

    return;
}

// Method: trajectoryStep
// generates a trajectory step from a previous known state
// inputs:
//  q: quaternion describing the real orientation
//  w: angular velocity measured in the sensor reference frame (rad/s), noise
//  added here
//  dtsim: time step of the simulation (s) nw: standard deviation of
//  the process noise for the angular velocity (rad*s^(-3/2))
// outputs:
//  q: new quaternion describing the real orientation of the system (it
//  satisfies qnew = q*qw) w: new angular velocity (rad/s)
void Simulator::trajectoryStep(Quat &q, Vec3 &w, double dtsim, double nw,
                               unsigned int *seed)
{
    // first of all we generate three random numbers normally distributed
    double sqrtdtsim = sqrt(dtsim);
    Vec3 noise = Vec3(myNormalRandom(seed, nw) * sqrtdtsim,
                      myNormalRandom(seed, nw) * sqrtdtsim,
                      myNormalRandom(seed, nw) * sqrtdtsim);
    w += noise;

    // we compute the next orientation using the previous orientation, and the
    // angular velocity
    //   w norm computation
    double wnorm = w.norm();
    //   we compute qw
    Quat qw;
    if (wnorm != 0.0) {
        double wdt05 = 0.5 * wnorm * dtsim;
        double swdt = sin(wdt05) / wnorm;
        qw.w() = cos(wdt05);
        qw.x() = w[0] * swdt;
        qw.y() = w[1] * swdt;
        qw.z() = w[2] * swdt;
    } else {
        qw = Quat(1, 0, 0, 0);
    }
    //   we compute the new state (q*qw,w)
    Quat qn = q * qw;
    q.coeffs() = qn.coeffs() / qn.norm();
}

// Method: IMU_Measurement
// generates an IMU measurement from a known state
// inputs:
//  q: quaternion describing the orientation of the system (transform vectors
//  from the sensor reference frame to the external reference frame)
//  w: real angular velocity measured in the sensor reference frame (rad/s)
//  v: real measured vector expressed in the external reference frame
//  nv: standard deviation of the process noise for the measured vector
//  rv: standard deviation of the vector measurements
//  rw: standard deviation of the gyroscope measurements (rad/s)
// outputs:
//  vm: simulated vector measurement
//  wm: simulated gyroscope measurement (rad/s)
void Simulator::get_measurement(Quat const &q, Vec3 const &w, Vec3 const &v,
                                double nv, double rv, double rw,
                                unsigned int *seed, Vec3 &vm, Vec3 &wm)
{
    // we compute the transposed rotation matrix
    const Mat3 R_transpose = q.toRotationMatrix().transpose();

    // then we generate three random numbers normally distributed
    Vec3 vv, nrv, nrw;
    for (int i = 0; i < 3; i++) {
        vv(i) = myNormalRandom(seed, nv) + v[i];  // qv + v
        nrv(i) = myNormalRandom(seed, rv);
        nrw(i) = myNormalRandom(seed, rw);
    }

    // we obtain the simulated measurements
    //   vm = RT*( qv + v ) + normrnd( 0 , rv , [3 1] );
    vm = R_transpose * vv + nrv;
    //   wm = w + normrnd( 0 , rw , [3 1] );
    wm = w + nrw;
}

// Function: thetaQ
// computes the angle between real and estimated orientations
// inputs:
//  qr: current real quaternion describing the orientation in the simulation
//  qe: current estimated quaternion describing the orientation in the
//  simulation
// (these quaternions transform vectors from the sensor reference frame to the
// external reference frame) outputs:
//  theta: angle between real and estimated orientations (rad)
double Simulator::thetaQ(Quat const &qr, Quat const &qe)
{
    // we compute the dot product of quaternions
    auto cr = qr.coeffs();
    auto ce = qe.coeffs();

    double delta0re = cr.dot(ce);
    //   the next computation only makes sense for a positive, and less than 1,
    //   delta0re
    if (delta0re < 0.0) delta0re = -delta0re;
    if (delta0re > 1.0) delta0re = 1.0;

    return 2.0 * acos(delta0re);
}

// Method: myNormalRandom
// generates a sample of a normal distribution
// inputs:
//  seed: the seed for the rand_r() method that produces uniform distributed
//  samples in the interval [0,1] sigma: standard deviation of the normal
//  distribution
// outputs:
//  rn: sample of the normal distribution (double)
double Simulator::myNormalRandom(unsigned int *seed, double sigma)
{
    // we generate two uniform random doubles
    double urand1 = (rand_r(seed) % 1000000 + 1) / 1000000.0;
    double urand2 = (rand_r(seed) % 1000000 + 1) / 1000000.0;
    // now we build the normal random double from the other two, and we return
    // it
    return sigma * sqrt(-2.0 * log(urand1)) * cos(2.0 * PI * urand2);
}

// Function: hasCrashed
// checks if there has been a crash in the algorithm
// inputs:
//  q: quaternion computed with the algorithm
// outputs:
//  crashed: boolean variable. It will be true if it has crashed; false if not
bool Simulator::hasCrashed(Quat const &q)
{
    double norm = q.norm();
    // we will consider a crash in the algorithm if:
    // - q is not of unit norm
    // - q is a NaN

    bool r = !(1.1 > norm);
    if (r) cout << "crashed with norm " << norm << endl;
    return r;
}

void Simulator::printMatrix(double *M, int n, int m)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            std::cout << std::setw(10) << std::setprecision(10) << " "
                      << M[i + j * n];
        std::cout << "\n";
    }
    std::cout << "\n";

    return;
}

// Function: test_computationalCost
// performs an evaluation of the computational cost of an estimator
// inputs:
//  Ntimes: number of updates to compute the computational cost
//  theEstimator: pointer to the EKF object to test
// outputs:
//  crashed: boolean variable. It will be true if it has crashed; false if not
void Simulator::test_computationalCost(long Ntimes, EKF *theEstimator)
{
    // we define the necessary entities
    double dt = 1.0e-2;
    double v[3] = {0.0, 0.0, 1.0};
    double vm[3] = {0.0, 1.0, 0.0};
    double wm[3] = {0.0, 0.0, 0.0};
    // we perform the computational cost evaluation
    double fm = 0.0;
    double fs = 0.0;
    const Vec3 vM = Map<Vec3>(v);
    const Vec3 vmM = Map<Vec3>(vm);
    const Vec3 wmM = Map<Vec3>(wm);
    for (long i = 0; i < Ntimes; i++) {
        clock_t begin = clock();
        theEstimator->update(vM, vmM, wmM, dt);
        clock_t end = clock();
        double f = CLOCKS_PER_SEC / double(end - begin);
        fm += f;
        fs += f * f;
    }
    fm /= Ntimes;
    fs = sqrt(fs / Ntimes - fm * fm);
    cout << fm << " " << fs << "\n";
}

#include <math.h>

/*
 *
 */
int main()
{
    EKF *e = new EKF();
    string title = "MEKF";

#define SIMULATION

#if defined SIMULATION
    std::string path = "./output/";
    std::string aftertitle = "";

#if defined R1e_2
    double R = 1.0e-2;
#elif defined R1e_4
    double R = 1.0e-4;
#elif defined R1e_6
    double R = 1.0e-6;
#endif

    double R = 1.0e-2;

    for (int k = 0; k < 3; k++) {
        aftertitle = std::to_string(R);
        try {
            std::cout << title << " " << R << "\n";

            Simulator myTester;
            myTester.set_title(path + title + aftertitle);
            const int Nfrequencies = 7;
            double f[Nfrequencies] = {2.0,   5.0,   10.0,  50.0,
                                      100.0, 500.0, 1000.0};
            // const int Nfrequencies = 1;
            // double f[Nfrequencies] = {50.0};
            myTester.set_f(f, Nfrequencies);
            myTester.set_convergenceThreshold(1.0);
            myTester.set_maxConvergenceUpdates(100000);
            myTester.set_Tsim(10.0);
            myTester.set_simStepsPerUpdate(100);
            myTester.set_Ntimes(10);
            myTester.set_QwMax(1.0e2);
            myTester.set_QvMax(1.0e0);
            myTester.set_Rw(R);
            myTester.set_Rv(R);

            // MAKE SURE THAT A FOLDER NAMED ./output/ EXISTS
            myTester.test(e, 2.0);
            // myTester.multithread_test( estimator[n] , 50.0 , 8 );

        } catch (int e) {
            return 1;
        }
        R *= 1.0e-2;
    }

#endif

    delete e;

    return 0;
}
