/*
 * LQGController.h
 *
 *  Created on: Sep 27, 2024
 *      Author: HP
 */

#ifndef SRC_LQGCONTROLLER_H_
#define SRC_LQGCONTROLLER_H_


#include <vector>
#include <iostream>

class LQGController {
private:
    // State-Space Model Matrices
    std::vector<std::vector<double>> A; // System dynamics matrix
    std::vector<std::vector<double>> B; // Control input matrix
    std::vector<std::vector<double>> Q; // State cost matrix
    std::vector<std::vector<double>> R; // Control cost matrix

    // Kalman Filter Matrices
    std::vector<std::vector<double>> P; // Error covariance matrix
    std::vector<std::vector<double>> H; // Measurement matrix
    std::vector<std::vector<double>> W; // Process noise covariance matrix
    std::vector<std::vector<double>> V; // Measurement noise covariance matrix

    // LQR Gain Matrix
    std::vector<std::vector<double>> K;

    // System States
    std::vector<double> x;    // State vector [x, theta, etc.]
    std::vector<double> x_hat; // Estimated state vector from Kalman filter

    // Measurement vector
    std::vector<double> y;

    // Control input
    double u;

    // Time step
    double dt;

    // Helper functions for matrix operations
    std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2);
    std::vector<double> matrixVectorMultiply(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec);
    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& mat);
    std::vector<std::vector<double>> inverse(const std::vector<std::vector<double>>& mat);
    std::vector<double> vectorAdd(const std::vector<double>& vec1, const std::vector<double>& vec2);
    std::vector<std::vector<double>> matrixAdd(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2);
    std::vector<double> vectorSubtract(const std::vector<double>& vec1, const std::vector<double>& vec2);
    std::vector<std::vector<double>> matrixSubtract(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2);

    void computeLQR();  // Computes the optimal LQR gain matrix K
    void kalmanFilter();  // Kalman filter for state estimation
    void updateState();  // Updates the system state using x = A * x + B * u

public:
    LQGController(double timeStep);

    // Setters for matrices and initial conditions
    void setSystemMatrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
    void setLQRParameters(const std::vector<std::vector<double>>& Q, const std::vector<std::vector<double>>& R);
    void setKalmanParameters(const std::vector<std::vector<double>>& P, const std::vector<std::vector<double>>& H,
                             const std::vector<std::vector<double>>& W, const std::vector<std::vector<double>>& V);
    void setInitialState(const std::vector<double>& initialState);

    // Run the controller
    void runController(const std::vector<double>& measurement);

    // Get the control input
    double getControlInput();
};

#endif /* SRC_LQGCONTROLLER_H_ */
