#include "ZeroWaitLQG.h"
#include <cmath>
#include <stdexcept>
#include <numeric>  // For accumulate function

LQGController::LQGController(double timeStep) : dt(timeStep), u(0.0) {
    // Initialize matrices and states with appropriate sizes or default values
}

void LQGController::setSystemMatrices(const std::vector<std::vector<double>>& A_, const std::vector<std::vector<double>>& B_) {
    A = A_;
    B = B_;
}

void LQGController::setLQRParameters(const std::vector<std::vector<double>>& Q_, const std::vector<std::vector<double>>& R_) {
    Q = Q_;
    R = R_;
    computeLQR();  // Compute LQR gain matrix K when Q and R are set
}

void LQGController::setKalmanParameters(const std::vector<std::vector<double>>& P_, const std::vector<std::vector<double>>& H_,
                                        const std::vector<std::vector<double>>& W_, const std::vector<std::vector<double>>& V_) {
    P = P_;
    H = H_;
    W = W_;
    V = V_;
}

void LQGController::setInitialState(const std::vector<double>& initialState) {
    x = initialState;
    x_hat = initialState;  // Initial estimate is assumed to be the same as the actual initial state
}

void LQGController::runController(const std::vector<double>& measurement) {
    // Step 1: Run Kalman Filter to estimate the state
    y = measurement;
    kalmanFilter();

    // Step 2: Compute the control input using LQR


    // Step 3: Update the system state (A * x + B * u)
    updateState();
}

double LQGController::getControlInput() {
    return u;
}
//
//// Helper Functions for Matrix Operations
//
//// Matrix-vector multiplication
std::vector<double> LQGController::matrixVectorMultiply(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) {
    if (mat[0].size() != vec.size()) {
        throw std::invalid_argument("Matrix columns must match vector size for multiplication.");
    }

    std::vector<double> result(mat.size(), 0.0);  // Result will have the same size as the number of rows in the matrix

    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += mat[i][j] * vec[j];  // Matrix row times vector column
        }
    }

    return result;
}
//
//
// Matrix-matrix multiplication
std::vector<std::vector<double>> LQGController::matrixMultiply(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2) {
    size_t n = mat1.size();
    size_t m = mat2[0].size();
    size_t p = mat2.size();
    std::vector<std::vector<double>> result(n, std::vector<double>(m, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < p; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

// Matrix transpose
std::vector<std::vector<double>> LQGController::transpose(const std::vector<std::vector<double>>& mat) {
    std::vector<std::vector<double>> result(mat[0].size(), std::vector<double>(mat.size(), 0.0));
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[i].size(); ++j) {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}
//
//// Matrix inverse (assuming 2x2 for simplicity)
std::vector<std::vector<double>> LQGController::inverse(const std::vector<std::vector<double>>& mat) {
    if (mat.size() != 2 || mat[0].size() != 2) {
        throw std::invalid_argument("Only 2x2 matrix inversion is supported");
    }
    double determinant = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    if (std::abs(determinant) < 1e-9) {
        throw std::invalid_argument("Matrix is singular, cannot be inverted");
    }
    double inv_det = 1.0 / determinant;
    return { {mat[1][1] * inv_det, -mat[0][1] * inv_det},
             {-mat[1][0] * inv_det, mat[0][0] * inv_det} };
}
//
//// Matrix Add
//// Overload for adding 2D matrices (matrix1 + matrix2)
std::vector<std::vector<double>> LQGController::matrixAdd(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2) {
    if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size()) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }

    std::vector<std::vector<double>> result(mat1.size(), std::vector<double>(mat1[0].size(), 0.0));

    for (size_t i = 0; i < mat1.size(); ++i) {
        for (size_t j = 0; j < mat1[0].size(); ++j) {
            result[i][j] = mat1[i][j] + mat2[i][j];
        }
    }

    return result;
}

std::vector<double> LQGController::vectorAdd(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size for addition.");
    }

    std::vector<double> result(vec1.size(), 0.0);
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }

    return result;
}


//// Matrix subtraction
//// Overload for subtracting 2D matrices
std::vector<std::vector<double>> LQGController::matrixSubtract(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2) {
    if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size()) {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction.");
    }

    std::vector<std::vector<double>> result(mat1.size(), std::vector<double>(mat1[0].size(), 0.0));

    for (size_t i = 0; i < mat1.size(); ++i) {
        for (size_t j = 0; j < mat1[0].size(); ++j) {
            result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }

    return result;
}

//// Overloaded matrixSubtract for 1D vectors
std::vector<double> LQGController::vectorSubtract(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction.");
    }

    std::vector<double> result(vec1.size(), 0.0);
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] - vec2[i];
    }
    return result;
}


// LQR Computation (simple version, using direct gain matrix for now)
void LQGController::computeLQR() {
    // Here, a direct method for calculating K can be implemented, such as solving the Riccati equation.
    // For simplicity, we'll assume that K has been precomputed.
    // In practice, you would solve the Algebraic Riccati Equation (ARE) here.

    // Placeholder for K
    K = { {1.0, 0.0, 0.0, 0.0},  // This is a simple feedback gain matrix
          {0.0, 1.0, 0.0, 0.0} };
}

// Kalman Filter Implementation
void LQGController::kalmanFilter() {
//     Predict step
    std::vector<double> x_pred = matrixVectorMultiply(A, x_hat);
    std::vector<std::vector<double>> P_pred = matrixAdd(matrixMultiply(A, matrixMultiply(P, transpose(A))), W);

//     Update step
    std::vector<double> y_pred = matrixVectorMultiply(H, x_pred);

//     Subtract vectors y and y_pred
    std::vector<double> y_err = vectorSubtract(y, y_pred);

    std::vector<std::vector<double>> S = matrixAdd(matrixMultiply(H, matrixMultiply(P_pred, transpose(H))), V);
    std::vector<std::vector<double>> Kf = matrixMultiply(P_pred, matrixMultiply(transpose(H), inverse(S)));

    x_hat = vectorAdd(x_pred, matrixVectorMultiply(Kf, y_err));
    P = matrixSubtract(P_pred, matrixMultiply(Kf, matrixMultiply(H, P_pred)));
}

// Update system state (x = A * x + B * u)
void LQGController::updateState() {
    std::vector<double> Ax = matrixVectorMultiply(A, x);
    std::vector<double> Bu = matrixVectorMultiply(B, {u});
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = Ax[i] + Bu[i];
    }
}

