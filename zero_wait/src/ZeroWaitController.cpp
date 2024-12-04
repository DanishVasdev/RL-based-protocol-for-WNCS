#include <omnetpp.h>
#include "inet/applications/base/ApplicationBase.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "inet/common/packet/Packet.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/networklayer/ipv4/Ipv4InterfaceData.h"
#include "inet/networklayer/common/L3AddressTag_m.h"
#include <fstream>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>

using namespace omnetpp;
using namespace inet;

class LQGController {
private:
    double dt;
    std::vector<std::vector<double>> A, B, Q, R, P, H, W, V, K;
    std::vector<double> x, x_hat, y;
    double u;

    // Helper functions for matrix operations
    std::vector<double> matrixVectorMultiply(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec);
    std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2);
    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& mat);
    std::vector<std::vector<double>> inverse(const std::vector<std::vector<double>>& mat);
    std::vector<std::vector<double>> matrixAdd(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2);
    std::vector<double> vectorAdd(const std::vector<double>& vec1, const std::vector<double>& vec2);
    std::vector<std::vector<double>> matrixSubtract(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2);
    std::vector<double> vectorSubtract(const std::vector<double>& vec1, const std::vector<double>& vec2);

    void computeLQR();
    void kalmanFilter(const std::vector<double>& measurement);
    void updateState();

public:
    LQGController(double timeStep);
    void setSystemMatrices(const std::vector<std::vector<double>>& A_, const std::vector<std::vector<double>>& B_);
    void setLQRParameters(const std::vector<std::vector<double>>& Q_, const std::vector<std::vector<double>>& R_);
    void setKalmanParameters(const std::vector<std::vector<double>>& P_, const std::vector<std::vector<double>>& H_,
                             const std::vector<std::vector<double>>& W_, const std::vector<std::vector<double>>& V_);
    void setInitialState(const std::vector<double>& initialState);
    void runController(const std::vector<double>& measurement);
    double getControlInput() const;
    std::vector<double> getStateEstimate() const;
};

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
    kalmanFilter(measurement);

    // Step 2: Compute the control input using LQR
    u = -matrixVectorMultiply(K, x_hat)[0];  // Assuming single input

    // Note: We no longer update the actual state here, as that's done in the real system
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
        EV_WARN<<vec1.size()<<" "<<vec2.size()<<endl;
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
//    K = { {1.0, 0.0, 0.0, 0.0},  // This is a simple feedback gain matrix
//          {0.0, 1.0, 0.0, 0.0} };
    K={{-2.4952,-3.7270,-59.8275,-13.2946}};
}
double LQGController::getControlInput() const {
    return u;
}
std::vector<double> LQGController::getStateEstimate() const {
    return x_hat;
}
// Kalman Filter Implementation
void LQGController::kalmanFilter(const std::vector<double>& measurement) {
    // Predict step
    std::vector<double> x_pred = matrixVectorMultiply(A, x_hat);
    std::vector<std::vector<double>> P_pred = matrixAdd(matrixMultiply(A, matrixMultiply(P, transpose(A))), W);

    // Update step
    std::vector<double> y_pred = matrixVectorMultiply(H, x_pred);
    std::vector<double> y = matrixVectorMultiply(H, measurement);
    std::vector<double> y_err = vectorSubtract(y, y_pred);

    std::vector<std::vector<double>> S = matrixAdd(matrixMultiply(H, matrixMultiply(P_pred, transpose(H))), V);
    std::vector<std::vector<double>> Kf = matrixMultiply(P_pred, matrixMultiply(transpose(H), inverse(S)));

    x_hat = vectorAdd(x_pred, matrixVectorMultiply(Kf, y_err));
    P = matrixSubtract(P_pred, matrixMultiply(Kf, matrixMultiply(H, P_pred)));
    EV_WARN<<"Kalman passed"<<endl;
}


// Update system state (x = A * x + B * u)
void LQGController::updateState() {
    std::vector<double> Ax = matrixVectorMultiply(A, x);
    std::vector<double> Bu = matrixVectorMultiply(B, {u});
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = Ax[i] + Bu[i];
    }
}


class ZeroWaitControllerApp : public ApplicationBase, public UdpSocket::ICallback
{
protected:
    UdpSocket socket;
    cMessage *timeoutTimer = nullptr;
    std::vector<int> sequenceNumbers;
    std::vector<bool> waitingForAcks;

    // Configuration
    int localPort, destPort;
    std::vector<L3Address> destAddrs;
    simtime_t sendInterval;

    // Controller specific
    LQGController lqgController;

    // RTT tracking
    std::vector<simtime_t> lastControllerPacketSentTimes;

    // Logging related variables
    std::ofstream csvFile;
    std::string csvFilename;

public:
    ZeroWaitControllerApp() : lqgController(0.02) {} // Assuming 20ms time step
    virtual ~ZeroWaitControllerApp();

protected:
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
    virtual void initialize(int stage) override;
    virtual void handleMessageWhenUp(cMessage *msg) override;
    virtual void finish() override;

    virtual bool isInitializeStage(int stage) const override { return stage == INITSTAGE_APPLICATION_LAYER; }
    virtual bool isModuleStartStage(int stage) const override { return stage == ModuleStartOperation::STAGE_APPLICATION_LAYER; }
    virtual bool isModuleStopStage(int stage) const override { return stage == ModuleStopOperation::STAGE_APPLICATION_LAYER; }
    virtual void handleStartOperation(LifecycleOperation *operation) override;
    virtual void handleStopOperation(LifecycleOperation *operation) override;
    virtual void handleCrashOperation(LifecycleOperation *operation) override;

    // UdpSocket::ICallback interface
    virtual void socketDataArrived(UdpSocket *socket, Packet *packet) override;
    virtual void socketErrorArrived(UdpSocket *socket, Indication *indication) override;
    virtual void socketClosed(UdpSocket *socket) override;

    void processPacket(Packet *pk);
    void sendUpdate(int destIndex);
    void handleTimeout();
    void logData(simtime_t currentTime, int destIndex, double controllerRTT, double lqgCost);
    void initializeCSVFile();
    void initializeLQGController();
};

Define_Module(ZeroWaitControllerApp);

ZeroWaitControllerApp::~ZeroWaitControllerApp()
{
    cancelAndDelete(timeoutTimer);
    if (csvFile.is_open()) {
        csvFile.close();
    }
}

void ZeroWaitControllerApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        localPort = par("localPort");
        destPort = par("destPort");
        sendInterval = par("sendInterval");

        timeoutTimer = new cMessage("timeoutTimer");

        initializeCSVFile();
        initializeLQGController();
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        socket.setOutputGate(gate("socketOut"));
        socket.bind(localPort);

        const char *destAddrsStr = par("destAddresses");
        cStringTokenizer tokenizer(destAddrsStr);
        const char *token;
        while ((token = tokenizer.nextToken()) != nullptr) {
            L3Address addr = L3AddressResolver().resolve(token);
            destAddrs.push_back(addr);
        }

        // Initialize sequence numbers and waiting flags for each destination
        sequenceNumbers.resize(destAddrs.size(), 0);
        waitingForAcks.resize(destAddrs.size(), false);
        lastControllerPacketSentTimes.resize(destAddrs.size(), simTime());

        socket.setCallback(this);

        auto ift = getModuleFromPar<IInterfaceTable>(par("interfaceTableModule"), this);
        auto ie = ift->findInterfaceByName("wlan0");
        if (ie) {
            auto ipv4Data = ie->findProtocolData<Ipv4InterfaceData>();
            if (ipv4Data) {
                EV_INFO << "Controller " << getParentModule()->getFullName() << " assigned IP: " << ipv4Data->getIPAddress() << endl;
            }
        }

        for (size_t i = 0; i < destAddrs.size(); i++) {
            EV_INFO << "Controller " << getParentModule()->getFullName() << " sending to destination " << i << ": " << destAddrs[i] << endl;
        }
    }
}

void ZeroWaitControllerApp::initializeLQGController()
{
    // Initialize LQG controller with appropriate matrices
    // These are placeholder values and should be adjusted based on your system
    std::vector<std::vector<double>> A = {{1, 0.02, 0, 0}, {0, 1, 0.178, 0}, {0, 0, 1, 0.02}, {0, 0,0.3924, 1}};
    std::vector<std::vector<double>> B = {{0.0}, {0.0182}, {0}, {-0.0364}};
    lqgController.setSystemMatrices(A, B);

    std::vector<std::vector<double>> Q = {{1, 0, 0, 0}, {0, 0.1, 0, 0}, {0, 0, 100, 0}, {0, 0, 0, 10}};
    std::vector<std::vector<double>> R = {{0.1}};
    lqgController.setLQRParameters(Q, R);

    std::vector<std::vector<double>> P = {{0.1, 0, 0, 0}, {0, 0.1, 0, 0}, {0, 0, 0.1, 0}, {0, 0, 0, 0.1}};
    std::vector<std::vector<double>> H = {{1, 0, 0, 0}, {0, 0, 1, 0}};
    std::vector<std::vector<double>> W = {{0.01, 0, 0, 0}, {0, 0.01, 0, 0}, {0, 0, 0.01, 0}, {0, 0, 0, 0.01}};
    std::vector<std::vector<double>> V = {{0.01, 0}, {0, 0.01}};
    lqgController.setKalmanParameters(P, H, W, V);

    std::vector<double> initialState = {0, 0, 0, 0};
    lqgController.setInitialState(initialState);
}

void ZeroWaitControllerApp::initializeCSVFile()
{
    std::time_t now = std::time(nullptr);
    char filename[100];
    std::strftime(filename, sizeof(filename), "controller_log_%Y%m%d_%H%M%S.csv", std::localtime(&now));
    csvFilename = filename;
    csvFile.open(csvFilename);
    if (csvFile.is_open()) {
        csvFile << "Time,Destination,Controller RTT,LQG Cost" << std::endl;
    } else {
        throw cRuntimeError("Unable to open CSV file for writing");
    }
}

void ZeroWaitControllerApp::handleMessageWhenUp(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        if (msg == timeoutTimer) {
            handleTimeout();
        }
    }
    else {
        socket.processMessage(msg);
    }
}

void ZeroWaitControllerApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    processPacket(packet);
}

void ZeroWaitControllerApp::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Socket error arrived" << endl;
    delete indication;
}

void ZeroWaitControllerApp::socketClosed(UdpSocket *socket)
{
    EV_WARN << "Socket closed" << endl;
}

void ZeroWaitControllerApp::processPacket(Packet *pk)
{
    auto macQueue = findModuleByPath("^.wlan[0].mac.queue");
        int queueLength = 0;

        if (macQueue) {
            cQueue* queue = check_and_cast<cQueue*>(macQueue);
            queueLength = queue->getLength();
        }

        EV_INFO << "Queue length at the time of receiving packet: " << queueLength << endl;
    auto chunk = pk->peekAtFront<BytesChunk>();
    auto bytes = chunk->getBytes();
    uint8_t receivedSeq = bytes[0];

    // Find which destination this packet is from
    L3Address srcAddr = pk->getTag<L3AddressInd>()->getSrcAddress();
    auto it = std::find(destAddrs.begin(), destAddrs.end(), srcAddr);
    if (it == destAddrs.end()) {
        EV_WARN << "Received packet from unknown source: " << srcAddr << endl;
        delete pk;
        return;
    }
    int destIndex = std::distance(destAddrs.begin(), it);

    EV_WARN << "Processing Packet at Controller from destination " << destIndex << endl;
    EV_INFO << "Received seq " << (int)receivedSeq << " expected " << sequenceNumbers[destIndex] << endl;

    simtime_t currentTime = simTime();
    double controllerRTT = (currentTime - lastControllerPacketSentTimes[destIndex]).dbl();
    EV_INFO << "Controller: Packet received from destination " << destIndex << ", RTT: " << controllerRTT << endl;

    waitingForAcks[destIndex] = false;

    // Extract the full state from the received packet
      std::vector<double> measuredState;
      for (size_t i = 1; i < bytes.size(); i += 2) {
          uint16_t intValue = (bytes[i] << 8) | bytes[i+1];
          double value = (static_cast<double>(intValue) / 1000.0) - 10;
          measuredState.push_back(value);
      }

      // Ensure we have a complete state measurement (assuming 4 state variables)
      if (measuredState.size() != 4) {
          EV_WARN << "Received incomplete state measurement. Expected 4 values, got " << measuredState.size() << endl;
          delete pk;
          return;
      }

      EV_INFO << "Received state from destination " << destIndex << ": "
              << "x=" << measuredState[0] << ", x_dot=" << measuredState[1]
              << ", theta=" << measuredState[2] << ", theta_dot=" << measuredState[3] << endl;

      // Run LQG controller with the measured state
      lqgController.runController(measuredState);

      // Get the updated state estimate from the controller
      std::vector<double> estimatedState = lqgController.getStateEstimate();

      // Calculate LQG cost using the estimated state
      double lqgCost = estimatedState[0]*estimatedState[0] + 100*estimatedState[2]*estimatedState[2];

    logData(currentTime, destIndex, controllerRTT, lqgCost);

    // Immediately send a new control update to this destination
    sendUpdate(destIndex);

    delete pk;
}

void ZeroWaitControllerApp::sendUpdate(int destIndex)
{
    auto packet = new Packet("ZeroWaitControllerData");
    const auto& payload = makeShared<BytesChunk>();
    std::vector<uint8_t> data;

    data.push_back(static_cast<uint8_t>(sequenceNumbers[destIndex]));

    // Controller sends force
    double force = lqgController.getControlInput();
    force = std::max(-10.0, std::min(10.0, force));  // Limit force to [-10, 10]
    uint16_t forceInt = static_cast<uint16_t>((force + 10) * 1000);  // Convert to fixed-point
    data.push_back(forceInt >> 8);
    data.push_back(forceInt & 0xFF);
    lastControllerPacketSentTimes[destIndex] = simTime();
    EV_WARN << "Controller update sent to destination " << destIndex << " (" << destAddrs[destIndex] << ") with seq no " << sequenceNumbers[destIndex] << endl;

    payload->setBytes(data);
    packet->insertAtBack(payload);
    EV_INFO << "Sending packet to destination " << destIndex << " (" << destAddrs[destIndex] << ")" << endl;
    socket.sendTo(packet, destAddrs[destIndex], destPort);
    waitingForAcks[destIndex] = true;
    sequenceNumbers[destIndex]++;

    // Set a timeout for this destination
    if (!timeoutTimer->isScheduled()) {
        scheduleAt(simTime() + 3.0, timeoutTimer);
    }
}
void ZeroWaitControllerApp::handleTimeout()
{
    EV_INFO << "Timeout occurred" << endl;
    for (size_t i = 0; i < waitingForAcks.size(); i++) {
        if (waitingForAcks[i]) {
            EV_INFO << "No response from destination " << i << ", resetting waiting state" << endl;
            waitingForAcks[i] = false;
        }
    }
    // Reschedule the timeout timer if any destinations are still waiting
    if (std::any_of(waitingForAcks.begin(), waitingForAcks.end(), [](bool v) { return v; })) {
        scheduleAt(simTime() + 3.0, timeoutTimer);
    }
}

void ZeroWaitControllerApp::logData(simtime_t currentTime, int destIndex, double controllerRTT, double lqgCost)
{
    if (csvFile.is_open()) {
        csvFile << currentTime.dbl() << "," << destIndex << "," << controllerRTT << "," << lqgCost << std::endl;
        EV_WARN << "Logging data for destination " << destIndex << ": " << controllerRTT << " " << lqgCost << std::endl;
    } else {
        EV_ERROR << "CSV file is not open for writing" << endl;
    }
}

void ZeroWaitControllerApp::handleStartOperation(LifecycleOperation *operation)
{
    // Controller doesn't initiate communication, it waits for sensor data
}

void ZeroWaitControllerApp::handleStopOperation(LifecycleOperation *operation)
{
    cancelEvent(timeoutTimer);
    socket.close();
}

void ZeroWaitControllerApp::handleCrashOperation(LifecycleOperation *operation)
{
    cancelEvent(timeoutTimer);
    if (socket.isOpen())
        socket.destroy();
}

void ZeroWaitControllerApp::finish()
{
    ApplicationBase::finish();
    if (csvFile.is_open()) {
        csvFile.close();
    }
}


