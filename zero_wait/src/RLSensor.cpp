#include <omnetpp.h>
#include "inet/applications/base/ApplicationBase.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "inet/common/packet/Packet.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/networklayer/ipv4/Ipv4InterfaceData.h"
#include <fstream>
#include <ctime>
#include <cmath>
#include <random>
#include <map>
using namespace omnetpp;
using namespace inet;

class CartPole {
private:
    double x;        // Cart position
    double x_dot;    // Cart velocity
    double theta;    // Pole angle
    double theta_dot;// Pole angular velocity
    double current_force; // Current force being applied

    double g = 9.8;   // Gravity
    double m_c = 1.0; // Mass of the cart
    double m_p = 0.1; // Mass of the pole
    double l = 0.5;   // Half-length of the pole
    double dt = 0.02; // Time step

public:
    CartPole() : x(0), x_dot(0), theta(0.1), theta_dot(0), current_force(0) {}

    void setForce(double force) {
        current_force = force;
    }

    void update() {
        step(current_force);
    }

    void step(double force) {
        current_force = force;
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);

        double temp = (force + m_p * l * theta_dot * theta_dot * sin_theta) / (m_c + m_p);
        double theta_acc = (g * sin_theta - cos_theta * temp) /
                           (l * (4.0/3.0 - m_p * cos_theta * cos_theta / (m_c + m_p)));
        double x_acc = temp - m_p * l * theta_acc * cos_theta / (m_c + m_p);

        x += x_dot * dt;
        x_dot += x_acc * dt;
        theta += theta_dot * dt;
        theta_dot += theta_acc * dt;
    }

    bool is_done() const {
        return std::abs(x) > 5 || std::abs(theta) > 0.3;
    }

    std::vector<double> get_state() const {
        return {x, x_dot, theta, theta_dot};
    }
};

class MDPTransferProtocol {
private:
    double epsilon;
    double learningRate;
    double discountFactor;
    std::map<std::string, double> qValues;
    std::mt19937 rng;



public:
    std::string getState(const std::vector<double>& cartPoleState, double networkCongestion) {
            int x = static_cast<int>(cartPoleState[0] * 10);
            int theta = static_cast<int>(cartPoleState[2] * 100);
            int congestion = static_cast<int>(networkCongestion * 10);
            return std::to_string(x) + "," + std::to_string(theta) + "," + std::to_string(congestion);
        }
    MDPTransferProtocol(double eps = 0.1, double alpha = 0.1, double gamma = 0.9)
        : epsilon(eps), learningRate(alpha), discountFactor(gamma), rng(std::random_device{}()) {}

    bool shouldSendPacket(const std::vector<double>& cartPoleState, double networkCongestion) {
        std::string state = getState(cartPoleState, networkCongestion);

        // Epsilon-greedy policy
        if (std::uniform_real_distribution<>(0, 1)(rng) < epsilon) {
            return std::uniform_real_distribution<>(0, 1)(rng) < 0.5;
        }

        // If Q-values are uninitialized, default to sending
        if (qValues.find(state + ",send") == qValues.end()) qValues[state + ",send"] = 0.0;
        if (qValues.find(state + ",wait") == qValues.end()) qValues[state + ",wait"] = 0.0;

        return qValues[state + ",send"] >= qValues[state + ",wait"];
    }

    void updateQValues(const std::string& prevStateStr, bool actionTaken,
                       const std::string& newStateStr, double reward) {
        std::string prevAction = actionTaken ? "send" : "wait";

        // Initialize Q-values if they don't exist
        if (qValues.find(newStateStr + ",send") == qValues.end()) qValues[newStateStr + ",send"] = 0.0;
        if (qValues.find(newStateStr + ",wait") == qValues.end()) qValues[newStateStr + ",wait"] = 0.0;

        double maxFutureQ = std::max(qValues[newStateStr + ",send"], qValues[newStateStr + ",wait"]);

        qValues[prevStateStr + "," + prevAction] =
            (1 - learningRate) * qValues[prevStateStr + "," + prevAction] +
            learningRate * (reward + discountFactor * maxFutureQ);
    }
};

class RLSensorApp : public ApplicationBase, public UdpSocket::ICallback
{
  protected:
    UdpSocket socket;
    cMessage *sendTimer = nullptr;
    cMessage *timeoutTimer = nullptr;
    cMessage *cartPoleUpdateTimer = nullptr;
    int sequenceNumber = 0;
    bool waitingForAck = false;

    // Configuration
    int localPort, destPort;
    L3Address destAddr;
    simtime_t sendInterval;
    simtime_t cartPoleUpdateInterval;
    // Cart-pole specific
    CartPole cartPole;

    // RTT tracking
    simtime_t lastSensorPacketSentTime;
    simtime_t lastRTT;
    double avgRTT;
    double avgRTTAlpha = 0.125;  // Smoothing factor for exponential moving average

    // Logging related variables
    std::ofstream csvFile;
    std::string csvFilename;

    // MDP related
    MDPTransferProtocol mdp;
    std::vector<double> prevState;
    double prevCongestion;
    bool prevActionTaken;
    bool firstAction;  // New variable to handle the first action

  public:
    RLSensorApp() : mdp(0.1, 0.1, 0.9), avgRTT(0.1), firstAction(true) {
        cartPoleUpdateInterval = 0.02;
    }  // Initialize avgRTT to 0.1s
    virtual ~RLSensorApp();
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
    void sendUpdate();
    void handleTimeout();
    void logData(simtime_t currentTime, double sensorRTT, double controllerRTT, double lqgCost,double reward);
    void initializeCSVFile();
    double estimateNetworkCongestion();
    double calculateReward(const std::vector<double>& state, double congestion, bool actionTaken);
    double calculateSTLReward(const std::vector<double>& state);
};

Define_Module(RLSensorApp);

RLSensorApp::~RLSensorApp()
{
    cancelAndDelete(sendTimer);
    cancelAndDelete(timeoutTimer);
    cancelAndDelete(cartPoleUpdateTimer);
    if (csvFile.is_open()) {
        csvFile.close();
    }
}

void RLSensorApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        localPort = par("localPort");
        destPort = par("destPort");
        sendInterval = par("sendInterval");

        sendTimer = new cMessage("sendTimer");
        timeoutTimer = new cMessage("timeoutTimer");
        cartPoleUpdateTimer = new cMessage("cartPoleUpdateTimer");
        initializeCSVFile();

        avgRTT = 0.1;  // Initialize avgRTT to a typical RTT value (100ms)
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        socket.setOutputGate(gate("socketOut"));
        socket.bind(localPort);
        if (cartPoleUpdateTimer->isScheduled()) {
            cancelEvent(cartPoleUpdateTimer);
        }
        scheduleAt(simTime() + cartPoleUpdateInterval, cartPoleUpdateTimer);
        const char *destAddrs = par("destAddresses");
        cStringTokenizer tokenizer(destAddrs);
        const char *token = nullptr;
        if ((token = tokenizer.nextToken()) != nullptr) {
            destAddr = L3AddressResolver().resolve(token);
        }

        socket.setCallback(this);

        auto ift = getModuleFromPar<IInterfaceTable>(par("interfaceTableModule"), this);
        auto ie = ift->findInterfaceByName("wlan0");
        if (ie) {
            auto ipv4Data = ie->findProtocolData<Ipv4InterfaceData>();
            if (ipv4Data) {
            }
        }

        EV_INFO << "Sensor " << getParentModule()->getFullName() << " sending to: " << destAddr << endl;

        if (!sendTimer->isScheduled()) {
            scheduleAt(simTime() + sendInterval, sendTimer);
        }

        prevState = cartPole.get_state();
        prevCongestion = estimateNetworkCongestion();
        prevActionTaken = false;
        firstAction = true;
    }
}

void RLSensorApp::handleMessageWhenUp(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        if (msg == sendTimer) {
            if (!waitingForAck) {
                sendUpdate();
            }
            cancelEvent(sendTimer);
            scheduleAt(simTime() + sendInterval, sendTimer);
        }
        else if (msg == timeoutTimer) {
            handleTimeout();
        }
        else if (msg == cartPoleUpdateTimer) {
            // Update cart-pole state
            cartPole.update();

            // Check if simulation should end
            if (cartPole.is_done()) {
                EV_WARN << "Cart-pole simulation ended during continuous update" << endl;

                // Reset the cart-pole for a new episode
                cartPole = CartPole();
                firstAction = true;

                // Reinitialize state tracking
                prevState = cartPole.get_state();
                prevCongestion = estimateNetworkCongestion();
                prevActionTaken = false;
            }

            // Cancel existing timer before scheduling new one
            cancelEvent(cartPoleUpdateTimer);
            scheduleAt(simTime() + cartPoleUpdateInterval, cartPoleUpdateTimer);
        }
    }
    else {
        socket.processMessage(msg);
    }
}

void RLSensorApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    processPacket(packet);
}

void RLSensorApp::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Socket error arrived" << endl;
    delete indication;
}

void RLSensorApp::socketClosed(UdpSocket *socket)
{
    EV_WARN << "Socket closed" << endl;
}

void RLSensorApp::initializeCSVFile()
{
    std::time_t now = std::time(nullptr);
    char filename[100];
    std::strftime(filename, sizeof(filename), "sensor_log_%Y%m%d_%H%M%S", std::localtime(&now));
    csvFilename = filename+static_cast<std::string>(getParentModule()->getFullName())+".csv";
    csvFile.open(csvFilename);
    if (csvFile.is_open()) {
        csvFile << "Time,Sensor RTT,LQG Cost,Reward" << std::endl;
    } else {
        throw cRuntimeError("Unable to open CSV file for writing");
    }
}

void RLSensorApp::sendUpdate()
{
    auto state = cartPole.get_state();
    double congestion = estimateNetworkCongestion();

    // Force send update if the state deviates significantly (e.g., pole angle exceeds threshold)
    bool forceSend = std::abs(state[2]) > 0.2;  // Threshold for pole angle deviation

    if (mdp.shouldSendPacket(state, congestion) || forceSend) {
        auto packet = new Packet("RLSensorData");
        const auto& payload = makeShared<BytesChunk>();
        std::vector<uint8_t> data;

        data.push_back(static_cast<uint8_t>(sequenceNumber));

        for (double value : state) {
            uint16_t intValue = static_cast<uint16_t>((value + 10) * 1000);  // Convert to fixed-point
            data.push_back(intValue >> 8);
            data.push_back(intValue & 0xFF);
        }
        lastSensorPacketSentTime = simTime();
        EV_WARN << "Sensor update sent to " << destAddr << " with seq no " << sequenceNumber << endl;

        payload->setBytes(data);
        packet->insertAtBack(payload);
        EV_INFO << "Sending packet to " << destAddr << endl;
        socket.sendTo(packet, destAddr, destPort);
        waitingForAck = true;
        sequenceNumber++;

        cancelEvent(timeoutTimer);
        scheduleAt(simTime() + 0.1, timeoutTimer);  // 3 second timeout

        prevActionTaken = true;
        EV_INFO <<getParentModule()->getFullName()<< "MDP decided to send update" << endl;
    } else {
        EV_INFO <<getParentModule()->getFullName()<< "MDP decided not to send update due to network conditions" << endl;
        prevActionTaken = false;
    }

    // Update Q-values if not the first action
    if (!firstAction) {
        std::string prevStateStr = mdp.getState(prevState, prevCongestion);
        std::string newStateStr = mdp.getState(state, congestion);
        //double reward = calculateSTLReward(state, congestion, prevActionTaken);
        double reward = calculateSTLReward(state);
        mdp.updateQValues(prevStateStr, prevActionTaken, newStateStr, reward);
    } else {
        firstAction = false;
    }

    prevState = state;
    prevCongestion = congestion;
}
double smoothMin(double a, double b, double eta = 300) {
    return (a + b - std::abs(a - b)) / 2.0;
}

double RLSensorApp::estimateNetworkCongestion() {
    // Use the exponential moving average of RTT to estimate congestion
    double normalizedRTT = avgRTT / 0.01;  // Normalize assuming 100ms is a "normal" RTT
    return std::min(1.0, std::max(0.0, normalizedRTT - 1.0));  // Clamp between 0 and 1
}
double RLSensorApp::calculateSTLReward(const std::vector<double>& state) {
    double x = state[0];
    double theta = state[2];
    double x_dot = state[1];
    double theta_dot = state[3];

    // STL Robustness
    double safetyRobustness = smoothMin(3 - std::abs(x), 0.3 - std::abs(theta));
    double livenessRobustness = smoothMin(1 - std::abs(x), 0.1 - std::abs(theta));

    // Penalize deviations (control error) and excessive control effort
    double statePenalty = x * x + 100 * theta * theta;


    // Combine STL robustness and penalties
    double reward = safetyRobustness + 0.5 * livenessRobustness ;

    return reward;
}

double RLSensorApp::calculateReward(const std::vector<double>& state, double congestion, bool actionTaken)
{
    // Penalize deviations in state and network congestion
    double statePenalty = state[0]*state[0]+100*state[2]*state[2]; // Emphasize pole angle deviation
    double congestionPenalty = congestion * 100; // Penalize network congestion

    // Reward sending updates when the state deviation is large
    double actionReward = 0.0;
    if (actionTaken && statePenalty > 1) {
        actionReward = 1.0; // Positive reward for sending updates when needed
    } else if (!actionTaken && statePenalty > 1) {
        actionReward = -1.0; // Negative reward for not sending updates when needed
    }

    return -statePenalty + actionReward;
}

void RLSensorApp::processPacket(Packet *pk)
{
    auto chunk = pk->peekAtFront<BytesChunk>();
    auto bytes = chunk->getBytes();
    uint8_t receivedSeq = bytes[0];
    EV_WARN << "Processing Packet at Sensor" << endl;
    EV_INFO << "Received seq " << (int)receivedSeq << " expected " << sequenceNumber - 1 << endl;
    if (receivedSeq == (sequenceNumber - 1)) {
        simtime_t currentTime = simTime();
        lastRTT = currentTime - lastSensorPacketSentTime;

        // Update the exponential moving average of RTT
        if (avgRTT == 0) {
            avgRTT = lastRTT.dbl();
        } else {
            avgRTT = (1 - avgRTTAlpha) * avgRTT + avgRTTAlpha * lastRTT.dbl();
        }

        EV_INFO << "Sensor: ACK received for sequence number " << (int)receivedSeq
                << ", RTT: " << lastRTT << ", Avg RTT: " << avgRTT << endl;

        waitingForAck = false;
        cancelEvent(timeoutTimer);

        // Sensor receives force
        uint16_t forceInt = (bytes[1] << 8) | bytes[2];
        double force = (static_cast<double>(forceInt) / 1000.0) - 10;
        cartPole.setForce(force);
        auto newState = cartPole.get_state();
        EV_INFO << "Applied force: " << force << ", New state: x=" << newState[0]
                << ", theta=" << newState[2] << endl;

        // Calculate LQG cost (simplified version)
        double lqgCost = newState[0]*newState[0] + 100*newState[2]*newState[2];  // Example cost function

        if (cartPole.is_done()) {
            EV_WARN << "Cart-pole simulation ended" << endl;

            // Add negative reward for end of episode
            double negativeReward = -100.0; // Define the end-of-episode penalty
            std::string prevStateStr = mdp.getState(prevState, prevCongestion);
            std::string newStateStr = mdp.getState(newState, prevCongestion);
            mdp.updateQValues(prevStateStr, prevActionTaken, newStateStr, negativeReward);

            // Log the final state
            logData(currentTime, lastRTT.dbl(), 0.0, lqgCost, negativeReward);

            // Reset the cart-pole for a new episode
            cartPole = CartPole();  // Reset to initial state
            firstAction = true;     // Reset MDP learning for the new episode

            // Reinitialize state tracking
            prevState = cartPole.get_state();
            prevCongestion = estimateNetworkCongestion();
            prevActionTaken = false;
        } else {
            // Update Q-values normally if not end of episode
            double congestion = estimateNetworkCongestion();
            std::string prevStateStr = mdp.getState(prevState, prevCongestion);
            std::string newStateStr = mdp.getState(newState, congestion);
            //double reward = calculateReward(newState, congestion, prevActionTaken);
            double reward = calculateSTLReward(newState);
            mdp.updateQValues(prevStateStr, prevActionTaken, newStateStr, reward);

            // Log the normal step data
            logData(currentTime, lastRTT.dbl(), 0.0, lqgCost, reward);
        }

        prevState = newState;
        prevCongestion = estimateNetworkCongestion();
        prevActionTaken = false;
    }

    delete pk;
}

void RLSensorApp::handleStartOperation(LifecycleOperation *operation)
{
    scheduleAt(simTime() + sendInterval, sendTimer);
    scheduleAt(simTime() + cartPoleUpdateInterval, cartPoleUpdateTimer);
}

void RLSensorApp::handleStopOperation(LifecycleOperation *operation)
{
    cancelEvent(sendTimer);
    cancelEvent(timeoutTimer);
    cancelEvent(cartPoleUpdateTimer);
    socket.close();
}

void RLSensorApp::handleCrashOperation(LifecycleOperation *operation)
{
    cancelEvent(sendTimer);
    cancelEvent(timeoutTimer);
    cancelEvent(cartPoleUpdateTimer);
    if (socket.isOpen())
        socket.destroy();
}

void RLSensorApp::handleTimeout()
{
    EV_INFO << "Timeout occurred" << endl;
    waitingForAck = false;
    // Do not retransmit immediately; wait for the next scheduled send
}

void RLSensorApp::logData(simtime_t currentTime, double sensorRTT, double controllerRTT, double lqgCost,double reward)
{
    if (csvFile.is_open()) {
        csvFile << currentTime.dbl() << "," << sensorRTT << "," << lqgCost <<","<<reward<< std::endl;
        EV_WARN << "Logging data " << sensorRTT << " " << lqgCost << std::endl;
    } else {
        EV_ERROR << "CSV file is not open for writing" << endl;
    }
}

void RLSensorApp::finish()
{
    ApplicationBase::finish();
    if (csvFile.is_open()) {
        csvFile.close();
    }
}
