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
#include <vector>
#include <deque>
#include <algorithm>

using namespace omnetpp;
using namespace inet;
using namespace std;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);

double relu(double x) {
    return std::max(0.0, x);
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// Neural Network Layer
class Layer {
public:
    vector<vector<double>> weights;
    vector<double> biases;

    // Variables to store values during the forward pass
    vector<double> last_input;
    vector<double> last_z;
    vector<double> last_output;

    Layer(int inputs, int outputs) {
        weights = vector<vector<double>>(outputs, vector<double>(inputs));
        biases = vector<double>(outputs, 0.0);
        for (auto& row : weights)
            for (double& w : row)
                w = (dis(gen) - 0.5) * 0.1;
    }

    vector<double> forward(const vector<double>& input) {
        last_input = input;
        last_z = vector<double>(weights.size(), 0.0);
        last_output = vector<double>(weights.size(), 0.0);

        for (int i = 0; i < weights.size(); ++i) {
            for (int j = 0; j < input.size(); ++j) {
                last_z[i] += weights[i][j] * input[j];
            }
            last_z[i] += biases[i];
            last_output[i] = relu(last_z[i]);
            last_output[i] *= 0.9; // Dropout-like effect
        }
        return last_output;
    }
};


// DQN Network with two hidden layers
class DQNNetwork {
public:
    Layer hidden1, hidden2, output;

    DQNNetwork(int input_size, int hidden_size, int output_size)
        : hidden1(input_size, hidden_size),
          hidden2(hidden_size, hidden_size),
          output(hidden_size, output_size) {}

    vector<double> predict(const vector<double>& state) {
        auto h1 = hidden1.forward(state);
        auto h2 = hidden2.forward(h1);
        return output.forward(h2);
    }

    void update_weights(const vector<double>& state, const vector<double>& target, double learning_rate) {
        // Forward pass
        auto h1 = hidden1.forward(state);
        auto h2 = hidden2.forward(h1);
        auto predicted = output.forward(h2);

        // Compute output layer gradient
        vector<double> output_grad(predicted.size());
        for (int i = 0; i < predicted.size(); ++i) {
            double error = target[i] - predicted[i];
            double d_relu = relu_derivative(output.last_z[i]) * 0.9;  // Adjusted derivative due to scaling
            output_grad[i] = error * d_relu;
        }

        // Backpropagate to hidden2
        vector<double> hidden2_grad(hidden2.weights.size(), 0.0);
        for (int i = 0; i < hidden2.weights.size(); ++i) {
            for (int j = 0; j < output.weights.size(); ++j) {
                hidden2_grad[i] += output_grad[j] * output.weights[j][i];
            }
            double d_relu = relu_derivative(hidden2.last_z[i]) * 0.9;
            hidden2_grad[i] *= d_relu;
        }

        // Backpropagate to hidden1
        vector<double> hidden1_grad(hidden1.weights.size(), 0.0);
        for (int i = 0; i < hidden1.weights.size(); ++i) {
            for (int j = 0; j < hidden2.weights.size(); ++j) {
                hidden1_grad[i] += hidden2_grad[j] * hidden2.weights[j][i];
            }
            double d_relu = relu_derivative(hidden1.last_z[i]) * 0.9;
            hidden1_grad[i] *= d_relu;
        }

        // Update output layer weights and biases
        for (int i = 0; i < output.weights.size(); ++i) {
            for (int j = 0; j < output.weights[i].size(); ++j) {
                output.weights[i][j] += learning_rate * output_grad[i] * output.last_input[j];
            }
            output.biases[i] += learning_rate * output_grad[i];
        }

        // Update hidden2 layer weights and biases
        for (int i = 0; i < hidden2.weights.size(); ++i) {
            for (int j = 0; j < hidden2.weights[i].size(); ++j) {
                hidden2.weights[i][j] += learning_rate * hidden2_grad[i] * hidden2.last_input[j];
            }
            hidden2.biases[i] += learning_rate * hidden2_grad[i];
        }

        // Update hidden1 layer weights and biases
        for (int i = 0; i < hidden1.weights.size(); ++i) {
            for (int j = 0; j < hidden1.weights[i].size(); ++j) {
                hidden1.weights[i][j] += learning_rate * hidden1_grad[i] * hidden1.last_input[j];
            }
            hidden1.biases[i] += learning_rate * hidden1_grad[i];
        }
    }

};

// Replay Buffer with Priority
class ReplayBuffer {
public:
    deque<tuple<vector<double>, int, double, vector<double>, double>> buffer;
    vector<double> priorities;
    int max_size;

    ReplayBuffer(int size) : max_size(size) {}

    void add(const vector<double>& state, int action, double reward, const vector<double>& next_state, double priority) {
        buffer.push_back({state, action, reward, next_state, priority});
        priorities.push_back(priority);
        if (buffer.size() > max_size) {
            buffer.pop_front();
            priorities.erase(priorities.begin());
        }
    }

    vector<tuple<vector<double>, int, double, vector<double>, double>> sample(int batch_size) {
        vector<tuple<vector<double>, int, double, vector<double>, double>> sample_data;

        std::discrete_distribution<> distr(priorities.begin(), priorities.end());
        for (int i = 0; i < batch_size; ++i) {
            int idx = distr(gen);
            sample_data.push_back(buffer[idx]);
        }
        return sample_data;
    }


    bool is_ready(int batch_size) {
        return buffer.size() >= batch_size;
    }
};

// DQN Agent with Target Network
class DQNAgent {
public:
    DQNNetwork network, target_network;
    ReplayBuffer replay_buffer;
    double epsilon, epsilon_min, epsilon_decay, gamma, learning_rate;
    int update_counter, target_update_freq;

    DQNAgent(int input_size, int hidden_size, int output_size, int buffer_size, double lr, double gamma_val, int target_freq)
        : network(input_size, hidden_size, output_size),
          target_network(input_size, hidden_size, output_size),
          replay_buffer(buffer_size),
          epsilon(1.0), epsilon_min(0.1), epsilon_decay(0.995),
          gamma(gamma_val), learning_rate(lr), update_counter(0), target_update_freq(target_freq) {}

    int select_action(const vector<double>& state) {
        if (dis(gen) < epsilon) {
            return rand() % 2;
        }
        auto q_values = network.predict(state);
        return max_element(q_values.begin(), q_values.end()) - q_values.begin();
    }

    void train(int batch_size) {
        if (!replay_buffer.is_ready(batch_size)) return;

        auto batch = replay_buffer.sample(batch_size);
        for (auto& experience : batch) {
            vector<double> state = get<0>(experience);
            int action = get<1>(experience);
            double reward = get<2>(experience);
            vector<double> next_state = get<3>(experience);

            auto q_values = network.predict(state);
            auto next_q_values = target_network.predict(next_state);
            double max_next_q = *max_element(next_q_values.begin(), next_q_values.end());

            q_values[action] = reward + gamma * max_next_q;
            network.update_weights(state, q_values, learning_rate);
        }

        if (++update_counter % target_update_freq == 0) {
            target_network = network;
        }

        epsilon = max(epsilon_min, epsilon * epsilon_decay);
    }

};

class CartPole {
private:
    double x, x_dot, theta, theta_dot, dt = 0.02;
    double g = 9.8, m_c = 1.0, m_p = 0.1, l = 0.5;

public:
    CartPole() : x(0), x_dot(0), theta(0.1), theta_dot(0) {}

    void update(double force) {
        double sin_theta = sin(theta);
        double cos_theta = cos(theta);

        double temp = (force + m_p * l * theta_dot * theta_dot * sin_theta) / (m_c + m_p);
        double theta_acc = (g * sin_theta - cos_theta * temp) /
                           (l * (4.0 / 3.0 - m_p * cos_theta * cos_theta / (m_c + m_p)));
        double x_acc = temp - m_p * l * theta_acc * cos_theta / (m_c + m_p);

        x += x_dot * dt;
        x_dot += x_acc * dt;
        theta += theta_dot * dt;
        theta_dot += theta_acc * dt;
    }

    bool is_done() const {
        return fabs(x) > 5 || fabs(theta) > 0.3;
    }

    vector<double> get_state() const {
        return {x, x_dot, theta, theta_dot};
    }
};


class DQNSensorApp : public ApplicationBase, public UdpSocket::ICallback {
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

    // DQN Agent for decision making
    DQNAgent dqnAgent{5, 64, 2, 10000, 0.001, 0.99, 10};  // Set target update frequency (e.g., 10)
  // Configure with input/output sizes and hyperparameters

    // DQN state tracking
    std::vector<double> prevState;
    int prevAction;

public:
    DQNSensorApp() : avgRTT(0.1) {
        cartPoleUpdateInterval = 0.02;
    }
    virtual ~DQNSensorApp();

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
    void logData(simtime_t currentTime, double sensorRTT, double lqgCost, double reward, const std::vector<double>& state, int action, double tdError, const std::vector<double>& qValues);
    void initializeCSVFile();
    double estimateNetworkCongestion();
    double calculateReward(const std::vector<double>& state, int action);
};

Define_Module(DQNSensorApp);
DQNSensorApp::~DQNSensorApp() {
    cancelAndDelete(sendTimer);
    cancelAndDelete(timeoutTimer);
    cancelAndDelete(cartPoleUpdateTimer);
    if (csvFile.is_open()) {
        csvFile.close();
    }
}
void DQNSensorApp::initialize(int stage)
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
                // Optional: log or configure based on IPv4 data if needed
            }
        }

        EV_INFO << "Sensor " << getParentModule()->getFullName() << " sending to: " << destAddr << endl;

        if (!sendTimer->isScheduled()) {
            scheduleAt(simTime() + sendInterval, sendTimer);
        }

        // Initialize previous state and action for DQN
        prevState = cartPole.get_state();
        prevAction = 0;  // Set an initial default action
    }
}
void DQNSensorApp::handleMessageWhenUp(cMessage *msg)
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
                    cartPole.update(0);  // No force applied directly, control logic applied later
                    auto state = cartPole.get_state();
                    double congestion = estimateNetworkCongestion();
                    state.push_back(congestion);  // Add congestion to the state

                    int action = dqnAgent.select_action(state);  // Choose action from DQN

                    double reward = calculateReward(state, action);

                    if (!prevState.empty()) {
                        double tdError = fabs(reward + dqnAgent.gamma * dqnAgent.target_network.predict(state)[0]
                                              - dqnAgent.network.predict(prevState)[prevAction]);
                        dqnAgent.replay_buffer.add(prevState, prevAction, reward, state, tdError);
                        dqnAgent.train(32);
                    }

                    prevState = state;
                    prevAction = action;

            // Check if simulation should end
            if (cartPole.is_done()) {
                EV_WARN << "Cart-pole simulation ended during continuous update" << endl;

                // Reset the cart-pole for a new episode
                cartPole = CartPole();

                // Reinitialize state and action tracking for DQN
                prevState = cartPole.get_state();
                prevAction = 0;  // Reset action for new episode
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



void DQNSensorApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    processPacket(packet);
}

void DQNSensorApp::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Socket error arrived" << endl;
    delete indication;
}

void DQNSensorApp::socketClosed(UdpSocket *socket)
{
    EV_WARN << "Socket closed" << endl;
}

void DQNSensorApp::initializeCSVFile()
{
    std::time_t now = std::time(nullptr);
    char filename[100];
    std::strftime(filename, sizeof(filename), "sensor_log_%Y%m%d_%H%M%S", std::localtime(&now));
    csvFilename = filename + static_cast<std::string>(getParentModule()->getFullName()) + ".csv";
    csvFile.open(csvFilename);
    if (csvFile.is_open()) {
        csvFile << "Time,Sensor RTT,LQG Cost,Reward" << std::endl;
    } else {
        throw cRuntimeError("Unable to open CSV file for writing");
    }
}


void DQNSensorApp::sendUpdate() {
    auto state = cartPole.get_state();
    double congestion = estimateNetworkCongestion();

    int action = dqnAgent.select_action(state);
    bool forceSend = fabs(state[2]) > 0.2;  // Force send if pole angle deviates too much

    if (action == 1 || forceSend) {
        auto packet = new Packet("DQNSensorData");
        const auto& payload = makeShared<BytesChunk>();
        vector<uint8_t> data;

        data.push_back(static_cast<uint8_t>(sequenceNumber));
        for (double value : state) {
            uint16_t intValue = static_cast<uint16_t>((value + 10) * 1000);
            data.push_back(intValue >> 8);
            data.push_back(intValue & 0xFF);
        }

        lastSensorPacketSentTime = simTime();
        payload->setBytes(data);
        packet->insertAtBack(payload);
        socket.sendTo(packet, destAddr, par("destPort").intValue());

        waitingForAck = true;
        sequenceNumber++;
        cancelEvent(timeoutTimer);
        scheduleAt(simTime() + 0.1, timeoutTimer);
    }

    double reward = calculateReward(state, action);

    if (!prevState.empty()) {
        double tdError = fabs(reward + dqnAgent.gamma * dqnAgent.target_network.predict(state)[0]
                              - dqnAgent.network.predict(prevState)[prevAction]);
        dqnAgent.replay_buffer.add(prevState, prevAction, reward, state, tdError);
        dqnAgent.train(32);
    }
    auto qValues = dqnAgent.network.predict(state);
    double tdError = fabs(reward + dqnAgent.gamma * dqnAgent.target_network.predict(state)[0]
                          - qValues[action]);
    prevState = state;
    prevAction = action;
}


double DQNSensorApp::estimateNetworkCongestion() {
    // Use the exponential moving average of RTT to estimate congestion
    double normalizedRTT = avgRTT / 0.01;  // Normalize assuming 100ms is a "normal" RTT
    return std::min(1.0, std::max(0.0, normalizedRTT - 1.0));  // Clamp between 0 and 1
}
double smoothMin(double a, double b, double eta = 300) {
    return (a + b - sqrt((a - b) * (a - b) + eta * eta)) / 2.0;
}

double DQNSensorApp::calculateReward(const std::vector<double>& state, int action) {
    double x = state[0];
    double theta = state[2];
    double x_dot = state[1];
    double theta_dot = state[3];

    double safetyRobustness = smoothMin(3 - fabs(x), 0.3 - fabs(theta));
    double livenessRobustness = smoothMin(1 - fabs(x), 0.1 - fabs(theta));
    double velocityPenalty = 0.1 * theta_dot * theta_dot;

    double actionPenalty = (action == 1 ? 1.0 : 0.0);
    double lqg=state[0]*state[0]+100*state[2]*state[2];
    double reward = -lqg;

    return reward;
}



void DQNSensorApp::processPacket(Packet *pk) {
    auto chunk = pk->peekAtFront<BytesChunk>();
    auto bytes = chunk->getBytes();
    uint8_t receivedSeq = bytes[0];

    if (receivedSeq == (sequenceNumber - 1)) {
        simtime_t currentTime = simTime();
        lastRTT = currentTime - lastSensorPacketSentTime;

        uint16_t forceInt = (bytes[1] << 8) | bytes[2];
        double force = (static_cast<double>(forceInt) / 1000.0) - 10;
        cartPole.update(force);

        auto newState = cartPole.get_state();
        double congestion = estimateNetworkCongestion();
        newState.push_back(congestion);

        double reward = calculateReward(newState, prevAction);

        if (!prevState.empty()) {
            double tdError = fabs(reward + dqnAgent.gamma * dqnAgent.target_network.predict(newState)[0]
                                  - dqnAgent.network.predict(prevState)[prevAction]);
            dqnAgent.replay_buffer.add(prevState, prevAction, reward, newState, tdError);
            dqnAgent.train(32);
        }
        double lqgCost = newState[0] * newState[0] + 100 * newState[2] * newState[2];
        auto qValues = dqnAgent.network.predict(newState);
        double tdError = fabs(reward + dqnAgent.gamma * dqnAgent.target_network.predict(newState)[0]
                              - qValues[prevAction]);
        logData(currentTime, lastRTT.dbl(),lqgCost, reward, newState, prevAction, tdError, qValues);
        if (cartPole.is_done()) {
            cartPole = CartPole();
            prevState = cartPole.get_state();
            prevState.push_back(estimateNetworkCongestion());
            prevAction = 0;
        } else {
            prevState = newState;
        }
    }

    delete pk;
}


void DQNSensorApp::handleStartOperation(LifecycleOperation *operation)
{
    scheduleAt(simTime() + sendInterval, sendTimer);
    scheduleAt(simTime() + cartPoleUpdateInterval, cartPoleUpdateTimer);
}

void DQNSensorApp::handleStopOperation(LifecycleOperation *operation)
{
    cancelEvent(sendTimer);
    cancelEvent(timeoutTimer);
    cancelEvent(cartPoleUpdateTimer);
    socket.close();
}

void DQNSensorApp::handleCrashOperation(LifecycleOperation *operation)
{
    cancelEvent(sendTimer);
    cancelEvent(timeoutTimer);
    cancelEvent(cartPoleUpdateTimer);
    if (socket.isOpen()) {
        socket.destroy();
    }
}

void DQNSensorApp::handleTimeout()
{
    EV_INFO << "Timeout occurred" << endl;
    waitingForAck = false;
    // Wait for the next scheduled send without immediate retransmission
}

void DQNSensorApp::logData(simtime_t currentTime, double sensorRTT, double lqgCost, double reward,
                           const std::vector<double>& state, int action, double tdError, const std::vector<double>& qValues) {
    double safetyRobustness = smoothMin(3 - fabs(state[0]), 0.3 - fabs(state[2]));
    double livenessRobustness = smoothMin(1 - fabs(state[0]), 0.1 - fabs(state[2]));

    if (csvFile.is_open()) {
        csvFile << currentTime.dbl() << "," << sensorRTT << "," << lqgCost << "," << reward << ","
                << safetyRobustness << "," << livenessRobustness << "," << tdError << "," << qValues[action];

        for (const auto& qValue : qValues) {
            csvFile << "," << qValue;
        }
        csvFile << std::endl;

        EV_INFO << "Logged Data: Time=" << currentTime.dbl()
                << ", RTT=" << sensorRTT
                << ", LQG=" << lqgCost
                << ", Reward=" << reward
                << ", Safety=" << safetyRobustness
                << ", Liveness=" << livenessRobustness
                << ", TD Error=" << tdError
                << ", Selected Q-Value=" << qValues[action] << std::endl;
    } else {
        EV_ERROR << "CSV file is not open for writing" << std::endl;
    }
}


void DQNSensorApp::finish()
{
    ApplicationBase::finish();
    if (csvFile.is_open()) {
        csvFile.close();
    }
    // Additional cleanup for DQN agent resources, if needed
}

