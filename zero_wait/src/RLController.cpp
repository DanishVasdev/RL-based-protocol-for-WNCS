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

using namespace omnetpp;
using namespace inet;

class PIDController {
private:
    double kp, ki, kd;
    double integral, prev_error;

public:
    PIDController(double p, double i, double d) : kp(p), ki(i), kd(d), integral(0), prev_error(0) {}

    double calculate(double setpoint, double measured_value, double dt) {
        double error = setpoint - measured_value;
        integral += error * dt;
        double derivative = (error - prev_error) / dt;
        prev_error = error;

        return kp * error + ki * integral + kd * derivative;
    }
};

class RLControllerApp : public ApplicationBase, public UdpSocket::ICallback
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

    // PID Controllers
    PIDController *pidCart;
    PIDController *pidPole;
    std::vector<double> currentState;

    // RTT tracking
    std::vector<simtime_t> lastControllerPacketSentTimes;

    // Logging related variables
    std::ofstream csvFile;
    std::string csvFilename;
    std::vector<simtime_t> lastUpdateTimes;

public:
    RLControllerApp() {
        timeoutTimer = nullptr;
        pidCart = new PIDController(192.189, 222.27, 137.44);
        pidPole = new PIDController(605.48, 219.14, 10.0);
        currentState.resize(4, 0.0);  // Initialize state vector with zeros
    }

    virtual ~RLControllerApp();

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
};

Define_Module(RLControllerApp);

RLControllerApp::~RLControllerApp()
{
    cancelAndDelete(timeoutTimer);
    if (csvFile.is_open()) {
        csvFile.close();
    }
    delete pidCart;
    delete pidPole;
}

void RLControllerApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        localPort = par("localPort");
        destPort = par("destPort");
        sendInterval = par("sendInterval");

        timeoutTimer = new cMessage("timeoutTimer");

        initializeCSVFile();
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
        lastUpdateTimes.resize(destAddrs.size(), simTime());

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

void RLControllerApp::initializeCSVFile()
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

void RLControllerApp::handleMessageWhenUp(cMessage *msg)
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

void RLControllerApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    processPacket(packet);
}

void RLControllerApp::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Socket error arrived" << endl;
    delete indication;
}

void RLControllerApp::socketClosed(UdpSocket *socket)
{
    EV_WARN << "Socket closed" << endl;
}

void RLControllerApp::processPacket(Packet *pk)
{
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

    // Update current state
    currentState = measuredState;

    EV_INFO << "Received state from destination " << destIndex << ": "
            << "x=" << currentState[0] << ", x_dot=" << currentState[1]
            << ", theta=" << currentState[2] << ", theta_dot=" << currentState[3] << endl;

    // Calculate LQG cost using the current state
    double lqgCost = currentState[0]*currentState[0] + 100*currentState[2]*currentState[2];

    logData(currentTime, destIndex, controllerRTT, lqgCost);

    // Immediately send a new control update to this destination
    sendUpdate(destIndex);

    delete pk;
}

void RLControllerApp::sendUpdate(int destIndex)
{
    auto packet = new Packet("RLControllerData");
    const auto& payload = makeShared<BytesChunk>();
    std::vector<uint8_t> data;

    data.push_back(static_cast<uint8_t>(sequenceNumbers[destIndex]));

    double control_cart = pidCart->calculate(0.0, currentState[0], 0.02);
    double control_pole = pidPole->calculate(0.0, currentState[2], 0.02);

    // Controller sends force
    double force = -10 * (control_pole - control_cart);
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

void RLControllerApp::handleTimeout()
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

void RLControllerApp::logData(simtime_t currentTime, int destIndex, double controllerRTT, double lqgCost)
{
    if (csvFile.is_open()) {
        csvFile << currentTime.dbl() << "," << destIndex << "," << controllerRTT << "," << lqgCost << std::endl;
        EV_WARN << "Logging data for destination " << destIndex << ": " << controllerRTT << " " << lqgCost << std::endl;
    } else {
        EV_ERROR << "CSV file is not open for writing" << endl;
    }
}

void RLControllerApp::handleStartOperation(LifecycleOperation *operation)
{
    // Controller doesn't initiate communication, it waits for sensor data
}

void RLControllerApp::handleStopOperation(LifecycleOperation *operation)
{
    cancelEvent(timeoutTimer);
    socket.close();
}

void RLControllerApp::handleCrashOperation(LifecycleOperation *operation)
{
    cancelEvent(timeoutTimer);
    if (socket.isOpen())
        socket.destroy();
}

void RLControllerApp::finish()
{
    ApplicationBase::finish();
    if (csvFile.is_open()) {
        csvFile.close();
    }
}
