#include <omnetpp.h>
#include "inet/applications/base/ApplicationBase.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "inet/common/packet/Packet.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/networklayer/ipv4/Ipv4InterfaceData.h"
#include <fstream>
#include <ctime>
#include <cmath>

using namespace omnetpp;
using namespace inet;

class CartPole {
private:
    double x;        // Cart position
    double x_dot;    // Cart velocity
    double theta;    // Pole angle
    double theta_dot;// Pole angular velocity

    const double g = 9.8;   // Gravity
    const double m_c = 1.0; // Mass of the cart
    const double m_p = 0.1; // Mass of the pole
    const double l = 0.5;   // Half-length of the pole
    const double dt = 0.02; // Time step

public:
    CartPole() : x(0), x_dot(0), theta(0.1), theta_dot(0) {}

    void step(double force) {
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);

        double temp = (force + m_p * l * theta_dot * theta_dot * sin_theta) / (m_c + m_p);
        double theta_acc = (g * sin_theta - cos_theta * temp) / (l * (4.0/3.0 - m_p * cos_theta * cos_theta / (m_c + m_p)));
        double x_acc = temp - m_p * l * theta_acc * cos_theta / (m_c + m_p);

        x += x_dot * dt;
        x_dot += x_acc * dt;
        theta += theta_dot * dt;
        theta_dot += theta_acc * dt;
    }

    bool is_done() const {
        return std::abs(x) > 2.4 || std::abs(theta) > 0.2095;
    }

    std::vector<double> get_state() const {
        return {x, x_dot, theta, theta_dot};
    }
};

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

class ZeroWaitCartPoleApp : public ApplicationBase, public UdpSocket::ICallback
{
  protected:
    UdpSocket socket;
    cMessage *sendTimer = nullptr;
    cMessage *timeoutTimer = nullptr;
    int sequenceNumber = 0;
    bool waitingForAck = false;

    // Configuration
    int localPort, destPort;
    L3Address destAddr;
    simtime_t sendInterval;

    // Cart-pole specific
    CartPole cartPole;
    PIDController pidController;
    bool isController;

    // RTT tracking
    simtime_t lastSensorPacketSentTime;
    simtime_t lastControllerPacketSentTime;

    // Logging related variables
    std::ofstream csvFile;
    std::string csvFilename;

  public:
    ZeroWaitCartPoleApp() : pidController(10, 0.1, 5) {}
    virtual ~ZeroWaitCartPoleApp();

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
    void logData(simtime_t currentTime, double sensorRTT, double controllerRTT, double lqgCost);
    void initializeCSVFile();
};

Define_Module(ZeroWaitCartPoleApp);

ZeroWaitCartPoleApp::~ZeroWaitCartPoleApp()
{
    cancelAndDelete(sendTimer);
    cancelAndDelete(timeoutTimer);
    if (csvFile.is_open()) {
        csvFile.close();
    }
}

void ZeroWaitCartPoleApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        localPort = par("localPort");
        destPort = par("destPort");
        sendInterval = par("sendInterval");
        isController = par("isController");

        sendTimer = new cMessage("sendTimer");
        timeoutTimer = new cMessage("timeoutTimer");

        initializeCSVFile();
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        socket.setOutputGate(gate("socketOut"));
        socket.bind(localPort);

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
                EV_INFO << "Host " << getParentModule()->getFullName() << " assigned IP: " << ipv4Data->getIPAddress() << endl;
            }
        }

        EV_INFO << "Host " << getParentModule()->getFullName() << " sending to: " << destAddr << endl;

        if (!sendTimer->isScheduled()) {
            scheduleAt(simTime() + sendInterval, sendTimer);
        }
    }
}

void ZeroWaitCartPoleApp::initializeCSVFile()
{
    std::time_t now = std::time(nullptr);
    char filename[100];
    std::strftime(filename, sizeof(filename), "simulation_log_%Y%m%d_%H%M%S.csv", std::localtime(&now));
    csvFilename = filename;
    csvFile.open(csvFilename);
    if (csvFile.is_open()) {
        csvFile << "Time,Sensor RTT,Controller RTT,LQG Cost" << std::endl;
    } else {
        throw cRuntimeError("Unable to open CSV file for writing");
    }
}

void ZeroWaitCartPoleApp::handleMessageWhenUp(cMessage *msg)
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
    }
    else {
        socket.processMessage(msg);
    }
}

void ZeroWaitCartPoleApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    processPacket(packet);
}

void ZeroWaitCartPoleApp::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Socket error arrived" << endl;
    delete indication;
}

void ZeroWaitCartPoleApp::socketClosed(UdpSocket *socket)
{
    EV_WARN << "Socket closed" << endl;
}

void ZeroWaitCartPoleApp::sendUpdate()
{
    auto packet = new Packet("ZeroWaitCartPoleData");
    const auto& payload = makeShared<BytesChunk>();
    std::vector<uint8_t> data;

    data.push_back(static_cast<uint8_t>(sequenceNumber));

    if (isController) {
        // Controller sends force
        double force = pidController.calculate(0, cartPole.get_state()[2], 0.02);
        force = std::max(-10.0, std::min(10.0, force));  // Limit force to [-10, 10]
        uint16_t forceInt = static_cast<uint16_t>((force + 10) * 1000);  // Convert to fixed-point
        data.push_back(forceInt >> 8);
        data.push_back(forceInt & 0xFF);
        lastControllerPacketSentTime = simTime();
        EV_WARN<<"Controller update sent to "<<destAddr<<"with seq no "<<sequenceNumber<<endl;
    } else {
        // Sensor sends state
        auto state = cartPole.get_state();
        for (double value : state) {
            uint16_t intValue = static_cast<uint16_t>((value + 10) * 1000);  // Convert to fixed-point
            data.push_back(intValue >> 8);
            data.push_back(intValue & 0xFF);
        }
        lastSensorPacketSentTime = simTime();
        EV_WARN<<"Sensor update sent"<<destAddr<<"with seq no "<<sequenceNumber<<endl;
    }

    payload->setBytes(data);
    packet->insertAtBack(payload);
    EV_INFO << "Sending packet to " << destAddr << endl;
    socket.sendTo(packet, destAddr, destPort);
    waitingForAck = true;
    sequenceNumber++;

    cancelEvent(timeoutTimer);
    scheduleAt(simTime() + 3.0, timeoutTimer);  // 1 second timeout
}

void ZeroWaitCartPoleApp::processPacket(Packet *pk)
{
    auto chunk = pk->peekAtFront<BytesChunk>();
    auto bytes = chunk->getBytes();
    uint8_t receivedSeq = bytes[0];
	EV_WARN<<"Processing Packer at Controller: "<<isController<<endl;
	EV_INFO<<"Recieved seq "<<receivedSeq<<" expected "<<sequenceNumber-1<<endl;
    if (receivedSeq == (sequenceNumber - 1)) {
        simtime_t currentTime = simTime();
        double sensorRTT = 0.0;
        double controllerRTT = 0.0;
        
        if (isController) {
            controllerRTT = (currentTime - lastControllerPacketSentTime).dbl();
            EV_INFO << "Controller: ACK received for sequence number " << (int)receivedSeq << ", RTT: " << controllerRTT << endl;
        } else {
            sensorRTT = (currentTime - lastSensorPacketSentTime).dbl();
            EV_INFO << "Sensor: ACK received for sequence number " << (int)receivedSeq << ", RTT: " << sensorRTT << endl;
        }

        waitingForAck = false;
        cancelEvent(timeoutTimer);

        double lqgCost = 0.0;

        if (isController) {
            // Controller receives state
            std::vector<double> state;
            for (size_t i = 1; i < bytes.size(); i += 2) {
                uint16_t intValue = (bytes[i] << 8) | bytes[i+1];
                double value = (static_cast<double>(intValue) / 1000.0) - 10;
                state.push_back(value);
            }
            EV_INFO << "Received state: x=" << state[0] << ", theta=" << state[2] << endl;

            // Calculate LQG cost (simplified version)
            lqgCost = state[0]*state[0] + 100*state[2]*state[2];  // Example cost function
        } else {
            // Sensor receives force
            uint16_t forceInt = (bytes[1] << 8) | bytes[2];
            double force = (static_cast<double>(forceInt) / 1000.0) - 10;
            cartPole.step(force);
            auto newState = cartPole.get_state();
            EV_INFO << "Applied force: " << force << ", New state: x=" << newState[0] << ", theta=" << newState[2] << endl;

            // Calculate LQG cost (simplified version)
            lqgCost = newState[0]*newState[0] + 100*newState[2]*newState[2];  // Example cost function

            if (cartPole.is_done()) {
                EV_WARN << "Cart-pole simulation ended" << endl;
                // You might want to add some logic here to stop the simulation or reset the cart-pole
            }
        }

        logData(currentTime, sensorRTT, controllerRTT, lqgCost);
    }

    delete pk;
}

void ZeroWaitCartPoleApp::handleStartOperation(LifecycleOperation *operation)
{
    scheduleAt(simTime() + sendInterval, sendTimer);
}

void ZeroWaitCartPoleApp::handleStopOperation(LifecycleOperation *operation)
{
    cancelEvent(sendTimer);
    cancelEvent(timeoutTimer);
    socket.close();
}

void ZeroWaitCartPoleApp::handleCrashOperation(LifecycleOperation *operation)
{
    cancelEvent(sendTimer);
    cancelEvent(timeoutTimer);
    if (socket.isOpen())
        socket.destroy();
}

void ZeroWaitCartPoleApp::handleTimeout()
{
    EV_INFO << "Timeout occurred" << endl;
    //sequenceNumber--;  // Resend the last sequence number
    waitingForAck = false;
    sendUpdate();
}

void ZeroWaitCartPoleApp::logData(simtime_t currentTime, double sensorRTT, double controllerRTT, double lqgCost)
{
    if (csvFile.is_open()) {
        csvFile << currentTime.dbl() << "," << sensorRTT << "," << controllerRTT << "," << lqgCost << std::endl;
        EV_WARN<<"Logging data"<<sensorRTT<<controllerRTT<<lqgCost<<std::endl;
    } else {
        EV_ERROR << "CSV file is not open for writing" << endl;
    }
}

void ZeroWaitCartPoleApp::finish()
{
    ApplicationBase::finish();
    if (csvFile.is_open()) {
        csvFile.close();
    }
}

