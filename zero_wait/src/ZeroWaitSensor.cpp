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

class ZeroWaitSensorApp : public ApplicationBase, public UdpSocket::ICallback
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

    // RTT tracking
    simtime_t lastSensorPacketSentTime;

    // Logging related variables
    std::ofstream csvFile;
    std::string csvFilename;

  public:
    ZeroWaitSensorApp() {}
    virtual ~ZeroWaitSensorApp();

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

Define_Module(ZeroWaitSensorApp);

ZeroWaitSensorApp::~ZeroWaitSensorApp()
{
    cancelAndDelete(sendTimer);
    cancelAndDelete(timeoutTimer);
    if (csvFile.is_open()) {
        csvFile.close();
    }
}

void ZeroWaitSensorApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        localPort = par("localPort");
        destPort = par("destPort");
        sendInterval = par("sendInterval");

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
                EV_INFO << "Sensor " << getParentModule()->getFullName() << " assigned IP: " << ipv4Data->getIPAddress() << endl;
            }
        }

        EV_INFO << "Sensor " << getParentModule()->getFullName() << " sending to: " << destAddr << endl;

        if (!sendTimer->isScheduled()) {
            scheduleAt(simTime() + sendInterval, sendTimer);
        }
    }
}

void ZeroWaitSensorApp::initializeCSVFile()
{
    std::time_t now = std::time(nullptr);
    char filename[100];
    std::strftime(filename, sizeof(filename), "sensor_log_%Y%m%d_%H%M%S", std::localtime(&now));
    csvFilename = filename+static_cast<std::string>(getParentModule()->getFullName())+".csv";
    csvFile.open(csvFilename);
    if (csvFile.is_open()) {
        csvFile << "Time,Sensor RTT,LQG Cost" << std::endl;
    } else {
        throw cRuntimeError("Unable to open CSV file for writing");
    }
}

void ZeroWaitSensorApp::handleMessageWhenUp(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        EV_WARN<<"Sensor: "<<getParentModule()->getFullName()<<"message"<<msg<<endl;
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

void ZeroWaitSensorApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    processPacket(packet);
}

void ZeroWaitSensorApp::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Socket error arrived" << endl;
    delete indication;
}

void ZeroWaitSensorApp::socketClosed(UdpSocket *socket)
{
    EV_WARN << "Socket closed" << endl;
}

void ZeroWaitSensorApp::sendUpdate()
{
    auto packet = new Packet("ZeroWaitSensorData");
    const auto& payload = makeShared<BytesChunk>();
    std::vector<uint8_t> data;

    data.push_back(static_cast<uint8_t>(sequenceNumber));

    // Sensor sends state
    auto state = cartPole.get_state();
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
    simtime_t currentTime = simTime();
    logData(currentTime, -1, 0.0, -1);
    cancelEvent(timeoutTimer);
    scheduleAt(simTime() + 0.1, timeoutTimer);  // 3 second timeout
}

void ZeroWaitSensorApp::processPacket(Packet *pk)
{
    auto chunk = pk->peekAtFront<BytesChunk>();
    auto bytes = chunk->getBytes();
    uint8_t receivedSeq = bytes[0];
    EV_WARN << "Processing Packet at Sensor" << endl;
    EV_INFO << "Received seq " << (int)receivedSeq << " expected " << sequenceNumber-1 << endl;
    if (receivedSeq == (sequenceNumber - 1)) {
        simtime_t currentTime = simTime();
        double sensorRTT = (currentTime - lastSensorPacketSentTime).dbl();
        EV_INFO << "Sensor: ACK received for sequence number " << (int)receivedSeq << ", RTT: " << sensorRTT << endl;

        waitingForAck = false;
        cancelEvent(timeoutTimer);

        // Sensor receives force
        uint16_t forceInt = (bytes[1] << 8) | bytes[2];
        double force = (static_cast<double>(forceInt) / 1000.0) - 10;
        cartPole.step(force);
        auto newState = cartPole.get_state();
        EV_INFO << "Applied force: " << force << ", New state: x=" << newState[0] << ", theta=" << newState[2] << endl;

        // Calculate LQG cost (simplified version)
        double lqgCost = newState[0]*newState[0] + 100*newState[2]*newState[2];  // Example cost function

        if (cartPole.is_done()) {
            EV_WARN << "Cart-pole simulation ended" << endl;
            // You might want to add some logic here to stop the simulation or reset the cart-pole
            finish();
        }
        EV_WARN<<static_cast<std::string>(getParentModule()->getFullName())<<endl;
        logData(currentTime, sensorRTT, 0.0, lqgCost);
    }

    delete pk;
}

void ZeroWaitSensorApp::handleStartOperation(LifecycleOperation *operation)
{
    EV_WARN<<"Sensor: "<<getParentModule()->getFullName()<<"start operation"<<endl;
    scheduleAt(simTime() + sendInterval, sendTimer);
}

void ZeroWaitSensorApp::handleStopOperation(LifecycleOperation *operation)
{
    EV_WARN<<"Sensor: "<<getParentModule()->getFullName()<<"stop operation"<<endl;
    cancelEvent(sendTimer);
    cancelEvent(timeoutTimer);
    socket.close();
}

void ZeroWaitSensorApp::handleCrashOperation(LifecycleOperation *operation)
{
    EV_WARN<<"Sensor: "<<getParentModule()->getFullName()<<"crash operation"<<endl;
    cancelEvent(sendTimer);
    cancelEvent(timeoutTimer);
    if (socket.isOpen())
        socket.destroy();
}

void ZeroWaitSensorApp::handleTimeout()
{
    EV_WARN<<"Sensor: "<<getParentModule()->getFullName()<<"timeout operation"<<endl;
    EV_INFO << "Timeout occurred" << endl;
    waitingForAck = false;
    sendUpdate();
}

void ZeroWaitSensorApp::logData(simtime_t currentTime, double sensorRTT, double controllerRTT, double lqgCost)
{
    if (csvFile.is_open()) {
        csvFile << currentTime.dbl() << "," << sensorRTT << "," << lqgCost << std::endl;
        EV_WARN << "Logging data " << sensorRTT << " " << lqgCost << std::endl;
    } else {
        EV_ERROR << "CSV file is not open for writing" << endl;
    }
}

void ZeroWaitSensorApp::finish()
{
    EV_WARN<<"Sensor: "<<getParentModule()->getFullName()<<"finished"<<endl;
    ApplicationBase::finish();
    if (csvFile.is_open()) {
        csvFile.close();
    }
    //endSimulation();
}
