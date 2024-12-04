#include <omnetpp.h>
#include "inet/applications/base/ApplicationBase.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "inet/common/packet/Packet.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/networklayer/ipv4/Ipv4InterfaceData.h" // Add this include

using namespace omnetpp;
using namespace inet;

class ZeroWaitApp : public ApplicationBase, public UdpSocket::ICallback
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

  public:
    ZeroWaitApp() {}
    virtual ~ZeroWaitApp();

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
};

ZeroWaitApp::~ZeroWaitApp()
{
    cancelAndDelete(sendTimer);
    cancelAndDelete(timeoutTimer);
}



void ZeroWaitApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        localPort = par("localPort");
        destPort = par("destPort");
        sendInterval = par("sendInterval");

        sendTimer = new cMessage("sendTimer");
        timeoutTimer = new cMessage("timeoutTimer");
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
        
        // Log the host's own IP address
        auto ift = getModuleFromPar<IInterfaceTable>(par("interfaceTableModule"), this);
        auto ie = ift->findInterfaceByName("wlan0");
        if (ie) {
            auto ipv4Data = ie->findProtocolData<Ipv4InterfaceData>();
            if (ipv4Data) {
                EV_INFO << "Host " << getParentModule()->getFullName() << " assigned IP: " << ipv4Data->getIPAddress() << endl;
            }
        }

        // Log the destination address
        EV_INFO << "Host " << getParentModule()->getFullName() << " sending to: " << destAddr << endl;

        // Only schedule if not already scheduled
        if (!sendTimer->isScheduled()) {
            scheduleAt(simTime() + sendInterval, sendTimer);
        }
    }
}


void ZeroWaitApp::handleMessageWhenUp(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        if (msg == sendTimer) {
            if (!waitingForAck) {
                sendUpdate();
            }
            // Cancel any existing sendTimer before scheduling a new one
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

void ZeroWaitApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    processPacket(packet);
}

void ZeroWaitApp::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Socket error arrived" << endl;
    delete indication;
}

void ZeroWaitApp::socketClosed(UdpSocket *socket)
{
    // Handle socket closed
    EV_WARN<<"Socket closed"<<endl;
}

void ZeroWaitApp::sendUpdate()
{
    auto packet = new Packet("ZeroWaitData");
    const auto& payload = makeShared<BytesChunk>();
    payload->setBytes({static_cast<unsigned char>(sequenceNumber)});
    packet->insertAtBack(payload);
	EV_WARN << "Sending packet to " << destAddr << endl;
    socket.sendTo(packet, destAddr, destPort);
    waitingForAck = true;
    sequenceNumber++;

    // Start timeout timer
    cancelEvent(timeoutTimer);
    scheduleAt(simTime() + 1.0, timeoutTimer);  // 1 second timeout
}

void ZeroWaitApp::processPacket(Packet *pk)
{
    // Check if it's an ACK
    const auto& chunk = pk->peekAtFront<BytesChunk>();
    uint8_t receivedSeq = chunk->getBytes().at(0);

    if (receivedSeq == (sequenceNumber - 1)) {
        EV_INFO << "ACK received for sequence number " << (int)receivedSeq << endl;
        waitingForAck = false;
        cancelEvent(timeoutTimer);
    }

    delete pk;
}
void ZeroWaitApp::handleStartOperation(LifecycleOperation *operation)
{
    scheduleAt(simTime() + sendInterval, sendTimer);
}

void ZeroWaitApp::handleStopOperation(LifecycleOperation *operation)
{
    cancelEvent(sendTimer);
    cancelEvent(timeoutTimer);
    socket.close();
}

void ZeroWaitApp::handleCrashOperation(LifecycleOperation *operation)
{
    cancelEvent(sendTimer);
    cancelEvent(timeoutTimer);
    if (socket.isOpen())
        socket.destroy();
}
void ZeroWaitApp::handleTimeout()
{
    EV_INFO << "Timeout occurred, resending last update" << endl;
    sequenceNumber--;  // Resend the last sequence number
    waitingForAck = false;
    sendUpdate();
}

void ZeroWaitApp::finish()
{
    ApplicationBase::finish();
}
Define_Module(ZeroWaitApp);