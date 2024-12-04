package zero_wait.simulations;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.WirelessHost;
import inet.physicallayer.wireless.common.contract.packetlevel.IRadioMedium;

network ZeroWaitNetwork
{
    parameters:
        int numSensors = default(7);  // 50 sensor nodes
        double areaWidth = default(500);
        double areaHeight = default(500);
        @display("bgb=1000,1000");
    submodules:
        configurator: Ipv4NetworkConfigurator {
            @display("p=100,100");
        }
        radioMedium: <default("UnitDiskRadioMedium")> like IRadioMedium {
            @display("p=100,200");
        }

        // Control node
        control: WirelessHost {
            @display("p=554.208,475.896"); // Centralized position for controller
        }

        // Sensor nodes dynamically from sensor0 to sensor49
        //sensor[numSensors]: WirelessHost {
        //    @display("p=600,600");
        //}
        sensor0: WirelessHost {
            @display("p=708.824,389.552");
        }
        sensor1: WirelessHost {
            @display("p=389.552,594.368");
        }
        sensor2: WirelessHost {
            @display("p=365.456,455.81598");
        }
        sensor3: WirelessHost {
            @display("p=608.424,311.24");
        }
        sensor4: WirelessHost {
            @display("p=443.76797,323.288");
        }
        sensor5: WirelessHost {
            @display("p=708.824,542.16");
        }
        sensor6: WirelessHost {
            @display("p=554.208,628.50397");
        }

    connections allowunconnected:
        // Wireless connections handled by the radio medium
}
