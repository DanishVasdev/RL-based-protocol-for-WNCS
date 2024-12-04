//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

package zero_wait.simulations;

import inet.applications.contract.IApp;

simple DQNControllerApp like IApp
{
    parameters:
        @networkNode;
        string interfaceTableModule;
        string localAddress = default(""); // Local IP address (empty string means auto-assign)
        int localPort = default(-1); // Local port number (-1: use ephemeral port)
        int destPort; // Destination port
        string destAddresses = default(""); // List of IP addresses, separated by spaces
        double sendInterval @unit(s) = default(1s);
        double startTime @unit(s) = default(0s); // Delay before starting to send messages
        bool useAutoConfig = default(true); // Whether to use auto-configuration for IP address
        @display("i=block/app");
    gates:
        input socketIn;
        output socketOut;
}
simple DQNSensorApp like IApp
{
    parameters:
        @networkNode;
        string interfaceTableModule;
        string localAddress = default(""); // Local IP address (empty string means auto-assign)
        int localPort = default(-1); // Local port number (-1: use ephemeral port)
        int destPort; // Destination port
        string destAddresses = default(""); // List of IP addresses, separated by spaces
        double sendInterval @unit(s) = default(0.1s);
        double startTime @unit(s) = default(0s); // Delay before starting to send messages
        bool useAutoConfig = default(true); // Whether to use auto-configuration for IP address
        @display("i=block/app");
    gates:
        input socketIn;
        output socketOut;
}