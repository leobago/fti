                         Mellanox Technologies
                         =====================

===============================================================================
                   Ethernet over IB (EoIB) for Linux README
                                  January 2013
                               Document No. 3289
===============================================================================

Contents:
=========
1. Overview
   1.1 General
   1.2 EoIB Topology
       1.2.1 External ports (eports) and GW
       1.2.2 Virtual Hubs (vHubs)
       1.2.3 Virtual NIC (vNic)
2. EoIB vNic Configuration
   2.1 EoIB Host Administered vNic
       2.1.1 Central Configuration File - mlx4_vnic.conf
       2.1.2 vNic Specific Configuration Files - ifcfg-eth<x>
       2.1.3 mlx4_vnic_confd
   2.2 EoIB Network Administered vNic
   2.3 VLAN Configuration
   2.4 EoIB Multicast Configuration
   2.5 EoIB and QoS
   2.6 IP Configuration Based on DHCP
       2.6.1 DHCP Server
   2.7 Static EoIB Configuration
   2.8 Sub Interfaces (VLAN)
3. EoIB Usage and Configuration
   3.1 mlx4_vnic_info
   3.2 ethtool
   3.3 Link State
   3.4 Offloads and Feature
   3.5 Jumbo Frames
   3.6 Discovery Partitions Configuration
   3.7 ALL VLAN
4. Advanced EoIB settings:
   4.1 Module Parameters
   4.2 Bonding
   4.3 vNic Interface Naming
   4.4 Para-Virtualized vNic
   4.5 EoIB Subnet Agent Query

1 Overview
==========

1.1 General
-----------
The Ethernet over IB (EoIB) mlx4_vnic module is a network interface
implementation over InfiniBand. EoIB encapsulates Layer 2 datagrams over an
InfiniBand Datagram (UD) transport service. The InfiniBand UD datagrams
encapsulates the entire Ethernet L2 datagram and its payload.

To perform this operation the module performs an address translation
from Ethernet layer 2 MAC addresses (48 bits long) to InfiniBand layer 2
addresses made of LID/GID and QPN. This translation is totally invisible
to the OS and user. Thus, differentiating EoIB
from IPoIB which exposes a 20 Bytes HW address to the OS.

The mlx4_vnic module is designed for Mellanox's ConnectX family of HCAs and
intended to be used with Mellanox's BridgeX gateway family. Having a BridgeX
gateway is a requirement for using EoIB. It performs the following operations:
 * Enables the layer 2 address translation required by the mlx4_vnic module.
 * Enables routing of packets from the InfiniBand fabric to a 1 or 10 GigE
   Ethernet subnet.

1.2 EoIB Topology
-----------------
EoIB is designed to work over an InfiniBand fabric and requires the presence
of two entities:
 * Subnet Manager (SM)
 * BridgeX gateway

The required subnet manager configuration is similar to that of other
InfiniBand applications and ULPs and is not unique to EoIB.
The BridgeX gateway is at the heart of EoIB. On one side, usually referred to
as the "internal" side, it is connected to the InfiniBand fabric by one or more
links. On the other side, usually referred to as the "external" side, it is
connected to the Ethernet subnet by one or more ports. The Ethernet connections
on the BridgeX's external side are called external ports or eports. Every
BridgeX that is in use with EoIB needs to have one or more eports connected.

1.2.1 External Ports (eports) and GW
---------------------------------------
The combination of a specific BridgeX box and a specific eport is referred to
as a gateway (GW). The GW is an entity that is visible to the EoIB host driver
and is used in the configuration of the network interfaces on the host side.
For example, in host administered vNics the user will request to open an
interface on a specific GW identifying it by the BridgeX box and eport name.
Distinguishing between GWs is important because they determine the network
topology and affect the path that a packet traverses between hosts. A packet
that is sent from the host on a specific EoIB interface will be routed to the
Ethernet subnet through a specific external port connection on the BridgeX box.

1.2.2 Virtual Hubs (vHubs)
---------------------------------------
Virtual hubs connect zero or more EoIB interfaces (on internal hosts) and an
eport through a virtual hub. Each vHub has a unique virtual LAN (VLAN) ID.
Virtual hub participants can send packets to one another directly without the
assistance of the Ethernet subnet (external side) routing. This means that two
EoIB interfaces on the same vHub will communicate solely using the InfiniBand
fabric. EoIB interfaces residing on two different vHubs (whether on the same
GW or not) cannot communicate directly.
There are two types of vHubs:
- a default vHub (one per GW) without a VLAN ID
- vHubs with unique different VLAN IDs
Each vHub belongs to a specific GW (BridgeX + eport),
and each GW has one default vHub, and zero or more
VLAN-associated vHubs. A specific GW can have multiple vHubs distinguishable
by their unique VLAN ID. Traffic coming from the Ethernet side
on a specific eport will be routed to the relevant vHub group based on its
VLAN tag (or to the default vHub for that GW if no VLAN ID is present).

1.2.3 Virtual NIC (vNic)
--------------------------------------
A virtual NIC is a network interface instance on the host side which belongs
to a single vHub on a specific GW. The vNic behaves like
any regular hardware network interface.
The host can have multiple interfaces that belong to the same vHub.

2 EoIB vNic Configuration
=========================
The mlx4_vnic module supports two different modes of configuration:
- host administration where the vNic is configured on the host side
- network administration where the configuration is done by the BridgeX
  and this configuration is passed to the host mlx4_vnic driver using the EoIB
  protocol.
Both modes of operation require the presence of a BridgeX gateway
in order to work properly. The EoIB driver supports a mixture
of host and network administered vNics.

2.1 EoIB Host Administered vNic
-------------------------------
In the host administered mode, vNics are configured using static configuration
files located on the host side. These configuration files define the number of
vNics, and the vHub that each host administered vNic will belong to (i.e.,  the
vNic's BridgeX box, eport and VLAN id properties). The mlx4_vnic_confd service
is used to read these configuration files and pass the relevant data to the
mlx4_vnic module.
EoIB Host Administered vNic supports two forms of configuration files:
- A central configuration file (mlx4_vnic.conf)
- vNic-specific configuration files (ifcfg-eth<x>)
Both forms of configuration supply the same functionality. If both forms of
configuration files exist, the central configuration file has precedence and
only this file will be used.

2.1.1 Central Configuration File - /etc/infiniband/mlx4_vnic.conf
----------------------------------------------------------------------
The mlx4_vnic.conf file consists of lines, each describing one vNic. The
following file format is used:
name=eth44 mac=00:25:8B:27:14:78 ib_port=mlx4_0:1 vid=3 vnic_id=5 bx=00:00:00:00:00:00:04:B2 eport=A10
name=eth45 mac=00:25:8B:27:15:78 ib_port=mlx4_0:1       vnic_id=6 bx=00:00:00:00:00:00:05:B2 eport=A10
name=eth47 mac=00:25:8B:27:16:84 ib_port=mlx4_0:1 vid=2 vnic_id=7 bx=BX001 eport=A11
name=eth40 mac=02:AA:8B:27:17:93 ib_port=mlx4_0:2       vnic_id=8 bx=BX001 eport=A12

The fields used in the file have the following meaning:
name     - The name of the interface that is displayed when running ifconfig.
mac      - The mac address to assign to the vNic.
ib_port  - The device name and port number in the form
           [device name]:[port number]. The device name can be retrieved by
           running ibv_devinfo and using the output of hca_id field. The port
           number can have a value of  1 or 2.
vid      - [Optional] If VLAN ID exists, the vNic will be assigned
           the specified VLAN ID. This value must be between 0 and 4095.
           - If the vid is set to 'all', the ALL-VLAN mode will be enabled and
             the vNic will support multiple vNic tags.
           - If no vid is specified or value -1 is set, the vNic will be assigned
             to the default vHub associated with the GW.
vnic_id  - A unique vNic number per GW per IOA, between 0 and 16K.
bx       - The BridgeX box system GUID or system name string.
eport    - The string describing the eport name.
pkey     - [Optional] If discovery_pkey module parameter is set, this
           value will control which partitions would be used to
           discover the gateways. For more information about discovery_pkeys 
           please refer to section #3.6.

2.1.2 vNic Specific Configuration Files - ifcfg-eth<x>
---------------------------------------------------------
EoIB configuration can use the ifcfg-eth<x> files used by the network
service to derive the needed configuration. In such case, a separate file
is required per vNic. Additionally, you need to update the ifcfg-eth<x> file
and add some new attributes to it.

On Red Hat the new file will be of the form:
DEVICE=eth2
HWADDR=00:30:48:7d:de:e4
BOOTPROTO=dhcp
ONBOOT=yes
BXADDR=BX001
BXEPORT=A10
VNICIBPORT=mlx4_0:1
VNICVLAN=3  (Optional field)
GW_PKEY=0xfff1

The fields used in the file have the following meaning:
DEVICE     - An optional field. The name of the interface that is displayed
             when running ifconfig. If it is not present, the trailer of the
             configuration file name (e.g. ifcfg-eth47 => "eth47") is used
             instead, when used please follow the scheme eth<number>.
BXADDR     - The BridgeX box system GUID or system name string.
BXEPORT    - The string describing the eport name.
VNICVLAN   - An optional field. If it exists, the vNic will be assigned the
             VLAN ID specified. This value must be between 0 and 4095, or
             'all' for ALL-VLAN feature.
VNICIBPORT - The device name and port number in the form
             [device name]:[port number]. The device name can be retrieved by
             running ibv_devinfo and using the output of hca_id field. The port
             number can have a value of 1 or 2.
HWADDR     - The mac address to assign the vNic.
GW_PKEY	   - An optional field. If discovery_pkey module parameter is set, this
             value will control which partitions would be used to
             discover the gateways. For more information about discovery_pkeys 
             please refer to section #3.6.

Other fields available for regular eth interfaces in the ifcfg-eth<x> files may
also be used.

2.1.3 mlx4_vnic_confd

After updating the configuration files you are ready to create the host
administered vNics.

Usage: /etc/init.d/mlx4_vnic_confd {start|stop|restart|reload|status}

Notes:
- This script manages host administrated vNics only, to retrieve general
  information on the vNics on the system including network administrated vNics,
  refer to mlx4_vnic_info section 3.1
- When using BXADDR/bx field, all vNics BX address configuration should be
  consistent: either all of them use GUID format, or name format.
- The MAC and VLAN values are set using the configuration files only, other
  tools such as (vconfig) for VLAN modification, or (ifconfig) for MAC
  modification, are not supported.

2.2 EoIB Network Administered vNic
----------------------------------
In network administered mode, the configuration of the vNic is done by the
BridgeX. If a vNic is configured for a specific host, it will appear on that
host once a connection is established between the BridgeX and the mlx4_vnic
module. This connection between the mlx4_vnic modules and all available BridgeX
boxes is established automatically  when the mlx4_vnic module is loaded. If the
BridgeX is configured to remove the vNic, or if the connection between the host
and BridgeX is lost, the vNic interface will disappear (running ifconfig will
not display the interface). Similar to host administered vNics, a network
administered, vNic resides on a specific vHub.

See BridgeX documentation on how to configure a network administered vNic.

To disable network administered vNics on the host side load mlx4_vnic module
with the net_admin module parameter set to 0.

2.3 VLAN configuration
----------------------
As explained in the topology section, a vNic instance is associated with a
specific vHub group. This vHub group is connected to a BridgeX external port
and has a VLAN tag attribute. When creating/configuring a vNic you define
the VLAN tag it will use via the vid or the VNICVLAN fields (if these fields
are absent, the vNic will not have a VLAN tag). The vNic's VLAN tag will be
present in all EoIB packets sent by the vNics and will be verified on
all packets received on the vNic. When passed from the InfiniBand to Ethernet,
the EoIB encapsulation will be disassembled but the VLAN tag will
remain.

For example, if the vNic "eth2" is associated with a vHub that uses BridgeX
"bridge01", eport "A10" and VLAN tag 8, all incoming and outgoing traffic on
eth2 will use a VLAN tag of 8. This will be enforced by both BridgeX and
destination hosts. When a packet is passed from the internal fabric to the
Ethernet subnet through the BridgeX it will have a "true" Ethernet VLAN tag
of 8.

The VLAN implementation used by EoIB uses OS unaware VLANs. This is in many
ways similar to switch tagging in which an external Ethernet switch adds/strips
tags on traffic preventing the need of OS intervention. EoIB does not support
OS aware VLANs in the form of vconfig.

2.3.1 Configuring VLANs
-------------------------
To configure VLAN tag for a vNic, add the VLAN tag property to the
configuration file in host administrated mode, or configure the vNic on the
appropriate vHub in network administered mode.
In the host administered mode when a vHub with the requested VLAN tag is not
available, the vNIC's login request will be rejected.

Host administered VLAN configuration in centralized configuration file:
Add "vid=<VLAN tag>" or remove vid property for no VLAN

Host administered VLAN configuration with ifcfg-eth<x>  configuration files
Add "VNICVLAN=<VLAN tag>" or remove VNICVLAN property for no VLAN

Notes:
o Using a VLAN tag value of 0 is not recommended because the traffic using
  it would not be separated from non VLAN traffic.
o For Host administered vNics, VLAN entry must be set in the BridgeX first,
  refer to BridgeX documentation for more information.
  
For information of how to configure ALL VLAN mode, 
please refer to section 3.7 ALL VLAN  

2.4 EoIB Multicast Configuration
-----------------------------------
Configuring Multicast for EoIB interfaces is identical to multicast
configuration for native Ethernet interfaces.
Note: EoIB maps Ethernet multicast addresses to InfiniBand MGIDs (Multicast
      GID). It ensures that different vHubs use mutually exclusive MGIDs.
      Thus preventing vNics on different vHubs from communicating
      with one another.

2.5 EoIB and QoS
----------------
EoIB enables the use of InfiniBand service levels. The configuration of the SL
is performed through the BridgeX and allows you set different data/control
service level values per BridgeX box.
Please refer to BridgeX documentation for the use of non default SL.

2.6 IP Configuration Based on DHCP
----------------------------------
Setting an EoIB interface configuration based on DHCP (v3.1.2 which is
available via www.isc.org) is performed similarly to the configuration of
Ethernet interfaces. When setting the EoIB configuration files, verify that
it includes following lines:
For RedHat: BOOTPROTO=dhcp
For SLES: BOOTPROTO='dchp'

Note: If EoIB configuration files are included, ifcfg-eth<x> files will be
      installed under:
      /etc/sysconfig/network-scripts/ on a RedHat machine
      /etc/sysconfig/network/ on a SuSE machine

2.6.1 DHCP Server
-----------------------
Using a DHCP server with EoIB does not require special configuration. The DHCP
server can run on a server located on the Ethernet side (using any
Ethernet HW) or on a server located on the InfiniBand side and running EoIB
module.

2.7 Static EoIB Configuration
-----------------------------
To configure a static EoIB you can use an EoIB configuration that is not based
on DHCP. Static configuration is similar to a typical Ethernet device
configuration. See your Linux distribution documentation for additional
information about configuring IP addresses.

Note: Ethernet configuration files are located at:
      /etc/sysconfig/network-scripts/ on a RedHat machine
      /etc/sysconfig/network/ on a SuSE machine

2.8 Sub Interfaces (VLAN)
-------------------------
EoIB interfaces do not support creating sub interfaces via the vconfig
command, unless working in ALL VLAN mode.
To create interfaces with VLAN, refer to the VLAN section 2.3.1.


3. EoIB Usage and Configuration
===============================

3.1 mlx4_vnic_info
------------------
To retrieve information regarding EoIB interfaces, use the script
mlx4_vnic_info. This script provides detailed information about a specific vNic
or all EoIB vNic interfaces, such as: BX info, IOA info, SL, PKEY, Link state
and interface features. If network administered vNics are enabled, this script
can also be used to discover the available BridgeX boxes from the host side.

To discover the available gateway, run:
	# mlx4_vnic_info -g

To receive the full vNic information of eth2, run:
	# mlx4_vnic_info -i eth2

For shorter information report on eth2, run:
	# mlx4_vnic_info -s eth2

For help and usage, run:
	# mlx4_vnic_info --help

3.2 ethtool
-----------
ethtool application is another method to retrieve interface information
and change its configuration. EoIB interfaces support ethtool
similarly to Hardware Ethernet interfaces.
The supported ethtool options include the following options:
- To check driver and device information run:
  # ethtool -i eth<x>
- To query stateless offload status run:
  # ethtool -k eth<x>
- To set stateless offload status run:
  # ethtool -K eth<x> [rx on|off] [tx on|off] [sg on|off] [tso on|off]
- To query interrupt coalescing settings run:
  # ethtool -c eth<x>
  Note: By default, the driver uses adaptive interrupt moderation for the
        receive path, which adjusts the moderation time according to the
        traffic pattern.
- To set interrupt coalescing settings run:
  # ethtool -C eth<x> [rx-usecs N] [rx-frames N]
  Note: usec settings correspond to the time to wait after the *last* packet
        sent/received before triggering an interrupt.
        If rx-usec or rx-frames is set to zero, then an interrupt will be
        triggered right away upon packet reception (latency mode).
- To obtain additional device statistics, run:
  # ethtool -S eth<x>
  Note: Since RX rings are per IOA, the RX packets/bytes statistics are
        shared among all vNics running over this IOA.
- For more information on ethtool run: ethtool -h

3.3 Link State
--------------
An EoIB interface can report two different link states:

- The physical link state of the interface that is made up of the actual HCA
  port link state and the status of the vNics connection with the BridgeX.
  If the HCA port link state is down or the EoIB connection with the BridgeX
  has failed,  the link will be reported as down because without the connection
  to the BridgeX the EoIB protocol cannot work and no data can be sent on the
  wire. The mlx4_vnic driver can also report the status of the external BridgeX
  port status by using the mlx4_vnic_info script.
  If the eport_state_enforce module parameter is set, then the external port
  state will be reported as the vNic interface link state.
  If the connection between the vNic and the BridgeX is broken
  (hence the external port state is unknown)the link will be reported as down.

- The link state of the external port associated with the vNic interface

Note: A link state is down on a host administrated vNic, when the BridgeX is
      connected and the InfiniBand fabric appears to be functional. The issue
      might result from a misconfiguration of either
      BXADDR or/and BXEPORT configuration file.

To query the link state run the following command and look for "Link detected":
# ethtool eth<x> | grep Link

3.4 Offloads and Features
-------------------------
EoIB driver supports different offloads and features for optimal performance.
The following features are supported by the driver and can be controlled via
ethtool:
- TX/RX Checksumming
- LRO/TSO
- Scatter Gather
- Interrupt Coalescing

It also supports other features, such as:
- Per-core NAPI (enabled by default)
- Multiple TX/RX rings (see section #4.1 for more details)
- Receive-Side Scaling (RSS) and Transmit-Side Scaling (TSS), these features
  require kernel support, to check whether the driver has enabled RSS/TSS, run:
  # mlx4_vnic_info -i eth<x> | egrep 'TSS|RSS'
  Note that RSS is supported in Software, and it's configurable by the BridgeX
  Administrator. For optimal performance, Software RSS size should be equal or
  larger than the host RX rings number.
- Jumbo frames

3.5 Jumbo Frames
----------------
EoIB supports jumbo frames up to the InfiniBand limit of 4K bytes.
The default Maximum Transmit Unit (MTU) for EoIB driver is 1500 bytes.
To configure EoIB to work with jumbo frames:
- Make sure that the IB HCA and Switches hardware support 4K MTU.
- Configure Mellanox low level driver to support 4K MTU. Add the
  mlx4_core module parameter to set_4k_mtu=1
- Change the MTU value of the vNic, for example, run:
  # ifconfig eth2 mtu 4034

Note: Due to EoIB protocol overhead, the maximum MTU value that can be set for
      the vNic interface is: 4034 bytes.

3.6 Discovery Partitions Configuration
--------------------------------------
EoIB enables to map VLANs to IB partitions. Mapping VLANs to partitions will
cause all EoIB data traffic and all the vNic related control traffic
to be sent on the mapped partitions. In rare cases it might be useful to
ensure that EoIB discovery packets (packets used for discovery of Gateways
(GWs) and vice versa) should be sent on the non default partition.
This might be used to limit and enforce the visibility of GWs by different
hosts.
The discovery_pkeys module parameter can be used to define which partitions
would be used in discovering GWs. The module parameters enables using up to
24 different PKEYs. If not set, the default PKEY will be used, and only GWs
using the default PKEY would be discovered.

For example, to configure a host to discover GWs on  three partitions
0xffff,0xfff1 and 0x3 add the following line to modprobe configuration file:
options mlx4_vnic discovery_pkeys=0xffff,0xfff1,0x3

When using this feature combined with host administrated vnics, each vnic
should also be configured with the partition it should be created on.

For example, for creating host admin vnic on I/F eth20, with pkey 0xfff1 add
the following line to ifcg-eth20:
GW_PKEY=0xfff1

Note: When using non default partition, the GW partitions should also be
      configured on the GW in the BridgeX. Also, the Subnet Manager must be
      configured accordingly.

3.7 ALL VLAN
--------------------------------------
In Ethernet over InfiniBand (EoIB), a vNic is a member of a vHUB that uniquely 
defines its Virtual Local Area Networks (VLAN) tag. The VLAN tag is used in 
the VLAN header within the EoIB packets, and is enforced by EoIB hosts when 
handling the EoIB packets. The tag is also extended to the Ethernet fabric when
packets pass through the BridgeX®. This model of operation ensures a high level
of security however, it requires each VLAN tag used to have its own individual
vNic to be created and each vHub requires InfiniBand fabric resources 
like multicast groups(MGIDs). 
If many VLANs are needed, the resources required to create and manage 
them are large. ALL VLAN vHub enables the user to use its resources efficiently 
by creating a vNic that can support multiple VLAN tags without 
creating multiple vNics. However, it reduces VLAN separation 
compared to the vNic /vHub model. 

3.7.1 All-VLAN Functionality
--------------------------------------
When ALL-VLAN is enabled, the address lookup on the BridgeX® consists of 
the MAC address only (without the VLAN), so all packets with the same MAC 
regardless of the VLAN, are sent to the same InfiniBand address. 
Same behavior can be expected from the host EoIB driver, which
also sends packets to the relevant InfiniBand addresses while 
disregarding the VLAN. In both scenarious, the Ethernet packet that is embedded 
in the EoIB packet includes the VLAN header enabling VLAN enforcement either 
in the Ethernet fabric or at the receiving EoIB host.

Notes:
* This feature must be supported by both the BridgeX and by the host side.
* When enabling this features all gateways (LAG or legacy) that
  have eports belonging to a gateway group (GWG) must be configured 
  to the same behavior. For example it is impossible to have gateway A2 
  configured to all-vlan mode and A3 to regular mode, because both belong to GWG A.
* A gateway that is configured to work in All-VLAN mode cannot accept login
  requests from 
  - vNics that do not support this mode
  - host admin vNics that were not configured to work in ALL VLAN mode, by
    setting the vlan-id value to a 'all', as described in section 3.7.2.2.

3.7.2 Creating vNICs that support All-VLAN mode
------------------------------------------------
VLANs are created on a vNIC that supports All-VLAN mode using "vconfig".

3.7.2.1 net-admin vNics
----------------------------------------------
The net-admin vNic supports All-VLAN mode once it is created on a gateway 
configured with All-VLAN mode.

3.7.2.2 host-admin vNics
-----------------------------------------------
To create an All-VLAN vnic, set the VLAN's ID to 'all'. 

A gateway that is configured to work in ALL VLAN mode, can only accept login
requests from hosts that are also working in a VLAN mode. e.g. the VLAN ID 
must be set to 'all'. 

* This is an example of how to create an All-VLAN vNic using the mlx4_vnic.conf file:
  name=eth44 mac=00:25:8B:27:14:78 ib_port=mlx4_0:1 vid=all vnic_id=5 bx=00:00:00:00:00:00:04:B2 eport=A10

* To create an All-VLAN vNic using a specific configuration file, add the following 
  line to the configuration file:
  VNICVLAN=all

For further information on how to create host-admin vNics, 
see section '2.1 EoIB Host Administered vNic'.

3.7.3 Checking the Configuration
------------------------------------------------
To verify the gateway / vNic is configured with the All-VLAN mode, use the
mlx4_vnic_info script. 

3.7.3.1 Gateway Support
------------------------------------------------
To verify the gateway is configured to All-VLAN mode. Run:
mlx4_vnic_info -g <GW-NAME>
for example:

# mlx4_vnic_info -g A2
IOA_PORT      mlx4_0:1
BX_NAME       bridge-119c64
BX_GUID       00:02:c9:03:00:11:61:67
EPORT_NAME    A2
EPORT_ID      63
STATE         connected
GW_TYPE       LEGACY
PKEY          0xffff
ALL_VLAN      yes

3.7.3.2 vNic Support
--------------------------------------------------
To verify the vNic is configured to All-VLAN mode. Run:
mlx4_vnic_info -i <interface>

for example:
# mlx4_vnic_info -i eth204
NETDEV_NAME   eth204
NETDEV_LINK   up
NETDEV_OPEN   yes
.
.
.
GW_TYPE       LEGACY
ALL_VLAN      yes
.
.

For further information on mlx4_vnic_info script, 
see section '3.1 mlx4_vnic_info'

4. Advanced EoIB Settings
=========================

4.1 Module Parameters
---------------------
The mlx4_vnic driver supports the following module parameters. These parameters
are intended to enable more specific configuration of the mlx4_vnic driver to
customer needs. The mlx4_vnic is also effected by module parameters of other
modules such as set_4k_mtu of mlx4_core. This modules are not addressed in this
document.

The available module parameters include:
* tx_rings_num: Number of TX rings, use 0 for #cpus [default 0, max 32]
* tx_rings_len: Length of TX rings, must be power of two [default 1024, max 8K]
* rx_rings_num: Number of RX rings, use 0 for #cpus [default 0, max 32]
* rx_rings_len: Length of RX rings, must be power of two [default 2048, max 8K]
* vnic_net_admin: Network administration enabled [default 1]
* lro_num: Number of LRO sessions per ring,
  use 0 to disable LRO [default 32, max 32]
* eport_state_enforce: Bring vNic up only when corresponding External Port
  is up [default 0]
* discovery_pkeys: Vector of up to 24 PKEYs to be used for discovery
  [default 0xFFFF] (array of int)
* sa_query: Queries the subnet agent about each InfiniBand address and ignores gateway assigned SLs
  [default 0]. It is required for toplogies that require special InfiniBand path configuration (such
as 3D Torus).

For all module parameters list and description, run:
# mlx4_vnic_info -I

To check the current module parameters, run:
# mlx4_vnic_info -P

Notes:
- Default RX/TX rings number is the number of logical CPUs (threads).
- To set non-default values to module parameters, the following line should be
  added to modprobe configuration file (e.g. /etc/modprobe.conf):
  "options mlx4_vnic <param_name>=<value> <param_name>=<value> ..."
- For more information about discovery_pkeys please refer to section #3.6

4.2 Bonding
-----------
EoIB uses the standard Linux bonding driver. For more information on the Linux
Bonding driver please refer to:
<kernel-source>/Documentation/networking/bonding.txt.
Currently only fail-over modes are supported by the EoIB driver, load-balancing
modes including static and dynamic (LACP) configurations are not supported.

4.3 vNic Interface Naming
-------------------------
The mlx4_vnic driver enables the kernel to determine the name of the registered
vNic. By default, the Linux kernel assigns each vNic interface the name eth<x>,
where <x> is an incremental number that keeps the interface name unique in the
system. The vNic interface name may not remain consistent among hosts or
BridgeX reboots as the vNic creation can happen in a different order each time.
Therefore, the interface name may change because of a "first-come-first-served"
kernel policy. In automatic network administered mode, the vNic MAC address may
also change, which makes it difficult to keep the interface configuration
persistent.

To control the interface name, you can use standard Linux utilities such as
IFRENAME(8), IP(8) or UDEV(7). For example, to change the interface eth2 name
to eth.bx01.a10, run:
#ifrename -i eth2 -n eth.bx01.a10

To generate a unique vNic interface name, use the mlx4_vnic_info script with
the '-u' flag. The script will generate a new name based on the scheme:
eth<pci-id>.<ib-port-num>.<gw_port_id>.[vlan-id]
For example, if vNic eth2 resides on an InfiniBand card on the PCI BUS ID
0a:00.0 PORT #1, and is connected to the GW PORT ID #3 without VLAN, its unique
name will be:
# mlx4_vnic_info -u eth2
eth2   eth10.1.3

You can add your own custom udev rule to use the output of the script and
to rename the vNic interfaces automatically.
To create a new udev rule file under /etc/udev/rules.d/61-vnic-net.rules,
include the line:
SUBSYSTEM=="net", PROGRAM=="/sbin/mlx4_vnic_info -u %k", NAME="%c{2+}"

Notes:
- UDEV service is active by default however if it is not active, run:
  # /sbin/udevd -d
- When vNic MAC address is consistent, you can statically name each interface
  using the UDEV following rule:
  SUBSYSTEM=="net", SYSFS{address}=="aa:bb:cc:dd:ee:ff", NAME="eth<x>"
  Refer to udev man pages for more details on UDEV rules syntax.

4.4 Para-Virtualized vNic
-------------------------
EoIB driver interfaces can be also used for Linux based virtualization
environment such Xen/KVM based Hypervisors, this section explains how to
configure Para-Virtualized (PV) EoIB to work in such an environment.

4.4.1 Driver Configuration:
For PV-EoIB the following features must be disabled in the driver:
o Large Receive Offload (LRO)
o TX completion polling
o RX fragmented buffers

These features can be controlled as module parameters:
Edit modprobe configuration file and include the line:
 options mlx4_vnic lro_num=0 tx_polling=0 rx_linear=1

For the full list of mlx4_vnic module parameters, run:
# modinfo mlx4_vnic

4.4.2 Network Configuration
PV-EoIB supports both L2 (bridged) and L3 (routed) network models.
The 'physical' interfaces that can be enslaved to the Hypervisor virtual bridge
are actually EoIB vNics, and they can be created as on native Linux machine,
PV-EoIB driver supports both host-administrated and network-administrated
vNics, please refer to section #2 for more information on vNics configuration.

Once an EoIB vNic is enslaved to a virtual bridge, it can be used by any
Guest OS that's supported by the Hypervisor, the driver will automatically
manage the resources required to serve the Guest OS network virtual interfaces
(based on their MAC address)
To see the list of MAC addresses served by an EoIB vNic, log into the
Hypervisor and run the command:
# mlx4_vnic_info -m <interface>

Note:
The driver detects virtual interfaces MAC addresses based in their outgoing
packets, so you may notice that the virtual MAC address is being detected by
the EoIB driver only after the first packet is sent out by the Guest OS.
Virtual resources MAC addresses cleanup is managed by mlx4_vnic daemon
explained later (Garbage Collection).

4.4.3 Multicast Configuration
Virtual machines multicast traffic over PV-EoIB is supported in promiscuous
mode, this means that all multicast traffic is sent over the broadcast domain,
and filtered in the VM level.
To enable promiscuous multicast, log into the BridgeX CLI and run the command:
BXCLI# bxm eoib mcast promiscuous
Please refer to BridgeX documentation for more details.

To see the multicast configuration of a vNic from the host, log into the
Hypervisor and run:
# mlx4_vnic_info -i <interface> | grep MCAST

4.4.4 VLANs
Virtual LANs are supported in EoIB vNic level, VLAN tagging/untagging is done
by EoIB driver. To enable VLANs on top of EoIB vNic, a new vNic interface must
be created with the corresponding VLAN ID, then it can be enslaved to a virtual
bridge and used by the Guest OS, VLAN tagging/untagging is transparent to the
Guest and managed in EoIB driver level.

Notes:
o The vconfig utility is not supported by EoIB driver, a new vNic instance must
  be created instead, see section #2.3 for details.
o Virtual Guest Tagging (VGT) is not supported. The model explained above
  applies to Virtual Switch Tagging (VST) only.

4.4.5 Migration
Some Hypervisors gives you the ability to migrate a virtual machine from one
physical server to another, this feature is seamlessly supported by PV-EoIB.
Any network connectivity over EoIB will automatically resume on the new
physical server, the downtime that may occur during this process is minor.

4.4.6 Resources Cleanup
When a virtual interface within the Guest OS is no longer connected to
EoIB link, its MAC address need to be cleaned up the EoIB driver, this task
is managed by the Garbage Collector (GC) service.
The GC functionality is included in the mlx4_vnic daemon (python script):
# /sbin/mlx4_vnicd

To enable/disable the mlx4_vnic daemon, edit /etc/infiniband/mlx4_vnic.conf
and include the line:
# mlx4_vnicd=<yes|no> [parameters]

Then, start the service mlx4_vnic_confd to read and apply the configuration:
# /etc/init.d/mlx4_vnic_confd start

To see full list of the daemon parameters, run:
# mlx4_vnicd --help

For example, to enable mlx4_vnic daemon with GC:
# cat /etc/infiniband/mlx4_vnic.conf
  mlx4_vnicd=yes gc_enable=yes

# /etc/init.d/mlx4_vnic_confd start
  Checking configuration file:                               [  OK  ]
  Starting mlx4_vnicd (pid 30920):                           [  OK  ]

Notes:
o The mlx4_vnicd daemon requires xenstore or libvirt to run.
o Some Hypervisors may not have enough memory for the driver domain, as a
  result mlx4_vnic driver may fail to initialize or create more vNics, this may
  cause the machine to be unresponsive. To avoid this behaviour, allocate more
  memory for the driver domain.
  For example, refer to this link for instructions on how to increase dom0_mem
  in XenServer 5.6: http://support.citrix.com/article/CTX126531
  You may also try to lower mlx4_vnic driver memory consumption by decreasing
  its RX/TX rings number and length, please refer to section #4.1 for more
  details on mlx4_vnic driver module parameters modification.

4.5 EoIB Subnet Agent Query
-------------------------
When enabled, the VNIC will ignore the SLs sent from BXM and will initiate an SA
path query to OpenSM to query the correct parameters of EoIB path required for
communicating with the end node (EoIB neighbor/BXM Gateway port).
This feature is only supported with FabricIT-BXM v2.1.2000 or newer, the BXM
needs a special configuration to work with EoIB VNIC driver when SA query is
enabled  (For further information, please see BXM the README files).

This feature is required when a special EoIB path configuration in OpenSM is needed,
for example when running on special IB topologies such as 3D Torus.

To enalble: set sa_query=1 in mlx4_vnic module parameters.

