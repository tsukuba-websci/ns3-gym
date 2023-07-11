#include <iostream>
uint32_t nLeaf =  1;
std::string transport_prot = "TcpRl";
double error_p = 0.0;
std::string bottleneck_bandwidth = "2Mbps";
std::string bottleneck_delay = "0.01ms";
std::string access_bandwidth = "10Mbps";
std::string access_delay = "20ms";
std::string prefix_file_name = "TcpVariantsComparison";
uint64_t data_mbytes = 0;
uint32_t mtu_bytes = 400;
double duration = 10.0;
uint32_t run = 0;
bool flow_monitor = false;
bool sack = true;
std::string queue_disc_type = "ns3::PfifoFastQueueDisc";
std::string recovery = "ns3::TcpClassicRecovery";