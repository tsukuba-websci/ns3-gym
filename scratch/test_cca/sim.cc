/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Piotr Gawlowicz
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 * Based on script: ./examples/tcp/tcp-variants-comparison.cc
 *
 * Topology:
 *
 *   Right Leafs (Clients)                      Left Leafs (Sinks)
 *           |            \                    /        |
 *           |             \    bottleneck    /         |
 *           |              R0--------------R1          |
 *           |             /                  \         |
 *           |   access   /                    \ access |
 *           N -----------                      --------N
 */

#include <iostream>
#include <fstream>
#include <string>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/applications-module.h"
#include "ns3/error-model.h"
#include "ns3/tcp-header.h"
#include "ns3/enum.h"
#include "ns3/event-id.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/traffic-control-module.h"

#include "ns3/opengym-module.h"
#include "tcp-rl.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpVariantsComparison");

static std::vector<uint32_t> rxPkts;

static void
CountRxPkts(uint32_t sinkId, Ptr<const Packet> packet, const Address & srcAddr)
{
  rxPkts[sinkId]++;
}

static void
PrintRxCount()
{
  uint32_t size = rxPkts.size();
  NS_LOG_UNCOND("RxPkts:");
  for (uint32_t i=0; i<size; i++){
    NS_LOG_UNCOND("---SinkId: "<< i << " RxPkts: " << rxPkts.at(i));
  }
}


// Throughput trace
double oldTotalRx = 0;
double newTotalRx = 0;
void TraceThroughput(Ptr<Application> app, Ptr<OutputStreamWrapper> stream, double timestep)
{
    Ptr<PacketSink> pktsink = DynamicCast<PacketSink>(app);
    newTotalRx = pktsink->GetTotalRx();

    // Calculate throughput
    *stream->GetStream() << Simulator::Now ().GetSeconds () << ", " << (newTotalRx - oldTotalRx) * 8.0 / timestep / 1024 << std::endl;
    oldTotalRx = newTotalRx;

    // Schedule next trace
    Simulator::Schedule(Seconds(timestep), &TraceThroughput, app, stream, timestep);
}


int main (int argc, char *argv[])
{
  LogComponentEnable("TcpVariantsComparison", LOG_LEVEL_ALL);
  LogComponentDisableAll(LOG_PREFIX_TIME);

  uint32_t openGymPort = 5555;
  double tcpEnvTimeStep = 0.1;

  uint32_t nLeaf = 2;
  std::string transport_prot = "TcpRl";
  double error_p = 0.0;
  std::string bottleneck_bandwidth = "10Mbps";
  std::string bottleneck_delay = "45ms";
  std::string access_bandwidth = "10Mbps";
  std::string access_delay = "45ms";
  std::string cross_traffic_data_rate = "6Mbps";
  std::string prefix_file_name = "TcpVariantsComparison";
  uint64_t data_mbytes = 0;
  uint32_t mtu_bytes = 400;
  double duration = 10.0;
  uint32_t run = 0;
  bool sack = true;
  std::string queue_disc_type = "ns3::PfifoFastQueueDisc";
  std::string recovery = "ns3::TcpClassicRecovery";

  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", run);
  cmd.AddValue ("envTimeStep", "Time step interval for time-based TCP env [s]. Default: 0.1s", tcpEnvTimeStep);
  // other parameters
  cmd.AddValue ("nLeaf",     "Number of left and right side leaf nodes", nLeaf);
  cmd.AddValue ("transport_prot", "Transport protocol to use: TcpNewReno, "
                "TcpHybla, TcpHighSpeed, TcpHtcp, TcpVegas, TcpScalable, TcpVeno, "
                "TcpBic, TcpYeah, TcpIllinois, TcpWestwood, TcpWestwoodPlus, TcpLedbat, "
		            "TcpLp, TcpRl, TcpRlTimeBased", transport_prot);
  cmd.AddValue ("error_p", "Packet error rate", error_p);
  cmd.AddValue ("bottleneck_bandwidth", "Bottleneck bandwidth", bottleneck_bandwidth);
  cmd.AddValue ("bottleneck_delay", "Bottleneck delay", bottleneck_delay);
  cmd.AddValue ("access_bandwidth", "Access link bandwidth", access_bandwidth);
  cmd.AddValue ("access_delay", "Access link delay", access_delay);
  cmd.AddValue ("cross_traffic_data_rate", "Data rate of cross traffic", cross_traffic_data_rate);
  cmd.AddValue ("prefix_name", "Prefix of output trace file", prefix_file_name);
  cmd.AddValue ("data", "Number of Megabytes of data to transmit", data_mbytes);
  cmd.AddValue ("mtu", "Size of IP packets to send in bytes", mtu_bytes);
  cmd.AddValue ("duration", "Time to allow flows to run in seconds", duration);
  cmd.AddValue ("run", "Run index (for setting repeatable seeds)", run);
  cmd.AddValue ("queue_disc_type", "Queue disc type for gateway (e.g. ns3::CoDelQueueDisc)", queue_disc_type);
  cmd.AddValue ("sack", "Enable or disable SACK option", sack);
  cmd.AddValue ("recovery", "Recovery algorithm type to use (e.g., ns3::TcpPrrRecovery", recovery);
  cmd.Parse (argc, argv);

  transport_prot = std::string ("ns3::") + transport_prot;

  SeedManager::SetSeed (1);
  SeedManager::SetRun (run);

  NS_LOG_UNCOND("Ns3Env parameters:");
  if (transport_prot.compare ("ns3::TcpRl") == 0 or transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
  {
    NS_LOG_UNCOND("--openGymPort: " << openGymPort);
  } else {
    NS_LOG_UNCOND("--openGymPort: No OpenGym");
  }

  NS_LOG_UNCOND("--seed: " << run);
  NS_LOG_UNCOND("--Tcp version: " << transport_prot); 

  Ptr<OpenGymInterface> openGymInterface;
  if (transport_prot.compare ("ns3::TcpRl") == 0)
  {
    openGymInterface = OpenGymInterface::Get(openGymPort);
    Config::SetDefault ("ns3::TcpRl::Reward", DoubleValue (2.0)); // Reward when increasing congestion window
    Config::SetDefault ("ns3::TcpRl::Penalty", DoubleValue (-30.0)); // Penalty when decreasing congestion window
  } 

  // Calculate the ADU size
  Header* temp_header = new Ipv4Header ();
  uint32_t ip_header = temp_header->GetSerializedSize ();
  NS_LOG_LOGIC ("IP Header size is: " << ip_header);
  delete temp_header;
  temp_header = new TcpHeader ();
  uint32_t tcp_header = temp_header->GetSerializedSize ();
  NS_LOG_LOGIC ("TCP Header size is: " << tcp_header);
  delete temp_header;
  uint32_t tcp_adu_size = mtu_bytes - 20 - (ip_header + tcp_header);
  NS_LOG_LOGIC ("TCP ADU size is: " << tcp_adu_size);

  // Set the simulation start and stop time
  double start_time = 0.1;
  double stop_time = start_time + duration;

  // 4 MB of TCP buffer
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (1 << 21));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (1 << 21));
  Config::SetDefault ("ns3::TcpSocketBase::Sack", BooleanValue (sack));
  Config::SetDefault ("ns3::TcpSocket::DelAckCount", UintegerValue (2));


  Config::SetDefault ("ns3::TcpL4Protocol::RecoveryType",
                      TypeIdValue (TypeId::LookupByName (recovery)));
  // Select TCP variant
  if (transport_prot.compare ("ns3::TcpWestwoodPlus") == 0)
    {
      // TcpWestwoodPlus is not an actual TypeId name; we need TcpWestwood here
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType", TypeIdValue (TcpWestwood::GetTypeId ()));
      // the default protocol type in ns3::TcpWestwood is WESTWOOD
      Config::SetDefault ("ns3::TcpWestwood::ProtocolType", EnumValue (TcpWestwood::WESTWOODPLUS));
    }
  else
    {
      TypeId tcpTid;
      NS_ABORT_MSG_UNLESS (TypeId::LookupByNameFailSafe (transport_prot, &tcpTid), "TypeId " << transport_prot << " not found");
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType", TypeIdValue (TypeId::LookupByName (transport_prot)));
    }

  // Configure the error model
  // Here we use RateErrorModel with packet error rate
  Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable> ();
  uv->SetStream (50);
  RateErrorModel error_model;
  error_model.SetRandomVariable (uv);
  error_model.SetUnit (RateErrorModel::ERROR_UNIT_PACKET);
  error_model.SetRate (error_p);

  // Create the point-to-point link helpers
  PointToPointHelper bottleNeckLink;
  bottleNeckLink.SetDeviceAttribute  ("DataRate", StringValue (bottleneck_bandwidth));
  bottleNeckLink.SetChannelAttribute ("Delay", StringValue (bottleneck_delay));
  bottleNeckLink.SetDeviceAttribute  ("ReceiveErrorModel", PointerValue (&error_model));

  PointToPointHelper pointToPointLeaf;
  pointToPointLeaf.SetDeviceAttribute  ("DataRate", StringValue (access_bandwidth));
  pointToPointLeaf.SetChannelAttribute ("Delay", StringValue (access_delay));

  PointToPointDumbbellHelper d (nLeaf, pointToPointLeaf,
                                nLeaf, pointToPointLeaf,
                                bottleNeckLink);

  // Install IP stack
  InternetStackHelper stack;
  stack.InstallAll ();

  // Traffic Control
  TrafficControlHelper tchPfifo;
  tchPfifo.SetRootQueueDisc ("ns3::PfifoFastQueueDisc");

  TrafficControlHelper tchCoDel;
  tchCoDel.SetRootQueueDisc ("ns3::CoDelQueueDisc");

  DataRate access_b (access_bandwidth);
  DataRate bottle_b (bottleneck_bandwidth);
  Time access_d (access_delay);
  Time bottle_d (bottleneck_delay);

  uint32_t size = static_cast<uint32_t>((std::min (access_b, bottle_b).GetBitRate () / 8) *
    ((access_d + bottle_d + access_d) * 2).GetSeconds ());

  Config::SetDefault ("ns3::PfifoFastQueueDisc::MaxSize",
                      QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, size / mtu_bytes)));
  Config::SetDefault ("ns3::CoDelQueueDisc::MaxSize",
                      QueueSizeValue (QueueSize (QueueSizeUnit::BYTES, size)));

  if (queue_disc_type.compare ("ns3::PfifoFastQueueDisc") == 0)
  {
    tchPfifo.Install (d.GetLeft()->GetDevice(1));
    tchPfifo.Install (d.GetRight()->GetDevice(1));
  }
  else if (queue_disc_type.compare ("ns3::CoDelQueueDisc") == 0)
  {
    tchCoDel.Install (d.GetLeft()->GetDevice(1));
    tchCoDel.Install (d.GetRight()->GetDevice(1));
  }
  else
  {
    NS_FATAL_ERROR ("Queue not recognized. Allowed values are ns3::CoDelQueueDisc or ns3::PfifoFastQueueDisc");
  }

  // Assign IP Addresses
  d.AssignIpv4Addresses (Ipv4AddressHelper ("10.1.1.0", "255.255.255.0"),
                         Ipv4AddressHelper ("10.2.1.0", "255.255.255.0"),
                         Ipv4AddressHelper ("10.3.1.0", "255.255.255.0"));


  NS_LOG_INFO ("Initialize Global Routing.");
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // Install apps in left and right nodes
  uint16_t port = 50000;
  Address sinkLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
  PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory", sinkLocalAddress);
  ApplicationContainer sinkApps;
  for (uint32_t i = 0; i < d.RightCount (); ++i)
  {
    sinkHelper.SetAttribute ("Protocol", TypeIdValue (TcpSocketFactory::GetTypeId ()));
    sinkApps.Add (sinkHelper.Install (d.GetRight (i)));
  }
  sinkApps.Start (Seconds (0.0));
  sinkApps.Stop  (Seconds (stop_time));

  for (uint32_t i = 0; i < d.LeftCount (); ++i)
  {
    AddressValue remoteAddress (InetSocketAddress (d.GetRightIpv4Address (i), port));
    Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (tcp_adu_size));
    if (i == 0) {
      BulkSendHelper ftp ("ns3::TcpSocketFactory", Address ());
      ftp.SetAttribute ("Remote", remoteAddress);
      ftp.SetAttribute ("SendSize", UintegerValue (tcp_adu_size));
      ftp.SetAttribute ("MaxBytes", UintegerValue (data_mbytes * 1000000));
      ApplicationContainer clientApp = ftp.Install (d.GetLeft (i));
      clientApp.Start (Seconds (start_time * i)); // Start after sink
      clientApp.Stop (Seconds (stop_time - 3)); // Stop before the sink
    } else {
      OnOffHelper onoff ("ns3::TcpSocketFactory", Address());
      onoff.SetConstantRate (DataRate (cross_traffic_data_rate));
      onoff.SetAttribute ("PacketSize", UintegerValue (1024));
      onoff.SetAttribute ("Remote", remoteAddress);
      ApplicationContainer clientApp = onoff.Install (d.GetLeft (i));
      clientApp.Start (Seconds (start_time * i)); // Start after sink
      clientApp.Stop (Seconds (stop_time - 3)); // Stop before the sink
    }
  }

  // Flow monitor
  // FlowMonitorHelper flowHelper;
  // flowHelper.InstallAll();

  // Count RX packets
  for (uint32_t i = 0; i < d.RightCount (); ++i)
  {
    rxPkts.push_back(0);
    Ptr<PacketSink> pktSink = DynamicCast<PacketSink>(sinkApps.Get(i));
    pktSink->TraceConnectWithoutContext ("Rx", MakeBoundCallback (&CountRxPkts, i));
  }

  // Trace setting
  AsciiTraceHelper ascii;
  std::string fname1 = std::string("scratch/test_cca/data/") + std::string("throughput") + std::to_string(openGymPort) + std::string(".csv");
  Ptr<OutputStreamWrapper> stream1 = ascii.CreateFileStream (fname1);

  Simulator::Schedule (Seconds (tcpEnvTimeStep), &TraceThroughput, sinkApps.Get(0), stream1, tcpEnvTimeStep);

  Simulator::Stop (Seconds (stop_time));
  Simulator::Run ();

  // Flow monitor
  // flowHelper.SerializeToXmlFile(prefix_file_name + ".flowmonitor", true, true);

  if (transport_prot.compare ("ns3::TcpRl") == 0 or transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
  {
    openGymInterface->NotifySimulationEnd();
  }

  PrintRxCount();
  Simulator::Destroy ();
  return 0;
}
