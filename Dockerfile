FROM ubuntu:20.04

# Install Install all required dependencies required by ns-3
RUN apt-get update && apt-get install -y gcc g++ python3-pip python git

# Install ZMQ and Protocol Buffers libs
RUN apt-get update && apt-get install -y libzmq5 libzmq3-dev libprotobuf-dev protobuf-compiler

# clone ns3-gym repo
COPY . /ns3-gym


# Change WORKDIR
WORKDIR /ns3-gym

# Configure and build ns-3 project
RUN ./waf configure && ./waf build

# Install ns3gym
RUN pip install --user ./src/opengym/model/ns3gym
RUN pip install -U protobuf~=3.20.0

WORKDIR /ns3-gym/scratch/rl-tcp

# Run ns3gym
CMD [ "python3", "test_tcp.py" ]