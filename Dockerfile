FROM ubuntu:20.04

# Install Install all required dependencies required by ns-3
RUN apt-get update && apt-get install -y gcc g++ python3-pip python git && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ZMQ and Protocol Buffers libs
RUN apt-get update && apt-get install -y libzmq5 libzmq3-dev libprotobuf-dev protobuf-compiler && apt-get clean && rm -rf /var/lib/apt/lists/*

# COPY ns3-gym
COPY . /ns3-gym

# Change WORKDIR
WORKDIR /ns3-gym

# Configure and build ns-3 project
RUN ./waf configure --enable-examples && ./waf build

# Install ns3gym
RUN pip install --user ./src/opengym/model/ns3gym
RUN pip install -U protobuf~=3.20.0

# Run ns3gym
CMD ["python", "./scratch/opengym/simple_test.py"]
