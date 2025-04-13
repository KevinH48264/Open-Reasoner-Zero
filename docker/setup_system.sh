#!/bin/bash
# sudo ./setup_system.sh (this is when you have a venv and you just need to install things in the system). Refer to setup.sh to install all the dependencies.
# I'm actually unsure how necessary this is, but it came with the docker file. If I run into issues with training, it's worth running this
set -e

echo "Updating apt and installing OS-level dependencies..."
apt-get update && apt-get install -y \
    openssh-server \
    iputils-ping \
    net-tools \
    iproute2 \
    traceroute \
    netcat \
    tzdata \
    build-essential \
    libopenexr-dev \
    libxi-dev \
    libglfw3-dev \
    libglew-dev \
    libomp-dev \
    libxinerama-dev \
    libxcursor-dev && apt-get clean

echo "Installing Java dependencies..."
apt-get update && apt-get install --fix-broken -y default-jre-headless openjdk-8-jdk

echo "Copying nccl.conf to /etc/nccl.conf..."
if [ -f "./nccl.conf" ]; then
    cp ./nccl.conf /etc/nccl.conf
else
    echo "nccl.conf not found; skipping."
fi

echo "Setting time zone to Asia/Shanghai..."
apt-get update && apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get clean

echo "System-level setup complete."
