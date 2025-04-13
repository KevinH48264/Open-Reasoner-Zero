#!/bin/bash
set -e

# Ensure a virtual environment is active (ensure you already ran "pip install -e .")
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: VIRTUAL_ENV is not set. Please activate your virtual environment."
    exit 1
fi

# Use the pip inside the virtual environment
PIP="$VIRTUAL_ENV/bin/pip"

# Set environment variables
export DEBIAN_FRONTEND="noninteractive"
export MAX_JOBS=8

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

echo "Installing torch (version 2.5.1)..."
$PIP install torch==2.5.1

echo "Installing py-spy and pytorch_memlab..."
$PIP install py-spy pytorch_memlab

echo "Installing flash-attn (version 2.7.4.post1) after torch..."
$PIP install flash-attn==2.7.4.post1 --no-build-isolation -U

echo "Installing remaining Python dependencies..."
$PIP install loguru tqdm ninja tensorboard \
    sentencepiece fire tabulate easydict \
    transformers==4.48.1 awscli rpyc pythonping \
    "torchvision==0.20.1" hydra-core accelerate \
    redis opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp prometheus-client \
    omegaconf black==22.8.0 mypy-extensions pathspec tensorboardX nvitop antlr4-python3-runtime==4.11.0 \
    ray==2.40.0 deepspeed==0.16.0 vllm==0.6.5 peft

echo "Setting up proxy (if needed) and installing Hydra from GitHub..."
eval "$(curl -s deploy.i.basemind.com/httpproxy)" && $PIP install git+https://github.com/facebookresearch/hydra.git

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

# I've just manually copied it over. Origina files are called "..._og.py"
# cp parameter_offload.py /data/users/kevihuang/projects/Open-Reasoner-Zero/venv/lib/python3.10/site-packages/deepspeed/runtime/zero/parameter_offload.py
# cp partitioned_param_coordinator.py /data/users/kevihuang/projects/Open-Reasoner-Zero/venv/lib/python3.10/site-packages/deepspeed/runtime/zero/partitioned_param_coordinator.py
# # Fix DeepSpeed bug by copying patched files into your venv's deepspeed directory
# DEEPSPEED_DIR="$VIRTUAL_ENV/lib/python3.10/site-packages/deepspeed/runtime/zero"
# echo "Fixing DeepSpeed bug by copying parameter_offload.py and partitioned_param_coordinator.py to $DEEPSPEED_DIR..."
# if [ -f "./parameter_offload.py" ]; then
#     cp ./parameter_offload.py "$DEEPSPEED_DIR/parameter_offload.py"
# else
#     echo "parameter_offload.py not found; skipping."
# fi
# if [ -f "./partitioned_param_coordinator.py" ]; then
#     cp ./partitioned_param_coordinator.py "$DEEPSPEED_DIR/partitioned_param_coordinator.py"
# else
#     echo "partitioned_param_coordinator.py not found; skipping."
# fi

# echo "Changing working directory to /workspace/..."
# if [ -d "/workspace" ]; then
#     cd /workspace/
# else
#     echo "/workspace directory does not exist. Creating it..."
#     mkdir -p /workspace && cd /workspace/
# fi

# echo "Setup complete."
