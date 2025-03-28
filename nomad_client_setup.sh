#!/bin/bash

echo "Advertise IP address is $1"

# Install the required packages
sudo apt-get update
sudo apt-get install -y unzip gcc

# Docker install
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Installing NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get install -y ubuntu-drivers-common nvidia-utils-545 nvidia-container-toolkit nvidia-cuda-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker info | grep -i runtime

wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run

sudo sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.8

# Create nomad config dir
sudo mkdir -p /etc/nomad.d/plugins

# Download the Nomad binary
cd /tmp
wget https://releases.hashicorp.com/nomad/1.9.7/nomad_1.9.7_linux_amd64.zip -O /tmp/nomad_1.9.7_linux_amd64.zip
wget https://releases.hashicorp.com/nomad-device-nvidia/1.1.0/nomad-device-nvidia_1.1.0_linux_amd64.zip -O /tmp/nomad-device-nvidia_1.1.0_linux_amd64.zip

unzip -o /tmp/nomad_1.9.7_linux_amd64.zip
unzip -o /tmp/nomad-device-nvidia_1.1.0_linux_amd64.zip
sudo mv /tmp/nomad /usr/local/bin/nomad
sudo mv /tmp/nomad-device-nvidia /etc/nomad.d/plugins/nomad-device-nvidia

nomad version

# Nomad configuration
sudo sed /mnt/c/temp/nomad.hcl.tpl -e "s/ADVERTISE_IP/$1/g" | sudo tee /etc/nomad.d/nomad.hcl > /dev/null

# Service setup
sudo cp /mnt/c/temp/nomad.service /etc/systemd/system/nomad.service
sudo systemctl daemon-reload
sudo systemctl enable nomad
sudo systemctl start nomad

