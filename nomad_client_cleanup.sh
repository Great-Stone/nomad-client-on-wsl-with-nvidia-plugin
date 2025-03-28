#!/bin/bash

sudo systemctl daemon-reload
sudo systemctl stop nomad

sudo rm -rf /etc/systemd/system/nomad.service
sudo rm -rf /var/nomad
sudo rm -rf /etc/nomad.d
sudo rm -rf /usr/local/bin/nomad
sudo rm -rf /tmp/nomad*