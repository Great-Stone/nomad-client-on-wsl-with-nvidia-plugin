datacenter = "dc1"
data_dir = "/var/nomad/data"
plugin_dir = "/etc/nomad.d/plugins"

bind_addr = "0.0.0.0"

advertise {
  http = "ADVERTISE_IP"
  rpc  = "ADVERTISE_IP"
  serf = "ADVERTISE_IP"
}

server {
  enabled = false
}

client {
  enabled = true
  servers = ["192.168.0.11:4647"]

  host_volume "host_temp" {
    path      = "/mnt/c/temp"
    read_only = false
  }
}

plugin "raw_exec" {
  config {
    enabled = true
  }
}

plugin "nomad-device-nvidia" {
  config {
    enabled            = true
    fingerprint_period = "1m"
  }
}