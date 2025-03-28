job "gpu-docker" {
  datacenters = ["dc1"]
  type = "batch"

  group "smi" {
    task "smi" {
      driver = "docker"

      config {
        image = "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"
        command = "nvidia-smi"
      }

      resources {
        device "nvidia/gpu" {
          count = 1

          # Add an affinity for a particular model
          affinity {
            attribute = "${device.model}"
            value     = "GeForce RTX 3060 Laptop GPU"
            weight    = 50
          }
        }
      }
    }
  }
}