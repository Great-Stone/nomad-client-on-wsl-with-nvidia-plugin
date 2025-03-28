job "gpu-python" {
  datacenters = ["dc1"]
  type = "batch"

  group "python" {

    task "pip" {
      driver = "raw_exec"

      lifecycle {
        hook = "prestart"
        sidecar = false
      }

      config {
        command = "pip.sh"
      }

      template {
        data = <<-EOF
        #!/bin/bash
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py
        # pip3 install torch numpy
        pip3 install virtualenv
        mkdir ../alloc/env
        virtualenv ../alloc/env
        EOF

        destination = "pip.sh"
      }
    }

    task "python-cuda" {
      driver = "raw_exec"

      config {
        command = "local/cuda.sh"
      }

      template {
        data = <<-EOF
        #!/bin/bash
        source ../alloc/env/bin/activate
        python3 -m pip install torch numpy
        python3 local/main.py
        EOF

        destination = "local/cuda.sh"
      }

      artifact {
        source = "https://raw.githubusercontent.com/Great-Stone/nomad-client-on-wsl-with-nvidia-plugin/refs/heads/main/python/main.py"
        destination = "local/"
      }

      resources {
        cpu = 5000
        memory = 2048
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

    task "model-copy-to-windows" {
      driver = "raw_exec"

      lifecycle {
        hook = "poststop"
        sidecar = false
      }

      config {
        command = "cp"
        args = ["${NOMAD_ALLOC_DIR}/model/simple_model.pth", "/mnt/c/temp/"]
      }
    }
  }
}