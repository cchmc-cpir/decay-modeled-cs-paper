# decay-modeled-cs-paper
![Figure1](Fig1.png)

A repository that contains non-Cartesian reconstruction software for hyperpolarized xenon MRI. Written by Joseph Plummer (joeyplummer1@gmail.com, GitHub username: joeyplum). 

Please post all issues, questions, or collaboration requests on the Issues tab. Alternatively, feel free to contact me directly via email or other means. 

## Installation:

In a Linux, WSL2, or Mac terminal, run the following commands in sequence to install (it is recommended to set up `ssh` keys (see GitHub help or ask ChatGPT)).

1. `git clone git@github.com:cchmc-cpir/decay-modeled-cs-paper.git`
2. `cd decay-modeled-cs-paper`
3. `conda update -n base -c defaults conda`
4. `make conda`
5. `conda activate decay-modeled-cs-paper`
6. `make pip`

**Troubleshooting**:

1. This repository was tested on an NVIDIA GPU. If running on a system without
   the same, please remove the following packages from `environment.yaml`:
   - `cudnn`
   - `nccl`
   - `cupy`
2. Additionally, if not using an NVIDIA GPU, please set `devnum = -1` for each
   reconstruction script.

## Running the scripts: 

It is recommended to run all scripts using the `Run Current File in Interactive Window' tool in VScode so that a reconstructions can be monitored and figures can be easily viewed. However, the scripts also work in command line. 
1. simulation_recon_2d.py
2. ventilation_recon_2d.py
3. ventilation_recon_3d.py
4. gasex_recon_3d.py

## Uninstall:

To uninstall, run the following commands:

1. `conda activate`
2. `make clean`


## DOI:
https://doi.org/10.1002/mrm.30188
