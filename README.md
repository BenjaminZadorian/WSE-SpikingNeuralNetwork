# Word Superiority Effect Experiments Implemented using a Spiking Neural Network
A SNN network used to experiment with Word Superiority Effect research.
> *These two systems are different architectures of an SNN network that conduct an occlusion experiment & a sequential letter presentation experiment relating to the WSE.*

This project was created as a Final Year Undergraduate project.  Using the PyNN & NEST libraries for Python, I worked on created a spiking neural network architecture that can replicate the conditions in the Rumelhart and McClelland Occlusion Experiment relating to WSE.

## Overview
Main Features:
- Two Systems (System 1 with a shared wickelfeature assembly for all letter assemblies, System 2 with each letter assembly having a personal wickelfeature assembly)
- Able to conduct 2 experiment: Letter Occlusion (features within the letter are obscured), and Sequential Letter Presentation (showing the letters in a word at sequential time steps)
- Plots the spike data and firing rates to histograms and bar charts respectively
- Exports the spike data to .pkl files, which can be read with files in the **Utils** directory

## Requirements
- Python 3.8 or later
- Linux or macOS (NEST does not support Windows natively)

### Installation (Linux)
1. Install Python
```py
sudo apt install python3-pip
```
2. Create a virtual environment - example directory name 'iac_model_env' (optional)
```py
apt install python3.[X]-venv
python3 -m venv iac_model_env
source iac_model_env/bin/activate
```
3. Get PPA Repository for NEST & Install NEST Simulator
```py
sudo add-apt-repository ppa:nest-simulator/nest
sudo apt-get update
sudo apt-get install nest
```
4. Install PyNN
```py
pip install PyNN
```
5. Install Other Dependancies
```py
pip install numpy matplotlib
```

## Usage Instructions
- Use RUN_OCCLUSION_TEST & RUN_SEQUENTIAL_TEST to control what tests are ran
```py
RUN_OCCLUSION_TEST = False
RUN_SEQUENTIAL_TEST = True
```
- Choose what letter features are activated in either 'occlusion_test' or 'sequential_test'
```py
# Test Title | Active features of the letter | If contextual feedback is enabled
    occlusion_test = [  
        ("Isolated_D", [1, 4], False),
    ]
```
- Alternatively, uncomment one of the preset experiments within the 'occusion_test' or 'sequential_test' arrays
  
## ToDo:
1. Add either CLI or GUI interface to make usage smoother
2. Remove Tsodyks-Markram synapses and try out adaptive neurons to fix runaway excitation
3. Add more varying features/letters/word
4. Recreate other WSE experiments
