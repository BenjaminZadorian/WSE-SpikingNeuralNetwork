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
