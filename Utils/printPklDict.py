# converts a dictionary saved in .pkl format to spikes
import pickle
import sys

def printPklDict(fileName):
    file = open(fileName, 'rb')
    neoObj = pickle.load(file)
    for key, spikeData in neoObj.items():
        #print(f"{key}: {spikeData.segments[0].spiketrains}")
        segments = spikeData.segments
        segment = segments[0]
        spiketrains = segment.spiketrains
        neurons = len(spiketrains)
        #print(neurons)
        print(f"{key} Spikes:")
        for neuron_count in range(neurons):
            if (len(spiketrains[neuron_count])>0):
                spikes = spiketrains[neuron_count]
                for spike in range(len(spikes)):
                    print(neuron_count, spikes[spike])
    file.close()

args = sys.argv
numberArgs = args.__len__()
if (numberArgs == 2):
    fileName = args[1]
else:
    fileName = 'tempSpikes.pkl'

printPklDict(fileName)