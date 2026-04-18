import pyNN.nest as sim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# -- Constants --
EXPORT_DIR = "System_1_Results"

# Excitatory Weight Parameters
LETTER_RELATED_EXCITATION_WEIGHT = 0.025 # weight from features to related letters
WORD_RELATED_EXCITATION_WEIGHT = 0.0015 # weight from letters to words
TOP_DOWN_EXCITATION_WEIGHT = 0.0005
UNRELATED_EXCITATION_WEIGHT = 0.01

# Inhibitory Weight Parameters
LETTER_LATERAL_INHIBITORY_WEIGHT = 1.25
WORD_LATERAL_INHIBITORY_WEIGHT = 1.25
WORD_TO_LETTER_INHIBITORY_WEIGHT = 1.25

# System Size Parameters
LETTER_ASSEMBLY_SIZE = 100
WORD_ASSEMBLY_SIZE = 100
DELAY = 1.0
FEATURES_LENGTH = 9

# Depressing Synapse Parameters
UTILIZATION = 0.5
TAU_REC = 200
TAU_FACIL = 0

# Input Parameters
DURATION = 1000.0
INPUT_WEIGHT = 0.25
INPUT_RATE = 50.0
INPUT_DURATION = 300.0

# Recurrent Parameters
LETTER_RECURRENT_WEIGHT = 0.0021
WORD_RECURRENT_WEIGHT = 0.0003
LETTER_RECURRENT_PROBABILITY = 0.25
WORD_RECURRENT_PROBABILITY = 0.10

FEATURES = ['horizontal',   #0
            'vertical',     #1
            'curve',        #2
            'intersection', #3 
            'closed',       #4
            'diagonal',     #5
            'symmetrical',  #6
            'open_top',     #7
            'junction']     #8

LETTERS = ['W', 'O', 'R', 'K', 'D', 'E', 'F', 'T', 'L', 'H', 'A']

LEXICON = ['WORK', 'WORD', 'FOLK', 'HELD', 'WOKE', 'FORT', 'ROLE', 'LORD']

# plot histogram of spikes overtime
def printResultsHistogram(spike_data, title, bin_size=25.0):
    segment = spike_data.segments[0]
    spiketrains = segment.spiketrains
    
    all_spikes = np.concatenate([st.magnitude for st in spiketrains])

    bins = np.arange(0, DURATION + bin_size, bin_size)
    hist, bin_edges = np.histogram(all_spikes, bins=bins)
    
    plt.figure(figsize=(12, 6))
    plt.bar(bin_edges[:-1], hist, width=bin_size*0.9, color='skyblue', edgecolor='black')
    
    plt.axvline(x=INPUT_DURATION, color='red', linestyle='--', linewidth=2, label=f"Input OFF ({INPUT_DURATION} ms)")
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Total Spikes per Bin (25ms)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plot results to a graph
def printResultsGraph(rates, active_features, label):
    fig, ax = plt.subplots(figsize=(10, 6))

    letters_list = list(rates.keys())
    ax.bar(letters_list, list(rates.values()), 
        color=['red', 'orange', 'green', 'blue'], 
        edgecolor='black', linewidth=1.5)

    # Add labels to the graph
    for i, (letter, rate) in enumerate(rates.items()):
        ax.text(i, rate + 0.2, f'{rate} Hz', ha='center', fontsize=11)

    ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax.set_title(f'{label} Assembly Firing Rates\n(active_features = ' + str(active_features) + ')', fontsize=14)
    ax.set_ylim(0, max(rates.values()) * 1.3)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

# Export results to .pkl file
def exportResults(spike_data, label):

    # check if directory already exists, if not then create it
    if not os.path.isdir(EXPORT_DIR):
        os.mkdir(EXPORT_DIR)
    
    with open(f"{EXPORT_DIR}/occlusion_experiment_{label}.pkl", "wb") as f:
        pickle.dump(spike_data, f, protocol=pickle.HIGHEST_PROTOCOL)

# Print results
def getResults(results):
    print(f"Results: {DURATION}ms Simulation: ")
    for test, result in results.items():
        print(f"{test}: {result}")

# Get spike data from recordings
def getSpikeData(lexicon, letters, feature_assembly):

    feature_spike_data = feature_assembly.get_data()
    
    letter_spike_data = {}
    letter_rates = {}
    for letter_label, letter_assembly in letters.items():
        data = letter_assembly.get_data()
        spiketrain = data.segments[0].spiketrains
        spike_count = [len(spikes) for spikes in spiketrain]
        rate = np.mean(spike_count)
        letter_rates[letter_label] = rate
        letter_spike_data[letter_label] = data

    word_spike_data = {}
    word_rates = {}
    for word_label, word_assembly in lexicon.items():
        data = word_assembly.get_data()
        spiketrain = data.segments[0].spiketrains
        spike_count = [len(spikes) for spikes in spiketrain]
        rate = np.mean(spike_count)
        word_rates[word_label] = rate
        word_spike_data[word_label] = data

    
    
    return letter_spike_data, letter_rates, word_spike_data, word_rates, feature_spike_data

# Connect Poisson input
def connectInput(active_features, feature_assembly):
    poisson_input = sim.Population(1, sim.SpikeSourcePoisson(rate=INPUT_RATE, duration=INPUT_DURATION))

    input_connections = []
    for feature in active_features:
        input_connections.append((0, feature, INPUT_WEIGHT, DELAY))

    sim.Projection(poisson_input, feature_assembly, sim.FromListConnector(input_connections), receptor_type='excitatory')

# Create interconnected synapses within each letter & word assembly
def createInternalConnections(letters, lexicon):
    # Create connections within letters
    for assembly in letters.values():
        sim.Projection(assembly, assembly, sim.FixedProbabilityConnector(p_connect = LETTER_RECURRENT_PROBABILITY), synapse_type=sim.StaticSynapse(weight = LETTER_RECURRENT_WEIGHT, delay = DELAY))

    for assembly in lexicon.values():
        sim.Projection(assembly, assembly, sim.FixedProbabilityConnector(p_connect = WORD_RECURRENT_PROBABILITY), synapse_type=sim.StaticSynapse(weight = WORD_RECURRENT_WEIGHT, delay = DELAY))

# Create inhibitory connections between words and unrelated letters
def createInhibitoryWordToLetterConnections(letter_connections, lexicon, letters):
    for word_label, related_letters in letter_connections.items():
        word_assembly = lexicon[word_label]
        unrelated_connections = []
        unrelated_letter_list = []
        for letter in letters.keys():
            if letter not in related_letters:
                unrelated_letter_list.append(letter)

        for unrelated_letter in unrelated_letter_list:
            letter_assembly = letters[unrelated_letter]
            for word_neuron in range(WORD_ASSEMBLY_SIZE):
                for letter_neuron in range(LETTER_ASSEMBLY_SIZE):
                    unrelated_connections.append((word_neuron, letter_neuron, WORD_TO_LETTER_INHIBITORY_WEIGHT, DELAY))
            sim.Projection(word_assembly, letter_assembly, sim.FromListConnector(unrelated_connections), receptor_type='inhibitory')

# Create Inhibitory connections between the words (adding competition between words)
def createInhibitoryWordConnections(lexicon):
    for source_word in lexicon.keys():
        for target_word in lexicon.keys():
            if source_word != target_word:
                source_assembly = lexicon[source_word]
                target_assembly = lexicon[target_word]

                sim.Projection(source_assembly, target_assembly, sim.AllToAllConnector(), sim.StaticSynapse(weight=WORD_LATERAL_INHIBITORY_WEIGHT), receptor_type='inhibitory')

# Create Excitatory connections from letters to words
def createLetterWordConnections(letter_connections, lexicon, letters, use_word_context):
    for word_label, related_letters in letter_connections.items():
        related_excitation_connections = []
        word_assembly = lexicon[word_label]

        for related_letter in related_letters:
            letter_assembly = letters[related_letter]
            for word_neuron in range(WORD_ASSEMBLY_SIZE):
                for letter_neuron in range(LETTER_ASSEMBLY_SIZE):
                    if use_word_context == True:
                        related_excitation_connections.append((letter_neuron, word_neuron, TOP_DOWN_EXCITATION_WEIGHT, DELAY))
                    else:
                        related_excitation_connections.append((letter_neuron, word_neuron, WORD_RELATED_EXCITATION_WEIGHT, DELAY))
            # letter -> word connection
            sim.Projection(letter_assembly, word_assembly, sim.FromListConnector(related_excitation_connections), receptor_type='excitatory')

            if use_word_context == True:
                
                depressing_synapse = sim.TsodyksMarkramSynapse(
                    weight = TOP_DOWN_EXCITATION_WEIGHT,
                    delay = DELAY,
                    U = UTILIZATION,
                    tau_rec = TAU_REC,
                    tau_facil = TAU_FACIL
                )
                sim.Projection(word_assembly, letter_assembly, sim.FromListConnector(related_excitation_connections), synapse_type=depressing_synapse,receptor_type='excitatory')

                # word -> letter connection
                #sim.Projection(word_assembly, letter_assembly, sim.FromListConnector(related_excitation_connections), receptor_type='excitatory')
                
# Create Inhibitory connections between letters
def createInhibitoryFeatureConnections(letters):
    for source_letter in letters.keys():
        for target_letter in letters.keys():
            if source_letter != target_letter:
                source_assembly = letters[source_letter]
                target_assembly = letters[target_letter]

                sim.Projection(source_assembly, target_assembly, sim.AllToAllConnector(), sim.StaticSynapse(weight=LETTER_LATERAL_INHIBITORY_WEIGHT), receptor_type='inhibitory')        

# Create Excitatory Connections between Letters (Un-related Features)
def createUnrelatedFeatureConnections(feature_connections, letters, feature_assembly):
    
    for letter_label, related_features in feature_connections.items():
        letter_assembly = letters[letter_label]
        unrelated_excitation_connections = []
        unrelated_feature_list = []
        for feature in range(FEATURES_LENGTH):
            if feature not in related_features:
                unrelated_feature_list.append(feature)

        for unrelated_feature in unrelated_feature_list:
            for letter_neuron in range(LETTER_ASSEMBLY_SIZE):
                unrelated_excitation_connections.append((unrelated_feature, letter_neuron, UNRELATED_EXCITATION_WEIGHT, DELAY))
                
        # Add that projection containing all unrelated connections
        sim.Projection(feature_assembly, letter_assembly, sim.FromListConnector(unrelated_excitation_connections), receptor_type='excitatory')

# Create Excitatory Connections from features to letters (Related Features)
def createRelatedFeatureConnections(feature_connections, letters, feature_assembly):
    
    for letter_label, related_features in feature_connections.items():
        related_excitation_connections = []
        letter_assembly = letters[letter_label]

        # loop through related features
        for related_feature in related_features:
            # loop through all assembly neurons
            for letter_neuron in range(LETTER_ASSEMBLY_SIZE):
                related_excitation_connections.append((related_feature, letter_neuron, LETTER_RELATED_EXCITATION_WEIGHT, DELAY))

        # Add that projection containing all related_feature projections
        sim.Projection(feature_assembly, letter_assembly, sim.FromListConnector(related_excitation_connections), receptor_type='excitatory')

# Defines Letters -> Word connections
def defineLetterConnections():
    letter_connections = {
        'WORK': ['W', 'O', 'R', 'K'],
        'WORD': ['W', 'O', 'R', 'D'],
        'FOLK': ['F', 'O', 'L', 'K'],
        'HELD': ['H', 'E', 'L', 'D'],
        'WOKE': ['W', 'O', 'K', 'E'],
        'FORT': ['F', 'O', 'R', 'T'],
        'ROLE': ['R', 'O', 'L', 'E'],
        'LORD': ['L', 'O', 'R', 'D']
    }
    return letter_connections

# Define Feature -> Letter Connections
def defineFeatureConnections():
    
    feature_connections = {
        'W': [1, 5, 6, 7],      # vertical - diagonal - symmetrical - open_top
        'O': [2, 4, 6],         # curved - closed - symmetrical
        'R': [0, 1, 2, 3],      # horizonal - vertical - curved - intersection
        'K': [1, 3, 5, 8],         # vertical - intersection - diagonal - junction
        'D': [1, 2, 4, 5],      # vertical - curved - closed - diagonal
        'E': [0, 1, 8],         # horizontal - vertical - junction
        'F': [0, 1, 7, 8],      # horizontal - vertical - open_top - junction
        'T': [0, 1, 6, 8],      # horizontal - vertical - symmetrical - junction
        'L': [0, 1],            # horizontal - vertical
        'H': [1, 3, 6],         # vertical - intersection - symmetrical
        'A': [0, 3, 5, 6, 8]    # horizontal - intersection - diagonal - symmetrical - junction
    }
    return feature_connections

# Create lexicon & record spikes of words
def createLexicon(lexicon):
    lexicon_assemblies = {}

    for word in lexicon:
        lexicon_assemblies[word] = sim.Population(WORD_ASSEMBLY_SIZE, sim.IF_cond_exp(), label=word)

    for population in lexicon_assemblies.values():
        population.record(['spikes'])

    return lexicon_assemblies

# Create Letter Assemblies & Record letter spikes - Layer 2
def createLetters(letters):

    letters_assemblies = {}

    for letter in letters:
        letters_assemblies[letter] = sim.Population(LETTER_ASSEMBLY_SIZE, sim.IF_cond_exp(), label=letter)

    for population in letters_assemblies.values():
        population.record(['spikes'])
    
    return letters_assemblies

# Create Wickelfeatures & Wickelfeature assembly - Layer 1
def createWicklefeatures(features):
    feature_assembly = sim.Population(len(features), sim.IF_cond_exp(), label='features')
    feature_assembly.record(['spikes'])
    return feature_assembly

def main():

    # Choose what active features to input
    # -- Active Features Guide --
    #           W = 1, 5
    #           O = 2, 4
    #           R = 0, 1, 2, 3
    #           K = 1, 3, 5
    #           D = 1, 2, 4
    # the more features are added, the more specific the guess becomes due to competition

        # Title - Occluded Active Features - Top Down Context Active?
    occlusion_test = [
        
        ("Isolated_D",              [1, 4],     False),
        #("Context_with_D",          [1, 4],     True),
        #("Isolated_K",              [1, 3, 5],  False),
        #("Context_with_K",          [1, 3, 5],  True),
        #("Isolated_A",              [0, 3, 6],  False),
        #("Unknown_A",               [0, 3, 6],  True),
        #("Isolated_O",              [2, 4],     False),
        #("Control_O",               [2, 4],     True)
    ]

    rates_results = {}
    spikes_results = {}

    for occlusion_test_label, active_features, use_word_context in occlusion_test:
        print(f"Occlusion Test: {occlusion_test_label}")
        
        # Setup
        sim.setup(timestep=0.1)

        # Create assembly of all features
        feature_assembly = createWicklefeatures(FEATURES)

        # Create letter assemblies
        letters = createLetters(LETTERS)

        # Create word assemblies
        lexicon = createLexicon(LEXICON)

        # Define related feature -> letter connections
        feature_connections = defineFeatureConnections()

        # Define letter -> word connections
        letter_connections = defineLetterConnections()

        # Create related feature connections
        createRelatedFeatureConnections(feature_connections, letters, feature_assembly)

        # Create unrelated feature connections
        createUnrelatedFeatureConnections(feature_connections, letters, feature_assembly)

        # Create Inhibitory connections between letters
        createInhibitoryFeatureConnections(letters)

        # Create bi-directional connections between letters & words
        createLetterWordConnections(letter_connections, lexicon, letters, use_word_context)

        # Create inhibitory connections between words to add competition
        createInhibitoryWordConnections(lexicon)

        # Create inhibitory connections between words and unrelated letters
        createInhibitoryWordToLetterConnections(letter_connections, lexicon, letters)

        # Create recurrent internal connections within letter & word assemblies
        createInternalConnections(letters, lexicon)

        # Loop through all active features and connect those with a Poisson Source
        connectInput(active_features, feature_assembly)

        # Run simulation
        sim.run(DURATION)

        # Get Spike data from all letters
        letter_spike_data, letter_rates, word_spike_data, word_rates, feature_spike_data = getSpikeData(lexicon, letters, feature_assembly)

        # --------------------------------------------------------------------
        # Comment out to avoid making graphs

        # printResultsHistogram(feature_spike_data, f"{occlusion_test_label} - Feature Layer Histogram")
        # for letter in letters.keys():
        #     printResultsHistogram(letter_spike_data[letter], f"{occlusion_test_label} - {letter} - Letter Assembly Histogram")
        # for word in lexicon.keys():
        #     printResultsHistogram(word_spike_data[word], f"{occlusion_test_label} - {word} - Word Layer Histogram")
        # --------------------------------------------------------------------

        # Store results in a dictionary
        rates_results[occlusion_test_label] = {"letters": letter_rates, "words": word_rates}
        spikes_results[occlusion_test_label] = {}
        
        # export data to .pkl file
        exportResults(letter_spike_data, f"{occlusion_test_label}_letter_data")
        exportResults(word_spike_data, f"{occlusion_test_label}_word_data")
        exportResults(feature_spike_data, f"{occlusion_test_label}_feature_data")

        # End simulation
        sim.end()

    print("Results")
    getResults(rates_results)

    # Print letter results graph
    printResultsGraph(letter_rates, active_features, "Letters")

    # Print word results graph
    printResultsGraph(word_rates, active_features, "Words")

# -- Run Program --
main()
