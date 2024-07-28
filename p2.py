import numpy as np
import pandas as pd

# Load the dataset
data = pd.DataFrame(data=pd.read_csv('ENJOYSPORT.csv'))

# Extract concepts and target from the dataset
concepts = np.array(data.iloc[:, 0:-1])
print("Concepts:\n", concepts)

target = np.array(data.iloc[:, -1])
print("Target:\n", target)

# Define the learning function
def learn(concepts, target):
    # Initialize specific hypothesis
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h:")
    print("Specific Hypothesis:\n", specific_h)
    
    # Initialize general hypothesis
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    print("General Hypothesis:\n", general_h)
    
    # Iterate through all concepts
    for i, h in enumerate(concepts):
        if target[i] == "yes":
            # Update specific hypothesis for positive examples
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
            print("\nSpecific Hypothesis after positive example", i + 1, ":\n", specific_h)
        if target[i] == "no":
            # Update general hypothesis for negative examples
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
            print("\nGeneral Hypothesis after negative example", i + 1, ":\n", general_h)
    
    # Remove redundant hypotheses
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    
    return specific_h, general_h

# Run the learning function
s_final, g_final = learn(concepts, target)
print("\nFinal Specific Hypothesis:\n", s_final)
print("\nFinal General Hypothesis:\n", g_final)
