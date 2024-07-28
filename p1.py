import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('Dataset10.csv')
print(data.head())

# Drop the 'Day' column
data = data.drop(['Day'], axis=1)
print(data.head())

# Extract attributes and target from the dataset
attribute = np.array(data)[:, :-1]
print("Attributes:\n", attribute)

target = np.array(data)[:, -1]
print("Target:\n", target)

# Define the training function
def train(att, tar):
    # Initialize specific hypothesis
    specific_h = None
    for i, val in enumerate(tar):
        if val == 'Yes':
            specific_h = att[i].copy()
            break

    # Generalize specific hypothesis based on positive examples
    for i, val in enumerate(att):
        if tar[i] == 'Yes':
            for x in range(len(specific_h)):
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'
                else:
                    pass
    return specific_h

# Train the model and print the result
print("Final Specific Hypothesis:\n", train(attribute, target))
