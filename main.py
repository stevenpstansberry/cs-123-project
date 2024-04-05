import pandas as pd


# Load the dataset
data = pd.read_csv('chemical_compounds.csv')

#print (data.columns)

# Assuming the last column 'class' is the target variable
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

print(y)