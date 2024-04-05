import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
data = pd.read_csv('chemical_compounds.csv')

# Find non-numeric columns
non_numeric_columns = data.columns[(data.dtypes == 'object').values].tolist()

# Print non-numeric columns
print(f"Non-numeric columns: {non_numeric_columns}")

# Drop the non-numeric 'PUBCHEM_COORDINATE_TYPE' column
data = data.drop('PUBCHEM_COORDINATE_TYPE', axis=1)


# Check for NaN values in the dataset
nan_values = data.isna()

# Print rows with NaN values
print("Rows with NaN values:")
print(data[nan_values.any(axis=1)])

# Drop any NaN values
# TODO check to see if there are any better ways to fix NaN values rather than just dropping
data = data.dropna()

# Find non-numeric columns
non_numeric_columns = data.columns[(data.dtypes == 'object').values].tolist()

# Print non-numeric columns
print(f"Non-numeric columns: {non_numeric_columns}")

print("\n \n \n \n \n ")


X = data.drop('Class', axis=1)  # Features of target variable
y = data['Class']  # Target variable

# Split the dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initalize SVM classifer with linear kernal
svm_classifier = SVC(kernel='linear')

# Train the classifier on the data
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


print("Accuracy: ", accuracy ,"\n")
#print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Class report: ")
print("--------------------------------------------------------")
print(class_report)