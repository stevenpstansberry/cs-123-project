import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('chemical_compounds.csv')

# Drop the non-numeric 'PUBCHEM_COORDINATE_TYPE' column and 'CID' column
data = data.drop(['PUBCHEM_COORDINATE_TYPE', 'CID'], axis=1)

# Drop any NaN values
data = data.dropna()

# Split features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split the dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SVM classifier with a linear kernel
svm_classifier = SVC(kernel='linear')

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Print out results of predictions against the test set
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Class report: ")
print("--------------------------------------------------------")
print(class_report)

# Get the feature names
feature_names = X_train.columns

# Get the coefficients from SVM
coefficients = svm_classifier.coef_[0]

# Create a DataFrame object to hold features and coefficents
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the DataFrame by absolute value of coefficients
feature_importance_df['Absolute Coefficient'] = feature_importance_df['Coefficient'].abs()
feature_importance_df = feature_importance_df.sort_values(by='Absolute Coefficient', ascending=False)

# Print the top features
print('Top features influencing BRAF inhibition prediction:')
print(feature_importance_df[['Feature', 'Coefficient']].head(10))

print('\nLoading in new data set')

# Load the new dataset
new_data = pd.read_csv('new_chemical_compounds.csv')

# Drop non-numeric columns and handle missing values 
new_data = new_data.drop(['PUBCHEM_COORDINATE_TYPE', 'CID'], axis=1)
new_data = new_data.dropna()

# Feature Set
X_new = new_data
X_new_scaled = scaler.transform(X_new)  # set scalar

# Set target variable to predict
y_new_pred = svm_classifier.predict(X_new_scaled)

# Add predictions to the new dataset
new_data['Predicted_Class'] = y_new_pred

# Save the results as a csv file 
new_data.to_csv('predicted_braf_inhibitors.csv', index=False)

print('Predicated molecules identified and saved to csv!')